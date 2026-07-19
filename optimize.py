#!/usr/bin/env python3
"""
Strategy Parameter Optimizer v2 — walk-forward + Monte Carlo + multi-objective.

Upgrades over v1:
    1. Multi-objective scoring (PF, drawdown, Sharpe, win rate, expectancy)
    2. Walk-forward validation to prevent overfitting
    3. Monte Carlo robustness testing
    4. Top-N consensus selection for stable params
    5. Detailed reporting with JSON export

Usage:
    python optimize.py                                        # Basic single-split
    python optimize.py --walk-forward                        # Walk-forward 4 folds
    python optimize.py --walk-forward --monte-carlo 1000     # + Monte Carlo
    python optimize.py --days 730 --walk-forward --report    # Full research run
    python optimize.py --instrument EUR_JPY --monte-carlo 500
"""
import sys, os, logging, time, argparse, json
import numpy as np
import pandas as pd
import ta
from itertools import product
from collections import Counter

os.makedirs("data", exist_ok=True)
logging.basicConfig(level=logging.WARNING, format="%(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler("data/optimize.log")])
logger = logging.getLogger("Optimizer")
logger.setLevel(logging.INFO)

from backtesting.data_loader import fetch_oanda_historical
from strategy.strategies import ALL_STRATEGIES
from strategy.strategy_selector import STRATEGY_CATEGORY, CATEGORY_WEIGHT
from config.settings import INSTRUMENTS, WATCHLIST


# ===== GRIDS =====

ENTRY_GRID = {
    "min_agreeing": [3, 4, 5],
    "min_categories": [2, 3],
}

EXIT_GRID = {
    "tp1_r": [1.0, 1.5, 2.0],
    "tp1_pct": [0.33, 0.50],
    "adverse_r": [0.4, 0.5, 0.6],
    "adverse_bars": [2, 3, 4],
    "time_stop_bars": [15, 20, 30],
    "time_stop_r": [0.2, 0.3],
}

CURRENT_DEFAULTS = {
    "min_agreeing": 5, "min_categories": 3,
    "tp1_r": 1.5, "tp1_pct": 0.33,
    "adverse_r": 0.6, "adverse_bars": 3,
    "time_stop_bars": 30, "time_stop_r": 0.3,
}


# ===== MULTI-OBJECTIVE SCORING =====

def compute_equity_metrics(trades, balance=100):
    if not trades:
        return {"max_dd_pct": 100, "sharpe": 0, "sortino": 0,
                "final_balance": balance, "return_pct": 0}

    equity = balance
    peak = balance
    max_dd = 0
    returns = []

    for pnl in trades:
        equity += pnl
        ret = pnl / peak if peak > 0 else 0
        returns.append(ret)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    returns = np.array(returns)
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    downside = returns[returns < 0]
    sortino = (np.mean(returns) / np.std(downside) * np.sqrt(252)) if len(downside) > 0 and np.std(downside) > 0 else 0

    return {
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "final_balance": round(equity, 2),
        "return_pct": round((equity - balance) / balance * 100, 2),
    }


def score_trades(trades, balance=100):
    """
    Multi-objective scoring.
    PF (40%) + MaxDD (25%) + Sharpe (15%) + WinRate (10%) + Expectancy (10%).
    """
    if len(trades) < 5:
        return {"pnl": sum(trades) if trades else 0, "n": len(trades),
                "wr": 0, "pf": 0, "aw": 0, "al": 0, "score": -9999,
                "max_dd_pct": 0, "sharpe": 0, "sortino": 0, "expectancy": 0}

    pnl = sum(trades)
    w = [t for t in trades if t > 0]
    lo = [t for t in trades if t <= 0]
    wr = len(w) / len(trades) * 100
    pf = abs(sum(w) / sum(lo)) if lo and sum(lo) != 0 else 0
    aw = np.mean(w) if w else 0
    al = np.mean(lo) if lo else 0
    expectancy = pnl / len(trades)

    eq = compute_equity_metrics(trades, balance)

    # Normalize to 0-1
    pf_score = min(pf / 2.0, 1.0)
    dd_score = max(1 - eq["max_dd_pct"] / 50, 0)
    sr_score = min(max(eq["sharpe"], 0) / 3.0, 1.0)
    wr_score = wr / 100.0
    exp_score = min(max(expectancy, 0) / 5.0, 1.0)

    score = (0.40 * pf_score + 0.25 * dd_score + 0.15 * sr_score +
             0.10 * wr_score + 0.10 * exp_score)

    trade_penalty = min(len(trades) / 20, 1.0)
    score *= trade_penalty

    if pnl > 0 and pf > 1.2:
        score *= 1.1

    return {
        "pnl": round(pnl, 2), "n": len(trades), "wr": round(wr, 1),
        "pf": round(pf, 2), "aw": round(aw, 2), "al": round(al, 2),
        "score": round(score, 4),
        "max_dd_pct": eq["max_dd_pct"], "sharpe": eq["sharpe"],
        "sortino": eq["sortino"], "expectancy": round(expectancy, 2),
    }


# ===== MONTE CARLO ROBUSTNESS =====

def monte_carlo_test(trades, n_sims=1000, balance=100):
    """Shuffle trade order to test if results depend on lucky sequencing."""
    if len(trades) < 10:
        return {"robust": False, "reason": "Too few trades", "sims": 0}

    trades_arr = np.array(trades)
    final_balances = np.zeros(n_sims)
    max_drawdowns = np.zeros(n_sims)
    ruin_count = 0

    for s in range(n_sims):
        shuffled = trades_arr.copy()
        np.random.shuffle(shuffled)

        equity = balance
        peak = balance
        max_dd = 0

        for pnl in shuffled:
            equity += pnl
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            if equity <= 0:
                ruin_count += 1
                break

        final_balances[s] = equity
        max_drawdowns[s] = max_dd

    return {
        "sims": n_sims,
        "median_final": round(float(np.median(final_balances)), 2),
        "p5_final": round(float(np.percentile(final_balances, 5)), 2),
        "p25_final": round(float(np.percentile(final_balances, 25)), 2),
        "p75_final": round(float(np.percentile(final_balances, 75)), 2),
        "p95_final": round(float(np.percentile(final_balances, 95)), 2),
        "median_dd": round(float(np.median(max_drawdowns)), 1),
        "p95_dd": round(float(np.percentile(max_drawdowns, 95)), 1),
        "ruin_pct": round(ruin_count / n_sims * 100, 2),
        "robust": float(np.percentile(final_balances, 25)) > balance,
        "profit_probability": round(float(np.mean(final_balances > balance) * 100), 1),
    }


# ===== CORE FUNCTIONS =====

def detect_regime(adx_val, bb_width_val, rsi_val):
    if adx_val > 25:
        return "trending"
    elif bb_width_val < 0.01:
        return "ranging"
    elif bb_width_val > 0.03:
        return "volatile"
    return "transitioning"


def precompute_votes(data, lookback=200):
    close, high, low = data["close"], data["high"], data["low"]
    adx = ta.trend.adx(high, low, close, window=14)
    bb_w = ta.volatility.BollingerBands(close, window=20, window_dev=2).bollinger_wband()
    rsi = ta.momentum.rsi(close, window=14)
    atr = ta.volatility.average_true_range(high, low, close, window=14)

    results = []
    for i in range(lookback, len(data)):
        window = data.iloc[max(0, i - lookback):i + 1]
        regime = detect_regime(
            adx.iloc[i] if not pd.isna(adx.iloc[i]) else 0,
            bb_w.iloc[i] if not pd.isna(bb_w.iloc[i]) else 0.02,
            rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50,
        )
        votes = []
        for strat in ALL_STRATEGIES:
            try:
                r = strat.evaluate(window, regime)
                if r["signal"] != "SKIP":
                    votes.append({
                        "strategy": strat.name,
                        "signal": r["signal"],
                        "raw_confidence": r["confidence"],
                        "category": STRATEGY_CATEGORY.get(strat.name, "unknown"),
                    })
            except Exception:
                pass

        results.append({
            "bar_idx": i,
            "votes": votes,
            "close": float(data.iloc[i]["close"]),
            "high": float(data.iloc[i]["high"]),
            "low": float(data.iloc[i]["low"]),
            "open_": float(data.iloc[i]["open"]),
            "atr": float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0,
            "regime": regime,
        })

    return results


def apply_entry_filter(precomputed, min_agreeing, min_categories):
    signals = {}
    for item in precomputed:
        if not item["votes"]:
            continue
        buy_v = [v for v in item["votes"] if v["signal"] == "BUY"]
        sell_v = [v for v in item["votes"] if v["signal"] == "SELL"]

        for direction, dv in [("BUY", buy_v), ("SELL", sell_v)]:
            if len(dv) < min_agreeing:
                continue
            cats = set(v["category"] for v in dv)
            if len(cats) < min_categories:
                continue
            n = len(dv)
            avg_raw = sum(v["raw_confidence"] for v in dv) / n
            max_raw = max(v["raw_confidence"] for v in dv)
            if avg_raw < 0.35:
                continue
            base = avg_raw * 0.5 + max_raw * 0.3
            cb = {1: 0, 2: 0.05, 3: 0.15, 4: 0.25, 5: 0.35, 6: 0.42, 7: 0.48}
            diversity = cb.get(len(cats), 0.48)
            wt = sum(CATEGORY_WEIGHT.get(c, 0.05) for c in cats)
            wq = min(wt / 0.80, 1.0) * 0.10
            vb = max(0, (n - min_agreeing)) * 0.02
            conf = round(min(base + diversity + wq + vb, 0.95), 4)
            if conf < 0.70:
                continue
            signals[item["bar_idx"]] = {
                "direction": direction, "confidence": conf, "atr": item["atr"],
            }
            break
    return signals


def simulate(data, signals, instrument, exit_params):
    spec = INSTRUMENTS.get(instrument, INSTRUMENTS["EUR_USD"])
    pv = 10 ** spec["pip_location"]
    sc = spec["spread_avg"] * pv / 2
    units = 1000

    tp1_r = exit_params["tp1_r"]
    tp1_pct = exit_params["tp1_pct"]
    ar = exit_params["adverse_r"]
    ab = exit_params["adverse_bars"]
    tsb = exit_params["time_stop_bars"]
    tsr = exit_params["time_stop_r"]

    atr_s = ta.volatility.average_true_range(
        data["high"], data["low"], data["close"], window=14)

    trades = []
    pos = None

    for i in range(len(data)):
        h = float(data.iloc[i]["high"])
        l = float(data.iloc[i]["low"])
        c = float(data.iloc[i]["close"])

        if pos:
            held = i - pos["eb"]
            if pos["d"] == "BUY":
                if h > pos["bh"]: pos["bh"] = h
            else:
                if l < pos["bl"]: pos["bl"] = l

            if (pos["d"] == "BUY" and l <= pos["sl"]) or \
               (pos["d"] == "SELL" and h >= pos["sl"]):
                p = (pos["sl"] - pos["ep"]) * pos["u"] if pos["d"] == "BUY" \
                    else (pos["ep"] - pos["sl"]) * pos["u"]
                trades.append(p)
                pos = None
                continue

            if held <= ab and not pos.get("be"):
                adv_p = (pos["ep"] - l) / pv if pos["d"] == "BUY" else (h - pos["ep"]) / pv
                if adv_p >= pos["rp"] * ar:
                    p = (c - pos["ep"]) * pos["u"] if pos["d"] == "BUY" \
                        else (pos["ep"] - c) * pos["u"]
                    trades.append(p)
                    pos = None
                    continue

            if held >= tsb and not pos.get("t1"):
                pp = (c - pos["ep"]) / pv if pos["d"] == "BUY" else (pos["ep"] - c) / pv
                if pp < pos["rp"] * tsr:
                    trades.append(pp * pv * pos["u"])
                    pos = None
                    continue

            if not pos.get("t1"):
                if (pos["d"] == "BUY" and h >= pos["t1p"]) or \
                   (pos["d"] == "SELL" and l <= pos["t1p"]):
                    tu = int(units * tp1_pct)
                    if tu > 0:
                        p = (pos["t1p"] - pos["ep"]) * tu if pos["d"] == "BUY" \
                            else (pos["ep"] - pos["t1p"]) * tu
                        trades.append(p)
                        pos["u"] -= tu
                    pos["t1"] = True
                    pos["be"] = True
                    pos["sl"] = pos["ep"]

            if pos and pos.get("t1") and not pos.get("t2"):
                if (pos["d"] == "BUY" and h >= pos["t2p"]) or \
                   (pos["d"] == "SELL" and l <= pos["t2p"]):
                    tu = int(units * 0.25)
                    if tu > 0 and pos["u"] > tu:
                        p = (pos["t2p"] - pos["ep"]) * tu if pos["d"] == "BUY" \
                            else (pos["ep"] - pos["t2p"]) * tu
                        trades.append(p)
                        pos["u"] -= tu
                    pos["t2"] = True

            if pos and pos.get("be") and i < len(atr_s) and not pd.isna(atr_s.iloc[i]):
                a = float(atr_s.iloc[i])
                m = 1.5 if pos.get("t2") else 2.0
                if pos["d"] == "BUY":
                    nt = pos["bh"] - a * m
                    if nt > pos["sl"]: pos["sl"] = nt
                else:
                    nt = pos["bl"] + a * m
                    if nt < pos["sl"]: pos["sl"] = nt

        if not pos and i in signals and i + 1 < len(data):
            sig = signals[i]
            ep = float(data.iloc[i + 1]["open"])
            ep = ep + sc if sig["direction"] == "BUY" else ep - sc
            atr_val = sig["atr"]
            rp = max(atr_val / pv * 2.0, 15) if atr_val > 0 else 30
            if sig["direction"] == "BUY":
                sl = ep - rp * pv
                t1p = ep + rp * tp1_r * pv
                t2p = ep + rp * 2.5 * pv
            else:
                sl = ep + rp * pv
                t1p = ep - rp * tp1_r * pv
                t2p = ep - rp * 2.5 * pv
            pos = {"d": sig["direction"], "ep": ep, "sl": sl,
                   "t1p": t1p, "t2p": t2p, "eb": i + 1, "rp": rp,
                   "u": units, "bh": ep, "bl": ep}

    if pos:
        c = float(data.iloc[-1]["close"])
        p = (c - pos["ep"]) * pos["u"] if pos["d"] == "BUY" \
            else (pos["ep"] - c) * pos["u"]
        trades.append(p)

    return trades


# ===== GRID SEARCH =====

def grid_search(votes, data, instrument, balance=100, top_n=20):
    """Search all entry+exit combos, return top N by multi-objective score."""
    best_entry_score = -99999
    best_ma, best_mc = 5, 3

    for ma, mc in product(ENTRY_GRID["min_agreeing"], ENTRY_GRID["min_categories"]):
        sigs = apply_entry_filter(votes, ma, mc)
        default_exit = {k: CURRENT_DEFAULTS[k] for k in
                        ["tp1_r", "tp1_pct", "adverse_r", "adverse_bars",
                         "time_stop_bars", "time_stop_r"]}
        trades = simulate(data, sigs, instrument, default_exit)
        m = score_trades(trades, balance)
        if m["score"] > best_entry_score:
            best_entry_score = m["score"]
            best_ma, best_mc = ma, mc

    sigs = apply_entry_filter(votes, best_ma, best_mc)
    combos = list(product(
        EXIT_GRID["tp1_r"], EXIT_GRID["tp1_pct"],
        EXIT_GRID["adverse_r"], EXIT_GRID["adverse_bars"],
        EXIT_GRID["time_stop_bars"], EXIT_GRID["time_stop_r"],
    ))

    results = []
    for tp1r, tp1p, a_r, a_b, ts_b, ts_r in combos:
        ep = {"tp1_r": tp1r, "tp1_pct": tp1p, "adverse_r": a_r,
              "adverse_bars": a_b, "time_stop_bars": ts_b, "time_stop_r": ts_r}
        trades = simulate(data, sigs, instrument, ep)
        m = score_trades(trades, balance)
        param_key = f"{best_ma}_{best_mc}_{tp1r}_{tp1p}_{a_r}_{a_b}_{ts_b}_{ts_r}"
        results.append({
            "key": param_key,
            "ma": best_ma, "mc": best_mc, "exit": ep,
            "metrics": m, "trades": trades,
        })

    results.sort(key=lambda x: x["metrics"]["score"], reverse=True)
    return results[:top_n]


# ===== WALK-FORWARD OPTIMIZATION =====

def walk_forward_optimize(instrument, days, n_folds=4, balance=100, mc_sims=0):
    """
    Anchored walk-forward: train on expanding window, test on next chunk.
    Proves the optimization PROCESS works, not just one lucky split.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"WALK-FORWARD: {instrument} ({n_folds} folds)")
    logger.info(f"{'='*60}")

    data = fetch_oanda_historical(instrument, "H1", days)
    if data.empty or len(data) < 500:
        logger.warning(f"Not enough data for {instrument}")
        return None

    total_bars = len(data)
    segment_size = total_bars // (n_folds + 1)

    if segment_size < 300:
        logger.warning(f"Segments too small ({segment_size} bars). Need more data or fewer folds.")
        return None

    logger.info(f"  Total bars: {total_bars} | Segment: ~{segment_size} bars")

    fold_results = []
    all_oos_trades = []
    param_keys_per_fold = []
    t_start = time.time()

    for fold in range(n_folds):
        train_end = (fold + 1) * segment_size
        test_start = train_end
        test_end = min((fold + 2) * segment_size, total_bars)

        train = data.iloc[:train_end].copy()
        test = data.iloc[test_start:test_end].copy()

        logger.info(f"\n  --- Fold {fold+1}/{n_folds} ---")
        logger.info(f"  Train: {len(train)} bars | Test: {len(test)} bars")

        t0 = time.time()
        train_votes = precompute_votes(train)
        logger.info(f"  Train votes: {len(train_votes)} in {time.time()-t0:.1f}s")

        top_n = grid_search(train_votes, train, instrument, balance, top_n=20)
        best = top_n[0]
        param_keys_per_fold.append([r["key"] for r in top_n])

        t0 = time.time()
        test_votes = precompute_votes(test)
        logger.info(f"  Test votes: {len(test_votes)} in {time.time()-t0:.1f}s")

        # Optimized on test
        opt_sigs = apply_entry_filter(test_votes, best["ma"], best["mc"])
        opt_trades = simulate(test, opt_sigs, instrument, best["exit"])
        opt_m = score_trades(opt_trades, balance)

        # Defaults on test (for comparison)
        def_sigs = apply_entry_filter(test_votes, 5, 3)
        def_exit = {k: CURRENT_DEFAULTS[k] for k in
                    ["tp1_r", "tp1_pct", "adverse_r", "adverse_bars",
                     "time_stop_bars", "time_stop_r"]}
        def_trades = simulate(test, def_sigs, instrument, def_exit)
        def_m = score_trades(def_trades, balance)

        all_oos_trades.extend(opt_trades)

        fold_results.append({
            "fold": fold + 1,
            "params": {"ma": best["ma"], "mc": best["mc"], "exit": best["exit"]},
            "train": best["metrics"],
            "test_opt": opt_m,
            "test_def": def_m,
        })

        logger.info(f"  Defaults:  {def_m['n']}t PF:{def_m['pf']} P&L:${def_m['pnl']:,.2f}")
        logger.info(f"  Optimized: {opt_m['n']}t PF:{opt_m['pf']} P&L:${opt_m['pnl']:,.2f} "
                    f"(MA={best['ma']} MC={best['mc']})")

    elapsed = time.time() - t_start

    # Cross-fold consensus
    all_keys = []
    for keys in param_keys_per_fold:
        all_keys.extend(keys)
    key_counts = Counter(all_keys)

    # Aggregate
    profitable_folds = sum(1 for fr in fold_results if fr["test_opt"]["pnl"] > 0)
    beats_default = sum(1 for fr in fold_results
                       if fr["test_opt"]["pnl"] > fr["test_def"]["pnl"])
    avg_pf = np.mean([fr["test_opt"]["pf"] for fr in fold_results])
    total_oos_pnl = sum(fr["test_opt"]["pnl"] for fr in fold_results)
    total_def_pnl = sum(fr["test_def"]["pnl"] for fr in fold_results)
    robust = profitable_folds >= n_folds * 0.75

    logger.info(f"\n  {'─'*40}")
    logger.info(f"  WALK-FORWARD SUMMARY ({elapsed:.0f}s)")
    logger.info(f"    Profitable folds:     {profitable_folds}/{n_folds}")
    logger.info(f"    Beats default:        {beats_default}/{n_folds}")
    logger.info(f"    Avg OOS PF:           {avg_pf:.2f}")
    logger.info(f"    Total OOS P&L (opt):  ${total_oos_pnl:,.2f}")
    logger.info(f"    Total OOS P&L (def):  ${total_def_pnl:,.2f}")
    logger.info(f"    Optimization value:   ${total_oos_pnl - total_def_pnl:+,.2f}")
    logger.info(f"    ROBUST: {'YES' if robust else 'NO'}")

    if key_counts.most_common(3):
        logger.info(f"    Consistent params across folds:")
        for k, c in key_counts.most_common(3):
            logger.info(f"      {k} — top-20 in {c}/{n_folds} folds")

    # Final optimization on all data for deployment (only if robust)
    deployment = None
    if robust:
        logger.info(f"\n  FINAL OPTIMIZATION (all {total_bars} bars for deployment)")
        t0 = time.time()
        all_votes = precompute_votes(data)
        top_n = grid_search(all_votes, data, instrument, balance, top_n=5)
        best = top_n[0]
        x = best["exit"]
        logger.info(f"    Computed in {time.time()-t0:.1f}s")
        logger.info(f"    MA={best['ma']} MC={best['mc']} | "
                    f"TP1={x['tp1_r']}R @{x['tp1_pct']:.0%} | "
                    f"AE={x['adverse_r']}R/{x['adverse_bars']}b | "
                    f"TS={x['time_stop_bars']}b/{x['time_stop_r']}R")
        logger.info(f"    All-data: {best['metrics']['n']}t PF:{best['metrics']['pf']} "
                    f"P&L:${best['metrics']['pnl']:,.2f}")
        deployment = {"ma": best["ma"], "mc": best["mc"], "exit": best["exit"],
                      "metrics": best["metrics"]}

    # Monte Carlo on combined OOS trades
    mc_result = None
    if mc_sims > 0 and all_oos_trades:
        logger.info(f"\n  MONTE CARLO ({mc_sims} sims on {len(all_oos_trades)} OOS trades)")
        mc_result = monte_carlo_test(all_oos_trades, mc_sims, balance)
        logger.info(f"    Median final:     ${mc_result['median_final']:,.2f}")
        logger.info(f"    5th-95th:         ${mc_result['p5_final']:,.2f} to ${mc_result['p95_final']:,.2f}")
        logger.info(f"    Ruin probability: {mc_result['ruin_pct']:.1f}%")
        logger.info(f"    Profit prob:      {mc_result['profit_probability']:.1f}%")
        logger.info(f"    95th pctl DD:     {mc_result['p95_dd']:.1f}%")
        logger.info(f"    MC Robust:        {'YES' if mc_result['robust'] else 'NO'}")

    return {
        "instrument": instrument,
        "n_folds": n_folds,
        "fold_results": fold_results,
        "profitable_folds": profitable_folds,
        "beats_default": beats_default,
        "avg_oos_pf": round(avg_pf, 2),
        "total_oos_pnl": round(total_oos_pnl, 2),
        "total_def_pnl": round(total_def_pnl, 2),
        "robust": robust,
        "deployment": deployment,
        "monte_carlo": mc_result,
        "elapsed": round(elapsed, 1),
    }


# ===== SINGLE-SPLIT OPTIMIZATION =====

def optimize_pair(instrument, days=365, balance=100, mc_sims=0):
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZING: {instrument}")
    logger.info(f"{'='*60}")

    data = fetch_oanda_historical(instrument, "H1", days)
    if data.empty or len(data) < 500:
        logger.warning(f"Not enough data for {instrument}")
        return None

    split = int(len(data) * 0.75)
    train = data.iloc[:split]
    test = data.iloc[split:]
    logger.info(f"Bars: {len(data)} | Train: {len(train)} | Test: {len(test)}")

    t0 = time.time()
    train_votes = precompute_votes(train)
    logger.info(f"Pre-computed {len(train_votes)} bar votes in {time.time()-t0:.1f}s")

    # Phase 1: Entry
    logger.info(f"\nPhase 1 — Entry thresholds")
    best_entry_score = -99999
    best_ma, best_mc = 5, 3

    for ma, mc in product(ENTRY_GRID["min_agreeing"], ENTRY_GRID["min_categories"]):
        sigs = apply_entry_filter(train_votes, ma, mc)
        default_exit = {k: CURRENT_DEFAULTS[k] for k in
                        ["tp1_r", "tp1_pct", "adverse_r", "adverse_bars",
                         "time_stop_bars", "time_stop_r"]}
        trades = simulate(train, sigs, instrument, default_exit)
        m = score_trades(trades, balance)
        tag = " <-- CURRENT" if ma == 5 and mc == 3 else ""
        logger.info(f"  MA={ma} MC={mc} | {m['n']:3d} trades | WR: {m['wr']:5.1f}% | "
                   f"PF: {m['pf']:.2f} | P&L: ${m['pnl']:>8,.2f} | "
                   f"Score: {m['score']:.4f}{tag}")
        if m["score"] > best_entry_score:
            best_entry_score = m["score"]
            best_ma, best_mc = ma, mc

    logger.info(f"  >>> Best entry: MA={best_ma}, MC={best_mc}")

    # Phase 2: Exit
    logger.info(f"\nPhase 2 — Exit management ({best_ma}/{best_mc} entry)")
    sigs = apply_entry_filter(train_votes, best_ma, best_mc)
    combos = list(product(
        EXIT_GRID["tp1_r"], EXIT_GRID["tp1_pct"],
        EXIT_GRID["adverse_r"], EXIT_GRID["adverse_bars"],
        EXIT_GRID["time_stop_bars"], EXIT_GRID["time_stop_r"],
    ))
    logger.info(f"  Testing {len(combos)} exit combinations...")

    best_exit_score = -99999
    best_exit = {k: CURRENT_DEFAULTS[k] for k in
                 ["tp1_r", "tp1_pct", "adverse_r", "adverse_bars",
                  "time_stop_bars", "time_stop_r"]}
    best_exit_metrics = None
    top_results = []

    for tp1r, tp1p, a_r, a_b, ts_b, ts_r in combos:
        ep = {"tp1_r": tp1r, "tp1_pct": tp1p, "adverse_r": a_r,
              "adverse_bars": a_b, "time_stop_bars": ts_b, "time_stop_r": ts_r}
        trades = simulate(train, sigs, instrument, ep)
        m = score_trades(trades, balance)
        top_results.append((m["score"], m, ep, trades))
        if m["score"] > best_exit_score:
            best_exit_score = m["score"]
            best_exit = ep.copy()
            best_exit_metrics = m

    top_results.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"\n  Top 5 exit configs (training):")
    for rank, (sc, m, ep, _) in enumerate(top_results[:5], 1):
        logger.info(f"  #{rank} TP1={ep['tp1_r']}R @{ep['tp1_pct']:.0%} | "
                   f"AE={ep['adverse_r']}R/{ep['adverse_bars']}b | "
                   f"TS={ep['time_stop_bars']}b/{ep['time_stop_r']}R | "
                   f"{m['n']}t WR:{m['wr']}% PF:{m['pf']} "
                   f"DD:{m['max_dd_pct']:.1f}% Sharpe:{m['sharpe']:.2f} "
                   f"P&L:${m['pnl']:,.2f}")

    # Phase 3: OOS validation
    logger.info(f"\nPhase 3 — Out-of-sample validation")
    test_votes = precompute_votes(test)

    def_sigs = apply_entry_filter(test_votes, 5, 3)
    def_exit = {k: CURRENT_DEFAULTS[k] for k in
                ["tp1_r", "tp1_pct", "adverse_r", "adverse_bars",
                 "time_stop_bars", "time_stop_r"]}
    def_trades = simulate(test, def_sigs, instrument, def_exit)
    def_m = score_trades(def_trades, balance)

    opt_sigs = apply_entry_filter(test_votes, best_ma, best_mc)
    opt_trades = simulate(test, opt_sigs, instrument, best_exit)
    opt_m = score_trades(opt_trades, balance)

    logger.info(f"  CURRENT  : {def_m['n']}t | WR:{def_m['wr']}% | PF:{def_m['pf']} | "
               f"DD:{def_m['max_dd_pct']:.1f}% | Sharpe:{def_m['sharpe']:.2f} | "
               f"P&L:${def_m['pnl']:,.2f}")
    logger.info(f"  OPTIMIZED: {opt_m['n']}t | WR:{opt_m['wr']}% | PF:{opt_m['pf']} | "
               f"DD:{opt_m['max_dd_pct']:.1f}% | Sharpe:{opt_m['sharpe']:.2f} | "
               f"P&L:${opt_m['pnl']:,.2f}")

    diff = opt_m["pnl"] - def_m["pnl"]
    logger.info(f"  IMPROVEMENT: ${diff:+,.2f}")

    # Monte Carlo
    mc_result = None
    if mc_sims > 0 and opt_trades:
        logger.info(f"\n  Monte Carlo ({mc_sims} sims on OOS trades):")
        mc_result = monte_carlo_test(opt_trades, mc_sims, balance)
        logger.info(f"    Median final:     ${mc_result['median_final']:,.2f}")
        logger.info(f"    5th-95th:         ${mc_result['p5_final']:,.2f} to ${mc_result['p95_final']:,.2f}")
        logger.info(f"    Ruin probability: {mc_result['ruin_pct']:.1f}%")
        logger.info(f"    Profit prob:      {mc_result['profit_probability']:.1f}%")
        logger.info(f"    95th pctl DD:     {mc_result['p95_dd']:.1f}%")
        logger.info(f"    MC Robust:        {'YES' if mc_result['robust'] else 'NO'}")

    return {
        "instrument": instrument,
        "best_ma": best_ma, "best_mc": best_mc,
        "best_exit": best_exit,
        "train": best_exit_metrics,
        "test_default": def_m,
        "test_optimized": opt_m,
        "improvement": diff,
        "monte_carlo": mc_result,
    }


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser(description="Strategy parameter optimizer v2")
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--balance", type=float, default=100)
    parser.add_argument("--walk-forward", action="store_true",
                       help="Walk-forward validation (prevents overfitting)")
    parser.add_argument("--wf-folds", type=int, default=4,
                       help="Walk-forward fold count (default 4)")
    parser.add_argument("--monte-carlo", type=int, default=0,
                       help="Monte Carlo sims (0=disabled, try 1000)")
    parser.add_argument("--report", action="store_true",
                       help="Save JSON report to data/optimize_report.json")
    args = parser.parse_args()

    pairs = [args.instrument] if args.instrument else WATCHLIST

    exit_count = 1
    for v in EXIT_GRID.values():
        exit_count *= len(v)
    entry_count = len(ENTRY_GRID["min_agreeing"]) * len(ENTRY_GRID["min_categories"])

    logger.info("=" * 60)
    logger.info("STRATEGY PARAMETER OPTIMIZER v2")
    logger.info(f"  Pairs:       {', '.join(pairs)}")
    logger.info(f"  Days:        {args.days} | Balance: ${args.balance:,.2f}")
    mode_str = f"walk-forward ({args.wf_folds} folds)" if args.walk_forward else "single split (75/25)"
    logger.info(f"  Mode:        {mode_str}")
    if args.monte_carlo:
        logger.info(f"  Monte Carlo: {args.monte_carlo} sims")
    logger.info(f"  Entry grid:  {entry_count} combos")
    logger.info(f"  Exit grid:   {exit_count} combos")
    logger.info(f"  Scoring:     PF(40%) + DD(25%) + Sharpe(15%) + WR(10%) + Exp(10%)")
    logger.info("=" * 60)

    results = {}
    t_total = time.time()

    for pair in pairs:
        if args.walk_forward:
            r = walk_forward_optimize(pair, args.days, args.wf_folds, args.balance,
                                      args.monte_carlo)
        else:
            r = optimize_pair(pair, args.days, args.balance, args.monte_carlo)
        if r:
            results[pair] = r

    elapsed_total = time.time() - t_total

    # ===== FINAL SUMMARY =====
    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL RESULTS ({elapsed_total:.0f}s total)")
    logger.info("=" * 60)

    if args.walk_forward:
        robust_pairs = []
        fragile_pairs = []

        for inst, r in results.items():
            tag = "ROBUST" if r["robust"] else "FRAGILE"
            if r["robust"]:
                robust_pairs.append(inst)
            else:
                fragile_pairs.append(inst)

            logger.info(f"\n  {inst} [{tag}]:")
            logger.info(f"    Profitable folds: {r['profitable_folds']}/{r['n_folds']}")
            logger.info(f"    Beats default:    {r['beats_default']}/{r['n_folds']}")
            logger.info(f"    Avg OOS PF:       {r['avg_oos_pf']:.2f}")
            logger.info(f"    OOS P&L (opt):    ${r['total_oos_pnl']:,.2f}")
            logger.info(f"    OOS P&L (def):    ${r['total_def_pnl']:,.2f}")
            logger.info(f"    Opt value:        ${r['total_oos_pnl'] - r['total_def_pnl']:+,.2f}")

            for fr in r["fold_results"]:
                tm = fr["test_opt"]
                dm = fr["test_def"]
                better = "+" if tm["pnl"] > dm["pnl"] else "-"
                logger.info(f"      F{fr['fold']}: {tm['n']}t PF:{tm['pf']} "
                           f"P&L:${tm['pnl']:>8,.2f} vs def ${dm['pnl']:>8,.2f} [{better}]")

            if r.get("monte_carlo"):
                mc = r["monte_carlo"]
                logger.info(f"    MC: Robust={'YES' if mc['robust'] else 'NO'} | "
                           f"Ruin:{mc['ruin_pct']:.1f}% | Profit:{mc['profit_probability']:.1f}%")

            if r.get("deployment"):
                d = r["deployment"]
                x = d["exit"]
                logger.info(f"    DEPLOY: MA={d['ma']} MC={d['mc']} | "
                           f"TP1={x['tp1_r']}R @{x['tp1_pct']:.0%} | "
                           f"AE={x['adverse_r']}R/{x['adverse_bars']}b | "
                           f"TS={x['time_stop_bars']}b/{x['time_stop_r']}R")

        logger.info(f"\n  {'─'*40}")
        logger.info(f"  VERDICT:")
        logger.info(f"    Robust pairs:  {', '.join(robust_pairs) if robust_pairs else 'NONE'}")
        logger.info(f"    Fragile pairs: {', '.join(fragile_pairs) if fragile_pairs else 'NONE'}")

        if robust_pairs:
            logger.info(f"\n  DEPLOYMENT CONFIG (robust pairs only):")
            for inst in robust_pairs:
                d = results[inst].get("deployment")
                if d:
                    x = d["exit"]
                    logger.info(f"    \"{inst}\": {{"
                               f"\"min_agreeing\": {d['ma']}, \"min_categories\": {d['mc']}, "
                               f"\"tp1_r\": {x['tp1_r']}, \"tp1_pct\": {x['tp1_pct']}, "
                               f"\"adverse_r\": {x['adverse_r']}, \"adverse_bars\": {x['adverse_bars']}, "
                               f"\"time_stop_bars\": {x['time_stop_bars']}, \"time_stop_r\": {x['time_stop_r']}}},")

    else:
        total_def = 0
        total_opt = 0

        for inst, r in results.items():
            x = r["best_exit"]
            tm = r["test_optimized"]
            dm = r["test_default"]
            total_def += dm["pnl"]
            total_opt += tm["pnl"]

            logger.info(f"\n  {inst}:")
            logger.info(f"    Entry: MA={r['best_ma']}, MC={r['best_mc']}")
            logger.info(f"    Exit:  TP1={x['tp1_r']}R @{x['tp1_pct']:.0%} | "
                       f"AE={x['adverse_r']}R/{x['adverse_bars']}b | "
                       f"TS={x['time_stop_bars']}b/{x['time_stop_r']}R")
            logger.info(f"    Test:  {tm['n']}t | WR:{tm['wr']}% | PF:{tm['pf']} | "
                       f"DD:{tm['max_dd_pct']:.1f}% | Sharpe:{tm['sharpe']:.2f} | "
                       f"P&L:${tm['pnl']:,.2f} (was ${dm['pnl']:,.2f})")
            if r.get("monte_carlo"):
                mc = r["monte_carlo"]
                logger.info(f"    MC: Robust={'YES' if mc['robust'] else 'NO'} | "
                           f"Ruin:{mc['ruin_pct']:.1f}% | Profit:{mc['profit_probability']:.1f}%")

        logger.info(f"\n  {'─'*40}")
        logger.info(f"  PORTFOLIO (test period):")
        logger.info(f"    Default total:   ${total_def:>10,.2f}")
        logger.info(f"    Optimized total: ${total_opt:>10,.2f}")
        logger.info(f"    Improvement:     ${total_opt - total_def:>+10,.2f}")

        logger.info(f"\n  {'─'*40}")
        logger.info(f"  OPTIMAL CONFIG (copy into settings.py):")
        for inst, r in results.items():
            x = r["best_exit"]
            logger.info(f"    \"{inst}\": {{"
                       f"\"min_agreeing\": {r['best_ma']}, \"min_categories\": {r['best_mc']}, "
                       f"\"tp1_r\": {x['tp1_r']}, \"tp1_pct\": {x['tp1_pct']}, "
                       f"\"adverse_r\": {x['adverse_r']}, \"adverse_bars\": {x['adverse_bars']}, "
                       f"\"time_stop_bars\": {x['time_stop_bars']}, \"time_stop_r\": {x['time_stop_r']}}},")

    # Save report
    if args.report:
        report_path = "data/optimize_report.json"
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {"days": args.days, "balance": args.balance,
                       "walk_forward": args.walk_forward, "wf_folds": args.wf_folds,
                       "monte_carlo": args.monte_carlo},
            "scoring": "PF(40%) + MaxDD(25%) + Sharpe(15%) + WR(10%) + Expectancy(10%)",
            "elapsed_seconds": round(elapsed_total, 1),
            "results": {},
        }
        for inst, r in results.items():
            clean = {k: v for k, v in r.items()
                     if k not in ("opt_trades", "all_test_trades", "all_oos_trades")}
            report["results"][inst] = clean
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\n  Report saved: {report_path}")

    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
