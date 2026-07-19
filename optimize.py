#!/usr/bin/env python3
"""
Strategy Parameter Optimizer — finds best settings per pair using historical data.
Pre-computes strategy votes once, then rapidly tests entry + exit parameter grids.
Train/test split prevents overfitting.

Usage:
    python optimize.py                    # All pairs, 365 days
    python optimize.py --instrument EUR_USD --days 365
"""
import sys, os, logging, time, argparse
import numpy as np
import pandas as pd
import ta
from itertools import product

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

TRAIN_PCT = 0.75

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
        bar = data.iloc[i]

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
            "close": float(bar["close"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "open_": float(bar["open"]),
            "atr": float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0,
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

            # Stop loss
            if (pos["d"] == "BUY" and l <= pos["sl"]) or \
               (pos["d"] == "SELL" and h >= pos["sl"]):
                p = (pos["sl"] - pos["ep"]) * pos["u"] if pos["d"] == "BUY" \
                    else (pos["ep"] - pos["sl"]) * pos["u"]
                trades.append(p)
                pos = None
                continue

            # Adverse excursion
            if held <= ab and not pos.get("be"):
                adv_p = (pos["ep"] - l) / pv if pos["d"] == "BUY" else (h - pos["ep"]) / pv
                if adv_p >= pos["rp"] * ar:
                    p = (c - pos["ep"]) * pos["u"] if pos["d"] == "BUY" \
                        else (pos["ep"] - c) * pos["u"]
                    trades.append(p)
                    pos = None
                    continue

            # Time stop
            if held >= tsb and not pos.get("t1"):
                pp = (c - pos["ep"]) / pv if pos["d"] == "BUY" else (pos["ep"] - c) / pv
                if pp < pos["rp"] * tsr:
                    trades.append(pp * pv * pos["u"])
                    pos = None
                    continue

            # TP1
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

            # TP2 at 2.5R
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

            # ATR trailing
            if pos and pos.get("be") and i < len(atr_s) and not pd.isna(atr_s.iloc[i]):
                a = float(atr_s.iloc[i])
                m = 1.5 if pos.get("t2") else 2.0
                if pos["d"] == "BUY":
                    nt = pos["bh"] - a * m
                    if nt > pos["sl"]: pos["sl"] = nt
                else:
                    nt = pos["bl"] + a * m
                    if nt < pos["sl"]: pos["sl"] = nt

        # Entry
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


def score_trades(trades):
    if len(trades) < 5:
        return {"pnl": sum(trades) if trades else 0, "n": len(trades),
                "wr": 0, "pf": 0, "aw": 0, "al": 0, "score": -9999}
    pnl = sum(trades)
    w = [t for t in trades if t > 0]
    lo = [t for t in trades if t <= 0]
    wr = len(w) / len(trades) * 100
    pf = abs(sum(w) / sum(lo)) if lo and sum(lo) != 0 else 0
    aw = np.mean(w) if w else 0
    al = np.mean(lo) if lo else 0
    consistency = min(len(trades) / 30, 1.0)
    score = pnl * consistency * (1 + max(0, pf - 1))
    return {"pnl": round(pnl, 2), "n": len(trades), "wr": round(wr, 1),
            "pf": round(pf, 2), "aw": round(aw, 2), "al": round(al, 2),
            "score": round(score, 2)}


def optimize_pair(instrument, days=365):
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZING: {instrument}")
    logger.info(f"{'='*60}")

    data = fetch_oanda_historical(instrument, "H1", days)
    if data.empty or len(data) < 500:
        logger.warning(f"Not enough data for {instrument}")
        return None

    split = int(len(data) * TRAIN_PCT)
    train = data.iloc[:split]
    test = data.iloc[split:]
    logger.info(f"Bars: {len(data)} | Train: {len(train)} | Test: {len(test)}")

    t0 = time.time()
    train_votes = precompute_votes(train)
    logger.info(f"Pre-computed {len(train_votes)} bar votes in {time.time()-t0:.1f}s")

    # Phase 1: Entry params
    logger.info(f"\nPhase 1 — Entry thresholds (default exits)")
    best_entry_score = -99999
    best_ma, best_mc = 5, 3

    for ma, mc in product(ENTRY_GRID["min_agreeing"], ENTRY_GRID["min_categories"]):
        sigs = apply_entry_filter(train_votes, ma, mc)
        default_exit = {k: CURRENT_DEFAULTS[k] for k in
                        ["tp1_r", "tp1_pct", "adverse_r", "adverse_bars",
                         "time_stop_bars", "time_stop_r"]}
        trades = simulate(train, sigs, instrument, default_exit)
        m = score_trades(trades)
        tag = " <-- CURRENT" if ma == 5 and mc == 3 else ""
        logger.info(f"  MA={ma} MC={mc} | {m['n']:3d} trades | WR: {m['wr']:5.1f}% | "
                   f"PF: {m['pf']:.2f} | P&L: ${m['pnl']:>8,.2f}{tag}")
        if m["score"] > best_entry_score:
            best_entry_score = m["score"]
            best_ma, best_mc = ma, mc

    logger.info(f"  >>> Best entry: MA={best_ma}, MC={best_mc}")

    # Phase 2: Exit params with best entry
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
    top_5 = []

    for tp1r, tp1p, a_r, a_b, ts_b, ts_r in combos:
        ep = {"tp1_r": tp1r, "tp1_pct": tp1p, "adverse_r": a_r,
              "adverse_bars": a_b, "time_stop_bars": ts_b, "time_stop_r": ts_r}
        trades = simulate(train, sigs, instrument, ep)
        m = score_trades(trades)
        top_5.append((m["score"], m, ep))
        if m["score"] > best_exit_score:
            best_exit_score = m["score"]
            best_exit = ep.copy()
            best_exit_metrics = m

    top_5.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"\n  Top 5 exit configs (training):")
    for rank, (sc, m, ep) in enumerate(top_5[:5], 1):
        logger.info(f"  #{rank} TP1={ep['tp1_r']}R @{ep['tp1_pct']:.0%} | "
                   f"AE={ep['adverse_r']}R/{ep['adverse_bars']}b | "
                   f"TS={ep['time_stop_bars']}b/{ep['time_stop_r']}R | "
                   f"{m['n']}t WR:{m['wr']}% PF:{m['pf']} P&L:${m['pnl']:,.2f}")

    # Phase 3: Validate on test data
    logger.info(f"\nPhase 3 — Out-of-sample validation")
    test_votes = precompute_votes(test)

    # Default params on test
    def_sigs = apply_entry_filter(test_votes, 5, 3)
    def_exit = {k: CURRENT_DEFAULTS[k] for k in
                ["tp1_r", "tp1_pct", "adverse_r", "adverse_bars",
                 "time_stop_bars", "time_stop_r"]}
    def_trades = simulate(test, def_sigs, instrument, def_exit)
    def_m = score_trades(def_trades)

    # Optimized params on test
    opt_sigs = apply_entry_filter(test_votes, best_ma, best_mc)
    opt_trades = simulate(test, opt_sigs, instrument, best_exit)
    opt_m = score_trades(opt_trades)

    logger.info(f"  CURRENT  (MA=5,MC=3 + default exits): "
               f"{def_m['n']}t | WR:{def_m['wr']}% | PF:{def_m['pf']} | "
               f"AvgW:${def_m['aw']:.2f} | AvgL:${def_m['al']:.2f} | P&L:${def_m['pnl']:,.2f}")
    logger.info(f"  OPTIMIZED (MA={best_ma},MC={best_mc} + tuned exits): "
               f"{opt_m['n']}t | WR:{opt_m['wr']}% | PF:{opt_m['pf']} | "
               f"AvgW:${opt_m['aw']:.2f} | AvgL:${opt_m['al']:.2f} | P&L:${opt_m['pnl']:,.2f}")

    diff = opt_m["pnl"] - def_m["pnl"]
    logger.info(f"  IMPROVEMENT: ${diff:+,.2f}")

    return {
        "instrument": instrument,
        "best_ma": best_ma, "best_mc": best_mc,
        "best_exit": best_exit,
        "train": best_exit_metrics,
        "test_default": def_m,
        "test_optimized": opt_m,
        "improvement": diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    pairs = [args.instrument] if args.instrument else WATCHLIST

    exit_count = 1
    for v in EXIT_GRID.values():
        exit_count *= len(v)

    logger.info("=" * 60)
    logger.info("STRATEGY PARAMETER OPTIMIZER")
    logger.info(f"  Pairs: {', '.join(pairs)}")
    logger.info(f"  Days: {args.days} | Train: {TRAIN_PCT:.0%} | Test: {1-TRAIN_PCT:.0%}")
    logger.info(f"  Entry grid: {len(ENTRY_GRID['min_agreeing'])*len(ENTRY_GRID['min_categories'])} combos")
    logger.info(f"  Exit grid: {exit_count} combos")
    logger.info("=" * 60)

    results = {}
    for pair in pairs:
        r = optimize_pair(pair, args.days)
        if r:
            results[pair] = r

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)

    total_def = 0
    total_opt = 0

    for inst, r in results.items():
        x = r["best_exit"]
        tm = r["test_optimized"]
        dm = r["test_default"]
        total_def += dm["pnl"]
        total_opt += tm["pnl"]

        logger.info(f"\n  {inst}:")
        logger.info(f"    Entry: MIN_AGREEING={r['best_ma']}, MIN_CATEGORIES={r['best_mc']}")
        logger.info(f"    Exit:  TP1={x['tp1_r']}R @{x['tp1_pct']:.0%} | "
                   f"AE={x['adverse_r']}R/{x['adverse_bars']}b | "
                   f"TS={x['time_stop_bars']}b/{x['time_stop_r']}R")
        logger.info(f"    Test:  {tm['n']}t | WR:{tm['wr']}% | PF:{tm['pf']} | "
                   f"P&L:${tm['pnl']:,.2f} (was ${dm['pnl']:,.2f})")

    logger.info(f"\n  {'─'*40}")
    logger.info(f"  PORTFOLIO (test period):")
    logger.info(f"    Default total:   ${total_def:>10,.2f}")
    logger.info(f"    Optimized total: ${total_opt:>10,.2f}")
    logger.info(f"    Improvement:     ${total_opt - total_def:>+10,.2f}")

    # Per-pair optimal config for copy-paste into settings
    logger.info(f"\n  {'─'*40}")
    logger.info(f"  OPTIMAL CONFIG (copy into settings.py):")
    for inst, r in results.items():
        x = r["best_exit"]
        logger.info(f"    \"{inst}\": {{"
                   f"\"ma\": {r['best_ma']}, \"mc\": {r['best_mc']}, "
                   f"\"tp1_r\": {x['tp1_r']}, \"tp1_pct\": {x['tp1_pct']}, "
                   f"\"ae_r\": {x['adverse_r']}, \"ae_bars\": {x['adverse_bars']}, "
                   f"\"ts_bars\": {x['time_stop_bars']}, \"ts_r\": {x['time_stop_r']}}},")


if __name__ == "__main__":
    main()
