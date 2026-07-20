"""
Backtest Runner — test the forex bot on historical data.

Usage:
    python backtest.py                          # All pairs, 180 days, fast mode
    python backtest.py --instrument EUR_USD     # Single pair only
    python backtest.py --days 365               # 1 year of data
    python backtest.py --mode full              # Full AI pipeline
    python backtest.py --monte-carlo            # Add Monte Carlo analysis
"""
import sys
import os
import logging
import argparse

os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/backtest.log"),
    ],
)
logger = logging.getLogger("Backtest")


def build_pipeline(args):
    from brain.providers import AIProvider
    from brain.router import TaskRouter
    from agents.technical import TechnicalAgent
    from agents.debate import DebateAgent
    from agents.risk_manager import RiskManager
    from agents.market_structure import MarketStructureAgent
    from agents.session_filter import SessionFilter
    from agents.spread_filter import SpreadFilter
    from agents.news_agent import NewsAgent
    from agents.pipeline import TradePipeline
    from strategy.strategy_selector import StrategySelector
    from data.trade_journal import TradeJournal

    technical = TechnicalAgent()
    strategy_selector = StrategySelector()
    swing_selector = None
    if args.style == "swing":
        from strategy.swing_selector import SwingSelector
        swing_selector = SwingSelector()
        logger.info("Swing engine loaded — gate+entry architecture")
    risk = RiskManager(account_balance=args.balance)
    structure = MarketStructureAgent()
    journal = TradeJournal(db_path="data/backtest_trades.db")
    session_filter = SessionFilter()
    spread_filter = SpreadFilter()
    news_agent = NewsAgent()

    debate = None
    router = None
    provider = AIProvider()
    if provider.clients:
        router = TaskRouter(provider)
        debate = DebateAgent(router=router, provider=provider)
        logger.info(f"AI brain active — providers: {list(provider.clients.keys())}")
    else:
        logger.info("AI brain disabled — no API keys configured")

    pipeline = TradePipeline(
        technical=technical,
        strategy_selector=strategy_selector,
        debate=debate,
        risk=risk,
        session_filter=session_filter,
        spread_filter=spread_filter,
        news_agent=news_agent,
        market_structure=structure,
        multi_tf=None,
        journal=journal,
        router=router,
        swing_selector=swing_selector,
    )

    return pipeline, risk


def run_single(args, instrument, style_override=None):
    from backtesting.data_loader import fetch_oanda_historical
    from backtesting.engine import BacktestEngine

    style = style_override or args.style
    timeframe = "H4" if style == "swing" else args.timeframe
    data = fetch_oanda_historical(instrument, timeframe, args.days)
    if data.empty:
        logger.warning(f"No data for {instrument} — skipping")
        return None

    daily_data = None
    if style == "swing":
        daily_data = fetch_oanda_historical(instrument, "D", args.days + 250)
        if not daily_data.empty:
            logger.info(f"Daily data loaded: {len(daily_data)} bars for multi-TF alignment")

    # Override style for pipeline building
    orig_style = args.style
    args.style = style
    pipeline, risk = build_pipeline(args)
    args.style = orig_style

    engine = BacktestEngine(pipeline, risk, instrument, args.mode, timeframe,
                            style=style, daily_data=daily_data)
    results = engine.run(data)
    return results


def main():
    from config.settings import WATCHLIST, SWING_WATCHLIST

    parser = argparse.ArgumentParser(description="Backtest the forex bot")
    parser.add_argument("--instrument", default=None,
                       help="Single pair (e.g. EUR_USD). Omit to test ALL pairs.")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--timeframe", default="H1", help="Candle timeframe")
    parser.add_argument("--mode", default="fast", choices=["fast", "full"],
                       help="fast=technical only, full=with AI")
    parser.add_argument("--style", default="scalp", choices=["scalp", "swing", "combined"],
                       help="scalp=H1 quick, swing=H4 big, combined=both")
    parser.add_argument("--balance", type=float, default=100, help="Starting balance")
    parser.add_argument("--monte-carlo", action="store_true",
                       help="Run Monte Carlo risk analysis")
    parser.add_argument("--mc-sims", type=int, default=10000,
                       help="Monte Carlo simulations")
    args = parser.parse_args()

    if args.style == "combined":
        scalp_instruments = [args.instrument] if args.instrument else WATCHLIST
        swing_instruments = [args.instrument] if args.instrument else SWING_WATCHLIST
        all_instruments_display = list(set(scalp_instruments + swing_instruments))
    elif args.instrument:
        scalp_instruments = [args.instrument]
        swing_instruments = [args.instrument]
        all_instruments_display = [args.instrument]
    elif args.style == "swing":
        scalp_instruments = []
        swing_instruments = SWING_WATCHLIST
        all_instruments_display = SWING_WATCHLIST
    else:
        scalp_instruments = WATCHLIST
        swing_instruments = []
        all_instruments_display = WATCHLIST

    logger.info("=" * 60)
    logger.info("FOREX BACKTEST")
    if args.style == "combined":
        logger.info(f"  Scalp pairs: {len(scalp_instruments)} | Swing pairs: {len(swing_instruments)}")
    else:
        logger.info(f"  Pairs: {len(all_instruments_display)} | Days: {args.days}")
    logger.info(f"  Mode: {args.mode} | Style: {args.style} | Balance: ${args.balance:,.2f}")
    logger.info(f"  Days: {args.days} | Instruments: {', '.join(all_instruments_display)}")
    logger.info("=" * 60)

    all_results = {}
    all_trades = []
    total_pnl = 0

    # SCALP runs
    if args.style in ("scalp", "combined"):
        for instrument in scalp_instruments:
            logger.info(f"\n{'─' * 40}")
            logger.info(f"SCALP: {instrument}")
            logger.info(f"{'─' * 40}")

            results = run_single(args, instrument, style_override="scalp")
            if results is None:
                continue

            key = f"{instrument}_scalp"
            all_results[key] = results
            total_pnl += results.get("net_pnl", 0)
            if results.get("trades"):
                all_trades.extend(results["trades"])

            trades = results["total_trades"]
            if trades > 0:
                logger.info(
                    f"  {instrument} (scalp): {trades} trades | "
                    f"Win: {results.get('win_rate', 0):.1f}% | "
                    f"P&L: ${results.get('net_pnl', 0):,.2f} | "
                    f"PF: {results.get('profit_factor', 0):.2f} | "
                    f"Sharpe: {results.get('sharpe_ratio', 0):.2f}"
                )
            else:
                logger.info(f"  {instrument} (scalp): 0 trades")

    # SWING runs
    if args.style in ("swing", "combined"):
        for instrument in swing_instruments:
            logger.info(f"\n{'─' * 40}")
            logger.info(f"SWING: {instrument}")
            logger.info(f"{'─' * 40}")

            results = run_single(args, instrument, style_override="swing")
            if results is None:
                continue

            key = f"{instrument}_swing" if args.style == "combined" else instrument
            all_results[key] = results
            total_pnl += results.get("net_pnl", 0)
            if results.get("trades"):
                all_trades.extend(results["trades"])

            trades = results["total_trades"]
            if trades > 0:
                logger.info(
                    f"  {instrument} (swing): {trades} trades | "
                    f"Win: {results.get('win_rate', 0):.1f}% | "
                    f"P&L: ${results.get('net_pnl', 0):,.2f} | "
                    f"PF: {results.get('profit_factor', 0):.2f} | "
                    f"Sharpe: {results.get('sharpe_ratio', 0):.2f}"
                )
            else:
                logger.info(f"  {instrument} (swing): 0 trades")

    # COMBINED RESULTS
    logger.info("\n" + "=" * 60)
    logger.info("COMBINED RESULTS — ALL PAIRS")
    logger.info("=" * 60)

    total_trades = len(all_trades)
    if total_trades > 0:
        import numpy as np
        wins = [t for t in all_trades if t["pnl"] > 0]
        losses = [t for t in all_trades if t["pnl"] <= 0]
        win_pnls = [t["pnl"] for t in wins]
        loss_pnls = [t["pnl"] for t in losses]

        win_rate = len(wins) / total_trades * 100
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        pf = abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else 0
        avg_pips = np.mean([t["pnl_pips"] for t in all_trades])

        logger.info(f"  Total trades: {total_trades}")
        logger.info(f"  Win rate: {win_rate:.1f}%")
        logger.info(f"  Net P&L: ${total_pnl:,.2f}")
        logger.info(f"  Profit factor: {pf:.2f}")
        logger.info(f"  Avg win: ${avg_win:,.2f}")
        logger.info(f"  Avg loss: ${avg_loss:,.2f}")
        logger.info(f"  Avg pips: {avg_pips:.1f}")

        # Per-pair breakdown
        logger.info(f"\n{'─' * 40}")
        logger.info("PER-PAIR BREAKDOWN")
        logger.info(f"{'─' * 40}")

        sorted_pairs = sorted(all_results.items(),
                             key=lambda x: x[1].get("net_pnl", 0), reverse=True)
        for inst, r in sorted_pairs:
            if r["total_trades"] > 0:
                logger.info(
                    f"  {inst:8s} | {r['total_trades']:3d} trades | "
                    f"Win: {r.get('win_rate', 0):5.1f}% | "
                    f"P&L: ${r.get('net_pnl', 0):>8,.2f} | "
                    f"PF: {r.get('profit_factor', 0):.2f}"
                )
    else:
        logger.info("  No trades across any pair")

    # PROP FIRM STATUS
    from config.settings import PROP_FIRM
    if PROP_FIRM.get("enabled") and total_trades > 0:
        logger.info("\n" + "=" * 60)
        logger.info(f"PROP FIRM STATUS — {PROP_FIRM['name']}")
        logger.info("=" * 60)

        final_bal = args.balance + total_pnl
        return_pct = total_pnl / args.balance * 100
        target = PROP_FIRM["profit_target_pct"]
        split = PROP_FIRM["profit_split"]

        eq_vals = []
        for inst, r in all_results.items():
            eq_vals.extend([e["equity"] for e in r.get("equity_curve", [])])
        peak = args.balance
        max_dd_pct = 0
        running = args.balance
        for t in all_trades:
            running += t["pnl"]
            if running > peak:
                peak = running
            dd = (peak - running) / peak * 100
            if dd > max_dd_pct:
                max_dd_pct = dd

        from collections import defaultdict
        daily_pnls = defaultdict(float)
        for t in all_trades:
            day_key = t.get("date", "unknown")
            daily_pnls[day_key] += t["pnl"]
        trading_days = len([d for d in daily_pnls if d != "unknown"])

        worst_daily = 0
        worst_daily_pct = 0
        worst_daily_date = ""
        best_daily = 0
        best_daily_date = ""
        for day, pnl in daily_pnls.items():
            if pnl < worst_daily:
                worst_daily = pnl
                worst_daily_date = day
                worst_daily_pct = abs(pnl) / args.balance * 100
            if pnl > best_daily:
                best_daily = pnl
                best_daily_date = day

        dd_ok = "PASS" if max_dd_pct < PROP_FIRM["max_total_loss_pct"] else "FAIL"
        daily_ok = "PASS" if worst_daily_pct < PROP_FIRM["max_daily_loss_pct"] else "FAIL"

        logger.info(f"  Account: ${args.balance:,.2f} → ${final_bal:,.2f}")
        logger.info(f"  Total P&L: ${total_pnl:,.2f} ({return_pct:+.2f}%)")
        logger.info(f"  Max drawdown: {max_dd_pct:.2f}% / {PROP_FIRM['max_total_loss_pct']}% limit [{dd_ok}]")
        logger.info(f"  Worst daily loss: ${worst_daily:,.2f} ({worst_daily_pct:.2f}%) on {worst_daily_date} [{daily_ok}]")
        logger.info(f"  Best daily gain: ${best_daily:,.2f} on {best_daily_date}")
        logger.info(f"  Trading days: {trading_days} (min {PROP_FIRM['min_trading_days']})")

        target_hit = return_pct >= target
        target_status = "HIT" if target_hit else f"{return_pct:.1f}% / {target}%"
        logger.info(f"  Profit target: {target_status}")
        fee = PROP_FIRM.get("accounts", {}).get(int(args.balance), 0)
        if fee:
            logger.info(f"  Challenge fee: ${fee:,} (refunded on first payout)")

        logger.info(f"  Profit split: {split:.0%}")
        if total_pnl > 0:
            payout = total_pnl * split
            logger.info(f"  YOUR PAYOUT: ${payout:,.2f}")
            if fee:
                logger.info(f"  NET AFTER FEE REFUND: ${payout + fee:,.2f}")
        else:
            logger.info(f"  Status: IN DRAWDOWN — no payout yet")

    if args.monte_carlo and all_trades:
        logger.info("\n" + "=" * 60)
        logger.info("MONTE CARLO RISK ANALYSIS (all pairs combined)")
        logger.info("=" * 60)

        from backtesting.monte_carlo import MonteCarloAnalyzer
        mc = MonteCarloAnalyzer(simulations=args.mc_sims)
        mc_results = mc.analyze(all_trades, args.balance)

        logger.info(f"  Simulations: {mc_results['simulations']}")
        logger.info(f"  Median outcome: ${mc_results['median_final_balance']:,.2f}")
        logger.info(f"  95% range: ${mc_results['confidence_intervals']['95%']['lower']:,.2f} "
                   f"to ${mc_results['confidence_intervals']['95%']['upper']:,.2f}")
        logger.info(f"  Ruin probability: {mc_results['probability_of_ruin']:.2%}")
        logger.info(f"  Median max drawdown: {mc_results['median_max_drawdown_pct']:.2f}%")

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
