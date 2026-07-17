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
    from agents.session_filter import SessionFilter
    from agents.spread_filter import SpreadFilter
    from agents.news_agent import NewsAgent
    from agents.market_structure import MarketStructureAgent
    from agents.pipeline import TradePipeline
    from strategy.strategy_selector import StrategySelector
    from data.trade_journal import TradeJournal

    technical = TechnicalAgent()
    strategy_selector = StrategySelector()
    risk = RiskManager(account_balance=args.balance)
    session_filter = SessionFilter()
    spread_filter = SpreadFilter()
    news_agent = NewsAgent()
    structure = MarketStructureAgent()
    journal = TradeJournal(db_path="data/backtest_trades.db")

    debate = None
    router = None
    if args.mode == "full":
        provider = AIProvider()
        router = TaskRouter(provider)
        debate = DebateAgent(router)

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
    )

    return pipeline, risk


def run_single(args, instrument):
    from backtesting.data_loader import fetch_oanda_historical
    from backtesting.engine import BacktestEngine

    data = fetch_oanda_historical(instrument, args.timeframe, args.days)
    if data.empty:
        logger.warning(f"No data for {instrument} — skipping")
        return None

    pipeline, risk = build_pipeline(args)
    engine = BacktestEngine(pipeline, risk, instrument, args.mode, args.timeframe)
    results = engine.run(data)
    return results


def main():
    from config.settings import WATCHLIST

    parser = argparse.ArgumentParser(description="Backtest the forex bot")
    parser.add_argument("--instrument", default=None,
                       help="Single pair (e.g. EUR_USD). Omit to test ALL pairs.")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--timeframe", default="H1", help="Candle timeframe")
    parser.add_argument("--mode", default="fast", choices=["fast", "full"],
                       help="fast=technical only, full=with AI")
    parser.add_argument("--balance", type=float, default=10000, help="Starting balance")
    parser.add_argument("--monte-carlo", action="store_true",
                       help="Run Monte Carlo risk analysis")
    parser.add_argument("--mc-sims", type=int, default=10000,
                       help="Monte Carlo simulations")
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else WATCHLIST

    logger.info("=" * 60)
    logger.info("FOREX BACKTEST")
    logger.info(f"  Pairs: {len(instruments)} | Days: {args.days}")
    logger.info(f"  Mode: {args.mode} | Balance: ${args.balance:,.2f}")
    logger.info(f"  Instruments: {', '.join(instruments)}")
    logger.info("=" * 60)

    all_results = {}
    all_trades = []
    total_pnl = 0

    for instrument in instruments:
        logger.info(f"\n{'─' * 40}")
        logger.info(f"TESTING: {instrument}")
        logger.info(f"{'─' * 40}")

        results = run_single(args, instrument)
        if results is None:
            continue

        all_results[instrument] = results
        total_pnl += results.get("net_pnl", 0)
        if results.get("trades"):
            all_trades.extend(results["trades"])

        trades = results["total_trades"]
        if trades > 0:
            logger.info(
                f"  {instrument}: {trades} trades | "
                f"Win: {results.get('win_rate', 0):.1f}% | "
                f"P&L: ${results.get('net_pnl', 0):,.2f} | "
                f"PF: {results.get('profit_factor', 0):.2f} | "
                f"Sharpe: {results.get('sharpe_ratio', 0):.2f}"
            )
        else:
            logger.info(f"  {instrument}: 0 trades")

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
