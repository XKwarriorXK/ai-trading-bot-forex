"""
Backtest Runner — test the forex bot on historical data.

Usage:
    python backtest.py                          # EUR/USD, 180 days, fast mode
    python backtest.py --instrument GBP_USD     # Test GBP/USD
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


def main():
    parser = argparse.ArgumentParser(description="Backtest the forex bot")
    parser.add_argument("--instrument", default="EUR_USD", help="Forex pair")
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

    logger.info("=" * 60)
    logger.info("FOREX BACKTEST")
    logger.info(f"  Instrument: {args.instrument} | Days: {args.days}")
    logger.info(f"  Mode: {args.mode} | Balance: ${args.balance:,.2f}")
    logger.info("=" * 60)

    from backtesting.data_loader import fetch_oanda_historical
    data = fetch_oanda_historical(args.instrument, args.timeframe, args.days)

    if data.empty:
        logger.error("No data loaded. Check OANDA connection.")
        return

    logger.info(f"Loaded {len(data)} bars")

    from backtesting.engine import BacktestEngine
    pipeline, risk = build_pipeline(args)

    engine = BacktestEngine(pipeline, risk, args.instrument, args.mode, args.timeframe)
    results = engine.run(data)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total trades: {results['total_trades']}")
    logger.info(f"  Win rate: {results.get('win_rate', 0):.1f}%")
    logger.info(f"  Net P&L: ${results.get('net_pnl', 0):,.2f}")
    logger.info(f"  Return: {results.get('return_pct', 0):.2f}%")
    logger.info(f"  Sharpe: {results.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Max drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
    logger.info(f"  Profit factor: {results.get('profit_factor', 0):.2f}")
    logger.info(f"  Avg win: ${results.get('avg_win', 0):,.2f}")
    logger.info(f"  Avg loss: ${results.get('avg_loss', 0):,.2f}")
    logger.info(f"  Avg pips: {results.get('avg_pips', 0):.1f}")
    logger.info(f"  Balance: ${results.get('initial_balance', 0):,.2f} → ${results.get('final_balance', 0):,.2f}")

    if args.monte_carlo and results.get("trades"):
        logger.info("\n" + "=" * 60)
        logger.info("MONTE CARLO RISK ANALYSIS")
        logger.info("=" * 60)

        from backtesting.monte_carlo import MonteCarloAnalyzer
        mc = MonteCarloAnalyzer(simulations=args.mc_sims)
        mc_results = mc.analyze(results["trades"], args.balance)

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
