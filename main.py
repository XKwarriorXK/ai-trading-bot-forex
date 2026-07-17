"""
AI Forex Trading Bot — Institutional-grade multi-pair scanner.
Full pipeline: Spread → Session → Multi-TF → Structure → Strategy Ensemble →
News → AI Debate → Risk → Execute → Journal → Learn

Usage:
    python main.py              # Full AI mode, all pairs
    python main.py --mode fast  # Technical only (no AI)
    python main.py --scan-once  # Single scan, no loop
"""
import sys
import os
import time
import logging
import argparse

os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/trading.log"),
    ],
)
logger = logging.getLogger("ForexBot")


def initialize(mode: str = "full"):
    from brain.providers import AIProvider
    from brain.router import TaskRouter
    from agents.technical import TechnicalAgent
    from agents.debate import DebateAgent
    from agents.risk_manager import RiskManager
    from agents.session_filter import SessionFilter
    from agents.spread_filter import SpreadFilter
    from agents.news_agent import NewsAgent
    from agents.market_structure import MarketStructureAgent
    from agents.multi_timeframe import MultiTimeframeAgent
    from agents.entry_sniper import EntrySniper
    from agents.pipeline import TradePipeline
    from strategy.strategy_selector import StrategySelector
    from execution.oanda_client import OandaClient
    from data.trade_journal import TradeJournal

    logger.info("=" * 60)
    logger.info("AI FOREX TRADING BOT — INITIALIZING")
    logger.info("=" * 60)

    # Execution
    oanda = OandaClient()
    acct = oanda.get_account()
    balance = acct.get("balance", 10000)
    logger.info(f"[Exchange] OANDA connected | Balance: ${balance:,.2f}")

    # Data
    journal = TradeJournal()
    logger.info("[Data] Trade journal initialized")

    # Technical + Structure
    technical = TechnicalAgent()
    structure = MarketStructureAgent()
    logger.info("[Analysis] Technical + Market Structure agents online")

    # Strategy
    strategy_selector = StrategySelector()
    logger.info("[Strategy] 4 strategies loaded (trend, mean_reversion, breakout, momentum)")

    # Filters
    session_filter = SessionFilter()
    spread_filter = SpreadFilter()
    news_agent = NewsAgent()
    logger.info("[Filters] Session + Spread + News filters online")

    # Multi-timeframe
    multi_tf = MultiTimeframeAgent(oanda)
    logger.info("[MTF] Multi-timeframe analysis online")

    # Entry sniper — drops to M15 for precise entries
    entry_sniper = EntrySniper(oanda)
    logger.info("[Sniper] M15 entry refinement online")

    # AI Brain
    debate = None
    router = None
    if mode == "full":
        provider = AIProvider()
        router = TaskRouter(provider)
        debate = DebateAgent(router)
        logger.info("[AI] Brain + Debate system online")
    else:
        logger.info("[AI] Skipped (fast mode)")

    # Risk
    risk = RiskManager(account_balance=balance)
    logger.info(f"[Risk] Manager online | Max daily loss: {risk.account_balance * 0.02:.2f}")

    # Pipeline
    pipeline = TradePipeline(
        technical=technical,
        strategy_selector=strategy_selector,
        debate=debate,
        risk=risk,
        session_filter=session_filter,
        spread_filter=spread_filter,
        news_agent=news_agent,
        market_structure=structure,
        multi_tf=multi_tf,
        journal=journal,
        router=router,
        entry_sniper=entry_sniper,
    )
    logger.info("[Pipeline] Full analysis pipeline assembled")

    # Performance summary
    perf = journal.get_performance_summary()
    if perf.get("total_trades", 0) > 0:
        logger.info(
            f"[History] {perf['total_trades']} trades | "
            f"Win rate: {perf['win_rate']}% | P&L: ${perf['total_pnl']:.2f}"
        )

    logger.info("=" * 60)
    logger.info("ALL SYSTEMS ONLINE")
    logger.info("=" * 60)

    return oanda, pipeline, risk, journal


def scan_watchlist(oanda, pipeline, risk, journal, mode: str = "full"):
    from config.settings import WATCHLIST, TIMEFRAMES

    signals = []

    for instrument in WATCHLIST:
        try:
            df = oanda.fetch_candles(instrument, granularity=TIMEFRAMES["entry"], count=300)
            if df.empty:
                continue

            price_data = oanda.fetch_price(instrument)

            result = pipeline.evaluate(df, instrument, price_data=price_data, mode=mode)

            if result["final_decision"] in ("BUY", "SELL"):
                signals.append(result)

        except Exception as e:
            logger.error(f"Error scanning {instrument}: {e}")

    if signals:
        signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        logger.info(f"Scan complete: {len(signals)} signal(s) found")
    else:
        logger.info("Scan complete: no signals")

    return signals


def execute_signals(oanda, signals, risk, journal):
    from config.settings import TRADING, RISK

    open_trades = oanda.get_open_trades()
    open_count = len(open_trades)

    if open_count >= RISK["max_open_trades"]:
        logger.info(f"Max open trades ({RISK['max_open_trades']}) reached — skipping execution")
        return

    for sig in signals:
        if open_count >= RISK["max_open_trades"]:
            break

        if TRADING["mode"] == "paper":
            logger.info(
                f"[PAPER] {sig['instrument']}: {sig['final_decision']} "
                f"{sig['units']} units | Conf: {sig['confidence']:.0%} | "
                f"Regime: {sig.get('regime')} | "
                f"Strategies: {sig.get('strategies', [])} | "
                f"SL: {sig.get('stop_loss_price')} | TP: {sig.get('take_profit_price')}"
            )
            journal.log_signal(
                sig["instrument"], sig["final_decision"], sig["confidence"],
                sig.get("regime", ""), sig.get("reasons", []),
                executed=False, skip_reason="paper_mode",
            )
            continue

        result = oanda.place_market_order(
            instrument=sig["instrument"],
            units=sig["units"],
            stop_loss_price=sig.get("stop_loss_price"),
            take_profit_price=sig.get("take_profit_price"),
        )

        if result["success"]:
            risk.record_trade_open(sig["instrument"])
            open_count += 1
            journal.log_trade_open(
                instrument=sig["instrument"],
                direction=sig["final_decision"],
                units=sig["units"],
                entry_price=result["price"],
                stop_loss=sig.get("stop_loss_price", 0),
                take_profit=sig.get("take_profit_price", 0),
                confidence=sig["confidence"],
                regime=sig.get("regime", ""),
                strategies=sig.get("strategies", []),
                reasons=sig.get("reasons", []),
                session=str(sig.get("session", [])),
                trade_id=result.get("trade_id"),
            )
            logger.info(
                f"EXECUTED: {sig['instrument']} {sig['final_decision']} "
                f"{sig['units']} units @ {result['price']}"
            )
        else:
            logger.warning(f"Order failed for {sig['instrument']}: {result.get('error')}")


def check_drawdown(risk, journal):
    from config.settings import RISK
    max_dd_pct = RISK["max_daily_loss_pct"]
    max_loss = risk.account_balance * (max_dd_pct / 100)

    if risk.daily_pnl < -max_loss:
        logger.warning(
            f"KILL SWITCH: Daily loss ${abs(risk.daily_pnl):.2f} "
            f"exceeds max ${max_loss:.2f} — trading halted"
        )
        return True
    return False


def run_live_loop(oanda, pipeline, risk, journal, mode: str = "full", interval: int = 300):
    from config.settings import WATCHLIST

    logger.info(f"Live loop starting | {len(WATCHLIST)} pairs | "
               f"Interval: {interval}s | Mode: {mode}")

    while True:
        try:
            if check_drawdown(risk, journal):
                logger.info("Waiting for next day to resume...")
                time.sleep(3600)
                continue

            open_trades = oanda.get_open_trades()
            if open_trades:
                logger.info(f"Open trades: {len(open_trades)}")
                for t in open_trades:
                    logger.info(
                        f"  {t['instrument']} | {t['units']} units | "
                        f"P&L: ${t['unrealized_pl']:.2f}"
                    )

            signals = scan_watchlist(oanda, pipeline, risk, journal, mode)

            if signals:
                execute_signals(oanda, signals, risk, journal)

            acct = oanda.get_account()
            logger.info(
                f"Account: ${acct.get('balance', 0):,.2f} | "
                f"Unrealized: ${acct.get('unrealized_pl', 0):,.2f} | "
                f"Open: {acct.get('open_trade_count', 0)} | "
                f"Daily P&L: ${risk.daily_pnl:,.2f}"
            )

            logger.info(f"Next scan in {interval}s...")
            time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="AI Forex Trading Bot")
    parser.add_argument("--mode", default="full", choices=["fast", "full"],
                       help="fast=technical only, full=with AI debate")
    parser.add_argument("--interval", type=int, default=300,
                       help="Seconds between scans (default 300)")
    parser.add_argument("--scan-once", action="store_true",
                       help="Scan once and exit")
    args = parser.parse_args()

    oanda, pipeline, risk, journal = initialize(args.mode)

    if args.scan_once:
        signals = scan_watchlist(oanda, pipeline, risk, journal, args.mode)
        if signals:
            execute_signals(oanda, signals, risk, journal)
    else:
        run_live_loop(oanda, pipeline, risk, journal, args.mode, args.interval)


if __name__ == "__main__":
    main()
