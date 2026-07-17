"""
AI Forex Trading Bot — OANDA + Multi-pair scanning.
Scans watchlist, finds best setups, trades via OANDA practice account.

Usage:
    python main.py              # Scan all watchlist pairs
    python main.py --mode fast  # Technical only (no AI)
    python main.py --mode full  # Full AI pipeline with debate
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
    from agents.pipeline import TradePipeline
    from execution.oanda_client import OandaClient
    from config.settings import OANDA

    oanda = OandaClient()

    acct = oanda.get_account()
    balance = acct.get("balance", 10000)

    technical = TechnicalAgent()
    session_filter = SessionFilter()
    risk = RiskManager(account_balance=balance)

    debate = None
    router = None
    if mode == "full":
        provider = AIProvider()
        router = TaskRouter(provider)
        debate = DebateAgent(router)

    pipeline = TradePipeline(
        technical=technical,
        debate=debate,
        risk=risk,
        session_filter=session_filter,
        router=router,
    )

    return oanda, pipeline, risk


def scan_watchlist(oanda, pipeline, risk, mode: str = "full"):
    from config.settings import WATCHLIST, TIMEFRAMES

    signals = []

    for instrument in WATCHLIST:
        try:
            df = oanda.fetch_candles(instrument, granularity=TIMEFRAMES["entry"], count=300)
            if df.empty:
                continue

            price_data = oanda.fetch_price(instrument)
            current_price = price_data.get("bid", 0) if price_data else 0

            result = pipeline.evaluate(df, instrument, price=current_price, mode=mode)

            if result["final_decision"] in ("BUY", "SELL"):
                signals.append(result)

        except Exception as e:
            logger.error(f"Error scanning {instrument}: {e}")

    if signals:
        signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        logger.info(f"Found {len(signals)} signal(s) this scan")
    else:
        logger.info("No signals this scan")

    return signals


def execute_signals(oanda, signals, risk):
    from config.settings import TRADING

    if TRADING["mode"] == "paper":
        for sig in signals:
            logger.info(
                f"[PAPER] Would trade {sig['instrument']}: {sig['final_decision']} "
                f"{sig['units']} units | Conf: {sig['confidence']:.0%} | "
                f"SL: {sig.get('stop_loss_price')} | TP: {sig.get('take_profit_price')}"
            )
        return

    best = signals[0]
    result = oanda.place_market_order(
        instrument=best["instrument"],
        units=best["units"],
        stop_loss_price=best.get("stop_loss_price"),
        take_profit_price=best.get("take_profit_price"),
    )

    if result["success"]:
        risk.record_trade_open(best["instrument"])
        logger.info(f"EXECUTED: {best['instrument']} {best['final_decision']} @ {result['price']}")
    else:
        logger.warning(f"Order failed: {result.get('error')}")


def run_live_loop(oanda, pipeline, risk, mode: str = "full", interval: int = 300):
    logger.info(f"Starting live scan loop | Interval: {interval}s | Mode: {mode}")

    while True:
        try:
            open_trades = oanda.get_open_trades()
            if open_trades:
                logger.info(f"Open trades: {len(open_trades)}")
                for t in open_trades:
                    logger.info(
                        f"  {t['instrument']} | {t['units']} units | "
                        f"P&L: ${t['unrealized_pl']:.2f}"
                    )

            signals = scan_watchlist(oanda, pipeline, risk, mode)

            if signals:
                execute_signals(oanda, signals, risk)

            acct = oanda.get_account()
            logger.info(
                f"Account: ${acct.get('balance', 0):,.2f} | "
                f"Unrealized: ${acct.get('unrealized_pl', 0):,.2f} | "
                f"Open: {acct.get('open_trade_count', 0)}"
            )

            logger.info(f"Sleeping {interval}s until next scan...")
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
                       help="Scan once and exit (no loop)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AI FOREX TRADING BOT")
    logger.info(f"  Mode: {args.mode} | Interval: {args.interval}s")
    logger.info("=" * 60)

    oanda, pipeline, risk = initialize(args.mode)

    health = oanda.health_check()
    acct = oanda.get_account()
    logger.info(f"OANDA: {'Connected' if health.get('connected') else 'FAILED'} "
               f"({health.get('environment', 'unknown')})")
    logger.info(f"Balance: ${acct.get('balance', 0):,.2f}")
    logger.info(f"Open trades: {acct.get('open_trade_count', 0)}")
    logger.info("=" * 60)

    if args.scan_once:
        signals = scan_watchlist(oanda, pipeline, risk, args.mode)
        if signals:
            execute_signals(oanda, signals, risk)
    else:
        run_live_loop(oanda, pipeline, risk, args.mode, args.interval)


if __name__ == "__main__":
    main()
