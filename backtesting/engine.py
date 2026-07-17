"""
Backtesting Engine — walks through historical forex data bar by bar.
Realistic cost modeling with spread and slippage.
"""
import logging
import numpy as np
import pandas as pd
from config.settings import INSTRUMENTS

logger = logging.getLogger(__name__)


class BacktestEngine:
    TIMEFRAME_SECONDS = {
        "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
        "H1": 3600, "H4": 14400, "D": 86400,
    }

    def __init__(self, pipeline, risk, instrument: str = "EUR_USD",
                 mode: str = "fast", timeframe: str = "H1"):
        self.pipeline = pipeline
        self.risk = risk
        self.instrument = instrument
        self.mode = mode
        self.timeframe = timeframe
        self.spec = INSTRUMENTS.get(instrument, INSTRUMENTS["EUR_USD"])
        self.pip_value = 10 ** self.spec["pip_location"]

        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.initial_balance = risk.account_balance
        self.balance = self.initial_balance

    def run(self, data: pd.DataFrame, lookback: int = 200) -> dict:
        logger.info(f"Backtesting {self.instrument} | {len(data)} bars | "
                   f"Mode: {self.mode} | Balance: ${self.balance:,.2f}")

        self.equity_curve = [{"bar": 0, "equity": self.balance}]

        last_date = None
        for i in range(lookback, len(data)):
            window = data.iloc[max(0, i - lookback):i + 1].copy()
            current_bar = data.iloc[i]

            bar_date = data.index[i].date() if hasattr(data.index[i], 'date') else None
            if bar_date and bar_date != last_date:
                self.risk._reset_daily(bar_date)
                last_date = bar_date

            if self.current_position:
                self._check_exit(current_bar, i)

            if not self.current_position:
                result = self.pipeline.evaluate(
                    window, self.instrument,
                    price_data={"bid": float(current_bar["close"]), "spread": self.spec["spread_avg"] * self.pip_value},
                    mode=self.mode,
                )

                if result["final_decision"] in ("BUY", "SELL"):
                    if i + 1 < len(data):
                        self._open_position(result, data.iloc[i + 1], i + 1)

            if i % 500 == 0:
                logger.info(f"  Bar {i}/{len(data)} | Balance: ${self.balance:,.2f} | "
                           f"Trades: {len(self.trades)}")

        if self.current_position:
            self._force_close(data.iloc[-1], len(data) - 1)

        return self._generate_report(data)

    def _open_position(self, signal, entry_bar, bar_idx):
        entry_price = float(entry_bar["open"])
        spread_cost = self.spec["spread_avg"] * self.pip_value / 2

        if signal["final_decision"] == "BUY":
            entry_price += spread_cost
        else:
            entry_price -= spread_cost

        sl_price = signal.get("stop_loss_price")
        tp_price = signal.get("take_profit_price")
        sl_pips = signal.get("stop_loss_pips", 30)
        tp_pips = signal.get("take_profit_pips", 60)

        if not sl_price:
            if signal["final_decision"] == "BUY":
                sl_price = entry_price - (sl_pips * self.pip_value)
                tp_price = entry_price + (tp_pips * self.pip_value)
            else:
                sl_price = entry_price + (sl_pips * self.pip_value)
                tp_price = entry_price - (tp_pips * self.pip_value)

        units = abs(signal.get("units", 1000))

        self.current_position = {
            "direction": signal["final_decision"],
            "entry_price": entry_price,
            "units": units,
            "stop_loss": sl_price,
            "original_sl": sl_price,
            "take_profit": tp_price,
            "entry_bar": bar_idx,
            "confidence": signal.get("confidence", 0),
            "highest": entry_price,
            "lowest": entry_price,
        }

    def _check_exit(self, bar, bar_idx):
        pos = self.current_position
        high = float(bar["high"])
        low = float(bar["low"])

        if pos["direction"] == "BUY":
            if high >= pos["take_profit"]:
                self._close(pos["take_profit"], bar_idx, "take_profit")
            elif low <= pos["stop_loss"]:
                self._close(pos["stop_loss"], bar_idx, "stop_loss")
        else:
            if low <= pos["take_profit"]:
                self._close(pos["take_profit"], bar_idx, "take_profit")
            elif high >= pos["stop_loss"]:
                self._close(pos["stop_loss"], bar_idx, "stop_loss")

    def _close(self, exit_price, bar_idx, reason):
        pos = self.current_position
        if pos["direction"] == "BUY":
            pnl_pips = (exit_price - pos["entry_price"]) / self.pip_value
        else:
            pnl_pips = (pos["entry_price"] - exit_price) / self.pip_value

        pnl_usd = pnl_pips * self.pip_value * pos["units"]
        self.balance += pnl_usd

        self.trades.append({
            "instrument": self.instrument,
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "units": pos["units"],
            "pnl": round(pnl_usd, 2),
            "pnl_pips": round(pnl_pips, 1),
            "reason": reason,
            "confidence": pos["confidence"],
            "bars_held": bar_idx - pos["entry_bar"],
        })
        self.equity_curve.append({"bar": bar_idx, "equity": self.balance})
        self.risk.record_trade_result(pnl_usd)
        self.current_position = None

    def _force_close(self, bar, bar_idx):
        self._close(float(bar["close"]), bar_idx, "end_of_data")

    def _generate_report(self, data) -> dict:
        if not self.trades:
            return {"total_trades": 0, "net_pnl": 0, "message": "No trades"}

        pnls = [t["pnl"] for t in self.trades]
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        win_pnls = [t["pnl"] for t in wins]
        loss_pnls = [t["pnl"] for t in losses]

        eq = [e["equity"] for e in self.equity_curve]
        peak = eq[0]
        max_dd = 0
        for e in eq:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

        returns = pd.Series(pnls)
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        return {
            "instrument": self.instrument,
            "mode": self.mode,
            "total_bars": len(data),
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(self.trades) * 100, 1),
            "net_pnl": round(sum(pnls), 2),
            "avg_win": round(np.mean(win_pnls), 2) if win_pnls else 0,
            "avg_loss": round(np.mean(loss_pnls), 2) if loss_pnls else 0,
            "largest_win": round(max(win_pnls), 2) if win_pnls else 0,
            "largest_loss": round(min(loss_pnls), 2) if loss_pnls else 0,
            "profit_factor": round(abs(sum(win_pnls) / sum(loss_pnls)), 2) if loss_pnls and sum(loss_pnls) != 0 else 0,
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "initial_balance": self.initial_balance,
            "final_balance": round(self.balance, 2),
            "return_pct": round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
            "avg_pips": round(np.mean([t["pnl_pips"] for t in self.trades]), 1),
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }
