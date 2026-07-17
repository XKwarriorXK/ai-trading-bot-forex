"""
Risk Manager — position sizing, daily limits, correlation checks for forex.
"""
import logging
from datetime import date
from config.settings import RISK, INSTRUMENTS, CORRELATION_GROUPS

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, account_balance: float = 10000):
        self.account_balance = account_balance
        self.daily_pnl = 0
        self.trades_today = 0
        self.current_date = date.today()
        self.open_instruments = []

    def _reset_daily(self, today=None):
        today = today or date.today()
        if today != self.current_date:
            self.daily_pnl = 0
            self.trades_today = 0
            self.current_date = today

    def check_trade(self, instrument: str, signal: str, confidence: float,
                    atr: float = 0, price: float = 0, regime: str = "") -> dict:
        self._reset_daily()

        if self.trades_today >= RISK["max_trades_per_day"]:
            return {"approved": False, "reason": f"Daily trade limit ({RISK['max_trades_per_day']})"}

        max_loss = self.account_balance * (RISK["max_daily_loss_pct"] / 100)
        if abs(self.daily_pnl) >= max_loss and self.daily_pnl < 0:
            return {"approved": False, "reason": "Daily loss limit hit"}

        if len(self.open_instruments) >= RISK["max_open_trades"]:
            return {"approved": False, "reason": f"Max open trades ({RISK['max_open_trades']})"}

        correlated = self._check_correlation(instrument)
        if correlated >= RISK["max_correlated_trades"]:
            return {"approved": False, "reason": f"Too many correlated trades ({correlated})"}

        spec = INSTRUMENTS.get(instrument, {})
        pip_loc = spec.get("pip_location", -4)
        pip_value = 10 ** pip_loc

        if atr > 0:
            stop_pips = max(atr / pip_value * 2.0, 15)
        else:
            stop_pips = RISK["default_stop_loss_pips"]

        risk_amount = self.account_balance * (RISK["risk_per_trade_pct"] / 100)
        units = int(risk_amount / (stop_pips * pip_value)) if stop_pips > 0 else 0

        if signal == "SELL":
            units = -units

        if regime == "trending":
            tp_pips = stop_pips * 3.0
        else:
            tp_pips = stop_pips * 2.0

        if price > 0:
            if signal == "BUY":
                sl_price = price - (stop_pips * pip_value)
                tp_price = price + (tp_pips * pip_value)
            else:
                sl_price = price + (stop_pips * pip_value)
                tp_price = price - (tp_pips * pip_value)
        else:
            sl_price = None
            tp_price = None

        return {
            "approved": True,
            "units": units,
            "stop_loss_pips": round(stop_pips, 1),
            "take_profit_pips": round(tp_pips, 1),
            "stop_loss_price": round(sl_price, 5) if sl_price else None,
            "take_profit_price": round(tp_price, 5) if tp_price else None,
            "risk_amount": round(risk_amount, 2),
        }

    def _check_correlation(self, instrument: str) -> int:
        count = 0
        for group_name, group_instruments in CORRELATION_GROUPS.items():
            if instrument in group_instruments:
                for open_inst in self.open_instruments:
                    if open_inst in group_instruments and open_inst != instrument:
                        count += 1
        return count

    def record_trade_open(self, instrument: str):
        self.trades_today += 1
        if instrument not in self.open_instruments:
            self.open_instruments.append(instrument)

    def record_trade_close(self, instrument: str, pnl: float):
        self.daily_pnl += pnl
        self.account_balance += pnl
        if instrument in self.open_instruments:
            self.open_instruments.remove(instrument)

    def record_trade_result(self, pnl: float):
        self.daily_pnl += pnl
        self.account_balance += pnl
