"""
Risk Manager — FTMO prop firm compliant.

Enforces:
    - 5% max daily loss (circuit breaker at 4.5%)
    - 10% max total loss from initial balance (circuit breaker at 8%)
    - 2% risk per trade
    - Position sizing, correlation checks
    - Account termination if hard limits breached

Daily loss is measured from start-of-day balance (FTMO standard).
Total loss is measured from initial account balance.
"""
import logging
from datetime import date
from config.settings import RISK, INSTRUMENTS, CORRELATION_GROUPS, PROP_FIRM

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, account_balance: float = 10000):
        self.initial_balance = account_balance
        self.account_balance = account_balance
        self.start_of_day_balance = account_balance
        self.daily_pnl = 0
        self.trades_today = 0
        self.current_date = date.today()
        self.open_instruments = []
        self.account_terminated = False
        self.termination_reason = None
        self.trading_days = set()
        self.peak_balance = account_balance

    def _reset_daily(self, today=None):
        today = today or date.today()
        if today != self.current_date:
            self.start_of_day_balance = self.account_balance
            self.daily_pnl = 0
            self.trades_today = 0
            self.current_date = today

    def _check_prop_firm_limits(self):
        if not PROP_FIRM.get("enabled"):
            return True, ""

        if self.account_terminated:
            return False, f"ACCOUNT TERMINATED: {self.termination_reason}"

        daily_loss_pct = abs(min(self.daily_pnl, 0)) / self.start_of_day_balance * 100
        hard_daily = PROP_FIRM["max_daily_loss_pct"]
        soft_daily = PROP_FIRM["daily_loss_buffer_pct"]

        if daily_loss_pct >= hard_daily:
            self.account_terminated = True
            self.termination_reason = (
                f"DAILY LOSS HARD LIMIT: -{daily_loss_pct:.1f}% "
                f"(limit {hard_daily}%) — account blown"
            )
            logger.critical(self.termination_reason)
            return False, self.termination_reason

        if daily_loss_pct >= soft_daily:
            return False, (
                f"Daily loss circuit breaker: -{daily_loss_pct:.1f}% "
                f"(buffer at {soft_daily}%, hard limit {hard_daily}%)"
            )

        total_loss_pct = max(0, (self.initial_balance - self.account_balance)) / self.initial_balance * 100
        hard_total = PROP_FIRM["max_total_loss_pct"]
        soft_total = PROP_FIRM["total_loss_buffer_pct"]

        if total_loss_pct >= hard_total:
            self.account_terminated = True
            self.termination_reason = (
                f"TOTAL LOSS HARD LIMIT: -{total_loss_pct:.1f}% "
                f"(limit {hard_total}%) — account blown"
            )
            logger.critical(self.termination_reason)
            return False, self.termination_reason

        if total_loss_pct >= soft_total:
            return False, (
                f"Total loss circuit breaker: -{total_loss_pct:.1f}% "
                f"(buffer at {soft_total}%, hard limit {hard_total}%)"
            )

        return True, ""

    def _would_breach_daily(self, risk_amount):
        if not PROP_FIRM.get("enabled"):
            return False
        potential_loss = abs(min(self.daily_pnl, 0)) + risk_amount
        potential_pct = potential_loss / self.start_of_day_balance * 100
        return potential_pct >= PROP_FIRM["daily_loss_buffer_pct"]

    def check_trade(self, instrument: str, signal: str, confidence: float,
                    atr: float = 0, price: float = 0, regime: str = "") -> dict:
        self._reset_daily()

        prop_ok, prop_reason = self._check_prop_firm_limits()
        if not prop_ok:
            return {"approved": False, "reason": prop_reason}

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

        if self._would_breach_daily(risk_amount):
            remaining = (self.start_of_day_balance * PROP_FIRM["daily_loss_buffer_pct"] / 100) - abs(min(self.daily_pnl, 0))
            if remaining > 0:
                risk_amount = remaining * 0.8
                logger.warning(
                    f"PROP FIRM | Reducing risk to ${risk_amount:.2f} to protect daily limit"
                )
            else:
                return {"approved": False, "reason": "No remaining daily risk budget"}

        units = int(risk_amount / (stop_pips * pip_value)) if stop_pips > 0 else 0

        if signal == "SELL":
            units = -units

        if regime == "trending":
            tp_pips = stop_pips * 2.5
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
        self.trading_days.add(self.current_date)
        if instrument not in self.open_instruments:
            self.open_instruments.append(instrument)

    def record_trade_close(self, instrument: str, pnl: float):
        self.daily_pnl += pnl
        self.account_balance += pnl
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance
        if instrument in self.open_instruments:
            self.open_instruments.remove(instrument)

        self._check_prop_firm_limits()

    def record_trade_result(self, pnl: float):
        self.daily_pnl += pnl
        self.account_balance += pnl
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance
        self.trading_days.add(self.current_date)

        self._check_prop_firm_limits()

    def get_prop_firm_status(self):
        if not PROP_FIRM.get("enabled"):
            return {}

        total_pnl = self.account_balance - self.initial_balance
        return_pct = total_pnl / self.initial_balance * 100
        daily_loss_pct = abs(min(self.daily_pnl, 0)) / self.start_of_day_balance * 100
        total_loss_pct = max(0, (self.initial_balance - self.account_balance)) / self.initial_balance * 100
        max_dd = max(0, (self.peak_balance - self.account_balance)) / self.peak_balance * 100

        target = PROP_FIRM["profit_target_pct"]
        target_hit = return_pct >= target

        return {
            "firm": PROP_FIRM["name"],
            "initial_balance": self.initial_balance,
            "current_balance": round(self.account_balance, 2),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(return_pct, 2),
            "daily_loss_pct": round(daily_loss_pct, 2),
            "daily_loss_limit": PROP_FIRM["max_daily_loss_pct"],
            "total_drawdown_pct": round(total_loss_pct, 2),
            "total_loss_limit": PROP_FIRM["max_total_loss_pct"],
            "max_drawdown_pct": round(max_dd, 2),
            "trading_days": len(self.trading_days),
            "min_trading_days": PROP_FIRM["min_trading_days"],
            "target_pct": target,
            "target_hit": target_hit,
            "account_terminated": self.account_terminated,
            "termination_reason": self.termination_reason,
            "profit_split": PROP_FIRM["profit_split"],
            "your_payout": round(total_pnl * PROP_FIRM["profit_split"], 2) if total_pnl > 0 else 0,
        }
