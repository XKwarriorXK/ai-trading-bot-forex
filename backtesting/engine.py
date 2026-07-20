"""
Backtesting Engine — institutional-grade exit management.
Partial profit scaling, time stops, adverse excursion, ATR trailing.
Per-pair profiles override default exit parameters when configured.
"""
import logging
import numpy as np
import pandas as pd
from config.settings import INSTRUMENTS, PAIR_PROFILES, SWING_EXIT, PROP_FIRM

logger = logging.getLogger(__name__)

DEFAULT_EXIT = {
    "tp1_r": 1.5, "tp1_pct": 0.33,
    "tp2_r": 2.5, "tp2_pct": 0.25,
    "tp3_r": 4.0, "tp3_pct": 0.15,
    "adverse_r": 0.6, "adverse_bars": 3,
    "time_stop_bars": 30, "time_stop_r": 0.3,
}


class BacktestEngine:
    TIMEFRAME_SECONDS = {
        "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
        "H1": 3600, "H4": 14400, "D": 86400,
    }

    def __init__(self, pipeline, risk, instrument: str = "EUR_USD",
                 mode: str = "fast", timeframe: str = "H1",
                 style: str = "scalp", daily_data=None):
        self.pipeline = pipeline
        self.risk = risk
        self.instrument = instrument
        self.mode = mode
        self.style = style
        self.timeframe = timeframe
        self.spec = INSTRUMENTS.get(instrument, INSTRUMENTS["EUR_USD"])
        self.pip_value = 10 ** self.spec["pip_location"]

        if style == "swing":
            self.exit_params = dict(SWING_EXIT)
        else:
            profile = PAIR_PROFILES.get(instrument, {})
            self.exit_params = {k: profile.get(k, DEFAULT_EXIT[k]) for k in DEFAULT_EXIT}

        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.initial_balance = risk.account_balance
        self.balance = self.initial_balance

        self.daily_trend_cache = {}
        self.daily_zones = []
        if daily_data is not None and not daily_data.empty:
            self._precompute_daily_trend(daily_data)
            self._precompute_daily_zones(daily_data)

    def run(self, data: pd.DataFrame, lookback: int = 200) -> dict:
        if self.style == "swing":
            lookback = max(lookback, 250)
        logger.info(f"Backtesting {self.instrument} | {len(data)} bars | "
                   f"Mode: {self.mode} | Style: {self.style} | Balance: ${self.balance:,.2f}")

        self.equity_curve = [{"bar": 0, "equity": self.balance}]

        import ta
        self.atr = ta.volatility.average_true_range(
            data["high"], data["low"], data["close"], window=14)

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

            if self.risk.account_terminated:
                logger.critical(
                    f"ACCOUNT TERMINATED at bar {i} | {self.risk.termination_reason} | "
                    f"Balance: ${self.balance:,.2f}")
                break

            if not self.current_position:
                bar_ts = data.index[i] if hasattr(data.index[i], 'hour') else None
                price_data = {
                    "bid": float(current_bar["close"]),
                    "spread": self.spec["spread_avg"] * self.pip_value,
                    "timestamp": bar_ts,
                }

                if self.style == "swing" and self.daily_zones:
                    zone = self._get_active_zone(float(current_bar["close"]), data.index[i])
                    if zone:
                        price_data["zone"] = zone

                result = self.pipeline.evaluate(
                    window, self.instrument,
                    price_data=price_data,
                    mode=self.mode,
                    style=self.style,
                )

                if result["final_decision"] in ("BUY", "SELL"):
                    if self.style == "swing" and self.daily_trend_cache:
                        daily = self._get_daily_trend(data.index[i])
                        if daily and daily != result["final_decision"]:
                            at_zone = result.get("at_zone", False) or price_data.get("zone") is not None
                            if at_zone:
                                result["confidence"] = round(result["confidence"] * 0.85, 4)
                                logger.info(
                                    f"DAILY PENALTY | {self.instrument} | "
                                    f"H4 {result['final_decision']} vs Daily {daily} — "
                                    f"zone present, conf reduced to {result['confidence']:.0%}")
                            else:
                                logger.info(
                                    f"DAILY FILTER | {self.instrument} | "
                                    f"H4 {result['final_decision']} blocked — Daily trend is {daily}")
                                continue

                    if i + 1 < len(data):
                        self._open_position(result, data.iloc[i + 1], i + 1)

            if i % 500 == 0:
                logger.info(f"  Bar {i}/{len(data)} | Balance: ${self.balance:,.2f} | "
                           f"Trades: {len(self.trades)}")

        if self.current_position:
            self._force_close(data.iloc[-1], len(data) - 1)

        return self._generate_report(data)

    def _precompute_daily_trend(self, daily_data):
        close = daily_data["close"]
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        for i in range(len(daily_data)):
            if pd.isna(ma50.iloc[i]) or pd.isna(ma200.iloc[i]):
                continue
            date = daily_data.index[i].date()
            ma50_val = float(ma50.iloc[i])
            ma200_val = float(ma200.iloc[i])

            if ma50_val > ma200_val:
                self.daily_trend_cache[date] = "BUY"
            else:
                self.daily_trend_cache[date] = "SELL"

        logger.info(f"Daily trend computed: {len(self.daily_trend_cache)} days cached")

    def _get_daily_trend(self, timestamp):
        bar_date = timestamp.date() if hasattr(timestamp, 'date') else None
        if bar_date is None:
            return None
        latest = None
        for d in sorted(self.daily_trend_cache.keys()):
            if d <= bar_date:
                latest = self.daily_trend_cache[d]
            else:
                break
        return latest

    def _precompute_daily_zones(self, daily_data):
        import ta as ta_lib
        atr = ta_lib.volatility.average_true_range(
            daily_data["high"], daily_data["low"], daily_data["close"], window=14)

        for i in range(1, len(daily_data) - 1):
            if pd.isna(atr.iloc[i]):
                continue
            atr_val = float(atr.iloc[i])
            if atr_val == 0:
                continue

            curr_body = abs(float(daily_data["close"].iloc[i]) - float(daily_data["open"].iloc[i]))
            next_body = abs(float(daily_data["close"].iloc[i + 1]) - float(daily_data["open"].iloc[i + 1]))

            if curr_body > atr_val * 0.5:
                continue
            if next_body < atr_val * 1.5:
                continue

            zone_top = max(float(daily_data["high"].iloc[i]), float(daily_data["high"].iloc[max(0, i - 1)]))
            zone_bottom = min(float(daily_data["low"].iloc[i]), float(daily_data["low"].iloc[max(0, i - 1)]))
            zone_date = daily_data.index[i].date()

            if float(daily_data["close"].iloc[i + 1]) > float(daily_data["open"].iloc[i + 1]):
                zone_type = "demand"
            else:
                zone_type = "supply"

            self.daily_zones.append({
                "type": zone_type,
                "top": zone_top,
                "bottom": zone_bottom,
                "date": zone_date,
                "strength": round(next_body / atr_val, 1),
            })

        logger.info(f"Daily S/D zones: {len(self.daily_zones)} found "
                    f"({sum(1 for z in self.daily_zones if z['type'] == 'demand')} demand, "
                    f"{sum(1 for z in self.daily_zones if z['type'] == 'supply')} supply)")

    def _get_active_zone(self, price, timestamp):
        bar_date = timestamp.date() if hasattr(timestamp, 'date') else None
        if not bar_date or not self.daily_zones:
            return None

        atr_buffer = abs(price) * 0.002

        for zone in reversed(self.daily_zones):
            if zone["date"] >= bar_date:
                continue
            age_days = (bar_date - zone["date"]).days
            if age_days > 120:
                continue

            if zone["type"] == "demand":
                if zone["bottom"] - atr_buffer <= price <= zone["top"] + atr_buffer:
                    return zone
            elif zone["type"] == "supply":
                if zone["bottom"] - atr_buffer <= price <= zone["top"] + atr_buffer:
                    return zone

        return None

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

        # Swing mode: structure-based stop loss
        struct = signal.get("structure") if self.style == "swing" else None
        if self.style == "swing" and struct:
            atr_val = signal.get("indicators", {}).get("atr", 0)
            buffer = atr_val * 0.5 if atr_val > 0 else self.pip_value * 20
            if signal["final_decision"] == "BUY" and struct.get("last_swing_low"):
                sl_price = struct["last_swing_low"] - buffer
            elif signal["final_decision"] == "SELL" and struct.get("last_swing_high"):
                sl_price = struct["last_swing_high"] + buffer

        if not sl_price:
            if signal["final_decision"] == "BUY":
                sl_price = entry_price - (sl_pips * self.pip_value)
                tp_price = entry_price + (tp_pips * self.pip_value)
            else:
                sl_price = entry_price + (sl_pips * self.pip_value)
                tp_price = entry_price - (tp_pips * self.pip_value)

        units = abs(signal.get("units", 1000))
        risk_pips = abs(entry_price - sl_price) / self.pip_value

        tp1_r = self.exit_params["tp1_r"]
        tp2_r = self.exit_params.get("tp2_r", 2.5)
        tp3_r = self.exit_params.get("tp3_r", 4.0)
        if signal["final_decision"] == "BUY":
            tp1_price = entry_price + (risk_pips * tp1_r * self.pip_value)
            tp2_price = entry_price + (risk_pips * tp2_r * self.pip_value)
            tp3_price = entry_price + (risk_pips * tp3_r * self.pip_value)
        else:
            tp1_price = entry_price - (risk_pips * tp1_r * self.pip_value)
            tp2_price = entry_price - (risk_pips * tp2_r * self.pip_value)
            tp3_price = entry_price - (risk_pips * tp3_r * self.pip_value)

        self.current_position = {
            "direction": signal["final_decision"],
            "entry_price": entry_price,
            "original_units": units,
            "units_remaining": units,
            "stop_loss": sl_price,
            "original_sl": sl_price,
            "take_profit": tp_price,
            "entry_bar": bar_idx,
            "confidence": signal.get("confidence", 0),
            "highest": entry_price,
            "lowest": entry_price,
            "risk_pips": risk_pips,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "tp3_price": tp3_price,
            "tp1_hit": False,
            "tp2_hit": False,
            "tp3_hit": False,
            "at_breakeven": False,
            "total_partial_pnl": 0,
            "partial_closes": 0,
            "best_r_multiple": 0,
        }

    def _check_exit(self, bar, bar_idx):
        pos = self.current_position
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        bars_held = bar_idx - pos["entry_bar"]

        if pos["direction"] == "BUY":
            if high > pos["highest"]:
                pos["highest"] = high
            current_r = (pos["highest"] - pos["entry_price"]) / (pos["risk_pips"] * self.pip_value) if pos["risk_pips"] > 0 else 0
        else:
            if low < pos["lowest"]:
                pos["lowest"] = low
            current_r = (pos["entry_price"] - pos["lowest"]) / (pos["risk_pips"] * self.pip_value) if pos["risk_pips"] > 0 else 0

        if current_r > pos["best_r_multiple"]:
            pos["best_r_multiple"] = current_r

        # === CUTTING LOSERS ===

        # 1. STOP LOSS — hard protection, closes ALL remaining units
        if pos["direction"] == "BUY":
            if low <= pos["stop_loss"]:
                reason = "trailing_stop" if pos["tp1_hit"] else ("breakeven_stop" if pos["at_breakeven"] else "stop_loss")
                self._close_full(pos["stop_loss"], bar_idx, reason)
                return
        else:
            if high >= pos["stop_loss"]:
                reason = "trailing_stop" if pos["tp1_hit"] else ("breakeven_stop" if pos["at_breakeven"] else "stop_loss")
                self._close_full(pos["stop_loss"], bar_idx, reason)
                return

        # 2. ADVERSE EXCURSION — entry was wrong, cut early
        ae_bars = self.exit_params["adverse_bars"]
        ae_r = self.exit_params["adverse_r"]
        if bars_held <= ae_bars and not pos["at_breakeven"]:
            if pos["direction"] == "BUY":
                adverse_pips = (pos["entry_price"] - low) / self.pip_value
            else:
                adverse_pips = (high - pos["entry_price"]) / self.pip_value

            if adverse_pips >= pos["risk_pips"] * ae_r:
                self._close_full(close, bar_idx, "adverse_excursion")
                return

        # 3. TIME STOP — trade going nowhere, dead money
        ts_bars = self.exit_params["time_stop_bars"]
        ts_r = self.exit_params["time_stop_r"]
        if bars_held >= ts_bars and not pos["tp1_hit"]:
            if pos["direction"] == "BUY":
                pips_profit = (close - pos["entry_price"]) / self.pip_value
            else:
                pips_profit = (pos["entry_price"] - close) / self.pip_value

            if pips_profit < pos["risk_pips"] * ts_r:
                self._close_full(close, bar_idx, "time_stop")
                return

        # === LETTING WINNERS RUN ===

        # 4. TP1 — partial close, move stop to breakeven
        if not pos["tp1_hit"]:
            tp1_triggered = False
            if pos["direction"] == "BUY":
                tp1_triggered = high >= pos["tp1_price"]
            else:
                tp1_triggered = low <= pos["tp1_price"]

            if tp1_triggered:
                units_to_close = int(pos["original_units"] * self.exit_params["tp1_pct"])
                if units_to_close > 0:
                    self._close_partial(pos["tp1_price"], bar_idx, "tp1_partial", units_to_close)
                pos["tp1_hit"] = True
                pos["at_breakeven"] = True
                pos["stop_loss"] = pos["entry_price"]

        # 5. TP2 — close chunk, tighten trail
        tp2_pct = self.exit_params.get("tp2_pct", 0.25)
        if pos["tp1_hit"] and not pos["tp2_hit"]:
            tp2_triggered = False
            if pos["direction"] == "BUY":
                tp2_triggered = high >= pos["tp2_price"]
            else:
                tp2_triggered = low <= pos["tp2_price"]

            if tp2_triggered:
                units_to_close = int(pos["original_units"] * tp2_pct)
                if units_to_close > 0 and pos["units_remaining"] > units_to_close:
                    self._close_partial(pos["tp2_price"], bar_idx, "tp2_partial", units_to_close)
                pos["tp2_hit"] = True

        # 6. TP3 — close another chunk, let remainder ride
        tp3_pct = self.exit_params.get("tp3_pct", 0.15)
        if pos["tp2_hit"] and not pos["tp3_hit"]:
            tp3_triggered = False
            if pos["direction"] == "BUY":
                tp3_triggered = high >= pos["tp3_price"]
            else:
                tp3_triggered = low <= pos["tp3_price"]

            if tp3_triggered:
                units_to_close = int(pos["original_units"] * tp3_pct)
                if units_to_close > 0 and pos["units_remaining"] > units_to_close:
                    self._close_partial(pos["tp3_price"], bar_idx, "tp3_partial", units_to_close)
                pos["tp3_hit"] = True

        # 7. ATR TRAILING STOP — ratchets up as price runs
        if pos["at_breakeven"] and bar_idx < len(self.atr) and not pd.isna(self.atr.iloc[bar_idx]):
            atr_val = float(self.atr.iloc[bar_idx])

            if pos["tp2_hit"]:
                trail_multiplier = 1.5
            else:
                trail_multiplier = 2.0

            trail_distance = atr_val * trail_multiplier

            if pos["direction"] == "BUY":
                new_trail = pos["highest"] - trail_distance
                if new_trail > pos["stop_loss"]:
                    pos["stop_loss"] = new_trail
            else:
                new_trail = pos["lowest"] + trail_distance
                if new_trail < pos["stop_loss"]:
                    pos["stop_loss"] = new_trail

        # 8. RUNNER CLEANUP — if only tiny units left and past 3R, take profit
        if pos["tp3_hit"] and pos["units_remaining"] <= int(pos["original_units"] * 0.15):
            if current_r >= 5.0:
                self._close_full(close, bar_idx, "runner_exit_5R")
                return

    def _close_partial(self, exit_price, bar_idx, reason, units_to_close):
        pos = self.current_position
        if pos["direction"] == "BUY":
            pnl_pips = (exit_price - pos["entry_price"]) / self.pip_value
        else:
            pnl_pips = (pos["entry_price"] - exit_price) / self.pip_value

        pnl_usd = pnl_pips * self.pip_value * units_to_close
        self.balance += pnl_usd
        pos["units_remaining"] -= units_to_close
        pos["total_partial_pnl"] += pnl_usd
        pos["partial_closes"] += 1

        self.trades.append({
            "instrument": self.instrument,
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "units": units_to_close,
            "pnl": round(pnl_usd, 2),
            "pnl_pips": round(pnl_pips, 1),
            "reason": reason,
            "confidence": pos["confidence"],
            "bars_held": bar_idx - pos["entry_bar"],
            "partial": True,
        })
        self.equity_curve.append({"bar": bar_idx, "equity": self.balance})
        self.risk.record_trade_result(pnl_usd)

    def _close_full(self, exit_price, bar_idx, reason):
        pos = self.current_position
        units = pos["units_remaining"]

        if units <= 0:
            self.current_position = None
            return

        if pos["direction"] == "BUY":
            pnl_pips = (exit_price - pos["entry_price"]) / self.pip_value
        else:
            pnl_pips = (pos["entry_price"] - exit_price) / self.pip_value

        pnl_usd = pnl_pips * self.pip_value * units
        self.balance += pnl_usd

        self.trades.append({
            "instrument": self.instrument,
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "units": units,
            "pnl": round(pnl_usd, 2),
            "pnl_pips": round(pnl_pips, 1),
            "reason": reason,
            "confidence": pos["confidence"],
            "bars_held": bar_idx - pos["entry_bar"],
            "partial": False,
            "total_trade_pnl": round(pnl_usd + pos["total_partial_pnl"], 2),
            "partial_closes": pos["partial_closes"],
            "best_r": round(pos["best_r_multiple"], 1),
        })
        self.equity_curve.append({"bar": bar_idx, "equity": self.balance})
        self.risk.record_trade_result(pnl_usd)
        self.current_position = None

    def _force_close(self, bar, bar_idx):
        self._close_full(float(bar["close"]), bar_idx, "end_of_data")

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

        exit_reasons = {}
        for t in self.trades:
            r = t["reason"]
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        full_closes = [t for t in self.trades if not t.get("partial", False)]
        partial_closes = [t for t in self.trades if t.get("partial", False)]

        avg_bars = np.mean([t["bars_held"] for t in full_closes]) if full_closes else 0
        avg_winner_bars = np.mean([t["bars_held"] for t in full_closes if t["pnl"] > 0]) if [t for t in full_closes if t["pnl"] > 0] else 0
        avg_loser_bars = np.mean([t["bars_held"] for t in full_closes if t["pnl"] <= 0]) if [t for t in full_closes if t["pnl"] <= 0] else 0

        report = {
            "instrument": self.instrument,
            "mode": self.mode,
            "total_bars": len(data),
            "total_trades": len(self.trades),
            "full_closes": len(full_closes),
            "partial_closes": len(partial_closes),
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
            "avg_bars_held": round(avg_bars, 1),
            "avg_winner_bars": round(avg_winner_bars, 1),
            "avg_loser_bars": round(avg_loser_bars, 1),
            "exit_reasons": exit_reasons,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }

        if PROP_FIRM.get("enabled"):
            report["prop_firm"] = self.risk.get_prop_firm_status()

        return report
