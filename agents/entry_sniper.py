"""
Entry Sniper — drops to M15 to find precise entries after H1 signals a setup.
H1 says WHAT to trade, M15 says WHEN to enter.
"""
import logging
import ta

logger = logging.getLogger(__name__)


class EntrySniper:
    def __init__(self, oanda_client):
        self.oanda = oanda_client

    def snipe_entry(self, instrument: str, signal: str, h1_price: float,
                    h1_atr: float = 0) -> dict:
        m15 = self.oanda.fetch_candles(instrument, "M15", 50)
        if m15 is None or m15.empty or len(m15) < 20:
            return {"confirmed": True, "reason": "No M15 data, using H1 entry"}

        close = m15["close"]
        high = m15["high"]
        low = m15["low"]
        open_ = m15["open"]

        rsi = ta.momentum.rsi(close, window=14)
        ema_9 = ta.trend.ema_indicator(close, window=9)
        ema_21 = ta.trend.ema_indicator(close, window=21)

        price = close.iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_close = close.iloc[-1]
        last_open = open_.iloc[-1]
        prev_close = close.iloc[-2]

        score = 0
        reasons = []

        if signal == "BUY":
            if last_close > last_open:
                score += 1
                reasons.append("M15 bullish candle")
            if last_close > prev_close:
                score += 1
                reasons.append("M15 higher close")
            if ema_9.iloc[-1] > ema_21.iloc[-1]:
                score += 1
                reasons.append("M15 EMA bullish")
            if 30 < last_rsi < 60:
                score += 1
                reasons.append("M15 RSI has room to run")
            if price <= ema_21.iloc[-1] * 1.002:
                score += 1
                reasons.append("M15 pullback to EMA21")
        else:
            if last_close < last_open:
                score += 1
                reasons.append("M15 bearish candle")
            if last_close < prev_close:
                score += 1
                reasons.append("M15 lower close")
            if ema_9.iloc[-1] < ema_21.iloc[-1]:
                score += 1
                reasons.append("M15 EMA bearish")
            if 40 < last_rsi < 70:
                score += 1
                reasons.append("M15 RSI has room to fall")
            if price >= ema_21.iloc[-1] * 0.998:
                score += 1
                reasons.append("M15 pullback to EMA21")

        confirmed = score >= 3
        m15_atr = ta.volatility.average_true_range(high, low, close, window=14)
        sniper_atr = float(m15_atr.iloc[-1]) if not m15_atr.empty else 0

        return {
            "confirmed": confirmed,
            "score": score,
            "max_score": 5,
            "reasons": reasons,
            "m15_price": round(price, 5),
            "m15_rsi": round(last_rsi, 1),
            "m15_atr": round(sniper_atr, 6),
            "reject_reason": f"M15 score {score}/5 (need 3)" if not confirmed else None,
        }
