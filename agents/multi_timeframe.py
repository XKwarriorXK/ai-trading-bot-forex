"""
Multi-Timeframe Analysis — confirms signals across H1, H4, Daily.
Higher timeframes set bias, lower timeframes refine entry.
"""
import logging
import ta

logger = logging.getLogger(__name__)


class MultiTimeframeAgent:
    def __init__(self, oanda_client):
        self.oanda = oanda_client

    def analyze(self, instrument: str) -> dict:
        h1 = self.oanda.fetch_candles(instrument, "H1", 200)
        h4 = self.oanda.fetch_candles(instrument, "H4", 200)
        daily = self.oanda.fetch_candles(instrument, "D", 200)

        h1_bias = self._get_bias(h1, "H1") if not h1.empty else {"bias": "neutral"}
        h4_bias = self._get_bias(h4, "H4") if not h4.empty else {"bias": "neutral"}
        daily_bias = self._get_bias(daily, "D") if not daily.empty else {"bias": "neutral"}

        biases = [h1_bias["bias"], h4_bias["bias"], daily_bias["bias"]]
        bullish = biases.count("bullish")
        bearish = biases.count("bearish")

        if bullish >= 2:
            alignment = "bullish"
            strength = bullish / 3
        elif bearish >= 2:
            alignment = "bearish"
            strength = bearish / 3
        else:
            alignment = "mixed"
            strength = 0

        # Daily bias has highest weight
        daily_agrees = (daily_bias["bias"] == alignment) if alignment != "mixed" else False

        return {
            "alignment": alignment,
            "strength": round(strength, 2),
            "daily_agrees": daily_agrees,
            "h1": h1_bias,
            "h4": h4_bias,
            "daily": daily_bias,
            "confidence_modifier": 0.10 if (alignment != "mixed" and daily_agrees) else
                                   0.05 if alignment != "mixed" else
                                   -0.05,
        }

    def _get_bias(self, df, timeframe: str) -> dict:
        if df is None or len(df) < 50:
            return {"bias": "neutral", "timeframe": timeframe}

        close = df["close"]
        ema_20 = ta.trend.ema_indicator(close, window=20)
        ema_50 = ta.trend.ema_indicator(close, window=50)
        rsi = ta.momentum.rsi(close, window=14)

        price = close.iloc[-1]
        above_20 = price > ema_20.iloc[-1]
        above_50 = price > ema_50.iloc[-1]
        ema_20_above_50 = ema_20.iloc[-1] > ema_50.iloc[-1]

        if above_20 and above_50 and ema_20_above_50:
            bias = "bullish"
        elif not above_20 and not above_50 and not ema_20_above_50:
            bias = "bearish"
        else:
            bias = "neutral"

        return {
            "bias": bias,
            "timeframe": timeframe,
            "rsi": round(rsi.iloc[-1], 1),
            "price_vs_ema20": "above" if above_20 else "below",
            "price_vs_ema50": "above" if above_50 else "below",
        }
