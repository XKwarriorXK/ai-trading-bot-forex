"""
Technical Analysis Agent — computes indicators on forex OHLCV data.
"""
import logging
import numpy as np
import ta

logger = logging.getLogger(__name__)


class TechnicalAgent:
    def __init__(self):
        pass

    def analyze(self, df, instrument: str = "EUR_USD") -> dict:
        if df is None or len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Insufficient data"}

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"] if "volume" in df.columns else None

            ema_9 = ta.trend.ema_indicator(close, window=9)
            ema_21 = ta.trend.ema_indicator(close, window=21)
            ema_50 = ta.trend.ema_indicator(close, window=50)
            ema_200 = ta.trend.ema_indicator(close, window=200) if len(df) >= 200 else None

            rsi = ta.momentum.rsi(close, window=14)
            macd = ta.trend.macd(close)
            macd_signal = ta.trend.macd_signal(close)
            macd_hist = ta.trend.macd_diff(close)
            adx = ta.trend.adx(high, low, close, window=14)
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_width = bb.bollinger_wband()
            stoch_k = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
            atr = ta.volatility.average_true_range(high, low, close, window=14)

            current = close.iloc[-1]
            prev = close.iloc[-2]

            regime = self._detect_regime(adx.iloc[-1], bb_width.iloc[-1], rsi.iloc[-1])

            signals = []
            reasons = []

            # EMA crossover
            if ema_9.iloc[-1] > ema_21.iloc[-1] and ema_9.iloc[-2] <= ema_21.iloc[-2]:
                signals.append(("BUY", 0.6))
                reasons.append("EMA 9/21 bullish cross")
            elif ema_9.iloc[-1] < ema_21.iloc[-1] and ema_9.iloc[-2] >= ema_21.iloc[-2]:
                signals.append(("SELL", 0.6))
                reasons.append("EMA 9/21 bearish cross")

            # EMA alignment (trend strength)
            if ema_9.iloc[-1] > ema_21.iloc[-1] > ema_50.iloc[-1]:
                signals.append(("BUY", 0.55))
                reasons.append("Bullish EMA alignment")
            elif ema_9.iloc[-1] < ema_21.iloc[-1] < ema_50.iloc[-1]:
                signals.append(("SELL", 0.55))
                reasons.append("Bearish EMA alignment")

            # RSI
            if rsi.iloc[-1] < 30:
                signals.append(("BUY", 0.65))
                reasons.append(f"RSI oversold ({rsi.iloc[-1]:.1f})")
            elif rsi.iloc[-1] > 70:
                signals.append(("SELL", 0.65))
                reasons.append(f"RSI overbought ({rsi.iloc[-1]:.1f})")

            # MACD
            if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
                signals.append(("BUY", 0.55))
                reasons.append("MACD histogram turned positive")
            elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
                signals.append(("SELL", 0.55))
                reasons.append("MACD histogram turned negative")

            # Bollinger Bands
            if current <= bb_lower.iloc[-1]:
                signals.append(("BUY", 0.60))
                reasons.append("Price at lower Bollinger Band")
            elif current >= bb_upper.iloc[-1]:
                signals.append(("SELL", 0.60))
                reasons.append("Price at upper Bollinger Band")

            # Stochastic
            if stoch_k.iloc[-1] < 20:
                signals.append(("BUY", 0.50))
                reasons.append("Stochastic oversold")
            elif stoch_k.iloc[-1] > 80:
                signals.append(("SELL", 0.50))
                reasons.append("Stochastic overbought")

            buy_signals = [(s, c) for s, c in signals if s == "BUY"]
            sell_signals = [(s, c) for s, c in signals if s == "SELL"]

            if len(buy_signals) >= 2:
                avg_conf = np.mean([c for _, c in buy_signals])
                final_signal = "BUY"
                final_conf = min(avg_conf + 0.05 * (len(buy_signals) - 2), 0.95)
            elif len(sell_signals) >= 2:
                avg_conf = np.mean([c for _, c in sell_signals])
                final_signal = "SELL"
                final_conf = min(avg_conf + 0.05 * (len(sell_signals) - 2), 0.95)
            else:
                final_signal = "SKIP"
                final_conf = 0

            return {
                "signal": final_signal,
                "confidence": round(final_conf, 4),
                "regime": regime,
                "reasons": reasons,
                "indicators": {
                    "price": round(current, 5),
                    "rsi": round(rsi.iloc[-1], 2),
                    "macd_histogram": round(macd_hist.iloc[-1], 6),
                    "adx": round(adx.iloc[-1], 2),
                    "ema_9": round(ema_9.iloc[-1], 5),
                    "ema_21": round(ema_21.iloc[-1], 5),
                    "ema_50": round(ema_50.iloc[-1], 5),
                    "bb_upper": round(bb_upper.iloc[-1], 5),
                    "bb_lower": round(bb_lower.iloc[-1], 5),
                    "bb_width": round(bb_width.iloc[-1], 6),
                    "stoch_k": round(stoch_k.iloc[-1], 2),
                    "atr": round(atr.iloc[-1], 5),
                },
            }
        except Exception as e:
            logger.error(f"Technical analysis failed for {instrument}: {e}")
            return {"signal": "SKIP", "confidence": 0, "reason": str(e)}

    def _detect_regime(self, adx_val, bb_width, rsi_val) -> str:
        if adx_val > 25:
            return "trending"
        elif bb_width < 0.01:
            return "ranging"
        elif bb_width > 0.03:
            return "volatile"
        else:
            return "transitioning"
