"""
Multi-Strategy Engine — each strategy votes independently.
Strategies: Trend Following, Mean Reversion, Breakout, Momentum Scalp.
"""
import logging
import numpy as np
import ta

logger = logging.getLogger(__name__)


class TrendStrategy:
    name = "trend_following"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 200:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 200 bars"}

        close = df["close"]
        ema_9 = ta.trend.ema_indicator(close, window=9)
        ema_21 = ta.trend.ema_indicator(close, window=21)
        ema_50 = ta.trend.ema_indicator(close, window=50)
        ema_200 = ta.trend.ema_indicator(close, window=200)
        adx = ta.trend.adx(df["high"], df["low"], close, window=14)
        macd_hist = ta.trend.macd_diff(close)

        if regime not in ("trending", "transitioning"):
            return {"signal": "SKIP", "confidence": 0, "reason": "Not a trending regime"}

        score = 0
        reasons = []

        # EMA alignment
        if ema_9.iloc[-1] > ema_21.iloc[-1] > ema_50.iloc[-1] > ema_200.iloc[-1]:
            score += 0.25
            reasons.append("Full bullish EMA stack")
        elif ema_9.iloc[-1] < ema_21.iloc[-1] < ema_50.iloc[-1] < ema_200.iloc[-1]:
            score -= 0.25
            reasons.append("Full bearish EMA stack")

        # EMA crossover
        if ema_9.iloc[-1] > ema_21.iloc[-1] and ema_9.iloc[-2] <= ema_21.iloc[-2]:
            score += 0.20
            reasons.append("Bullish EMA cross")
        elif ema_9.iloc[-1] < ema_21.iloc[-1] and ema_9.iloc[-2] >= ema_21.iloc[-2]:
            score -= 0.20
            reasons.append("Bearish EMA cross")

        # ADX strength
        if adx.iloc[-1] > 25:
            score *= 1.3
            reasons.append(f"Strong trend ADX={adx.iloc[-1]:.0f}")

        # MACD confirmation
        if macd_hist.iloc[-1] > 0 and score > 0:
            score += 0.10
        elif macd_hist.iloc[-1] < 0 and score < 0:
            score -= 0.10

        # Pullback to EMA (entry refinement)
        price = close.iloc[-1]
        if score > 0 and abs(price - ema_21.iloc[-1]) / price < 0.003:
            score += 0.15
            reasons.append("Price at EMA21 pullback")
        elif score < 0 and abs(price - ema_21.iloc[-1]) / price < 0.003:
            score -= 0.15
            reasons.append("Price at EMA21 pullback")

        if abs(score) < 0.20:
            return {"signal": "SKIP", "confidence": 0, "reason": "Weak trend signal"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class MeanReversionStrategy:
    name = "mean_reversion"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        rsi = ta.momentum.rsi(close, window=14)
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        stoch = ta.momentum.stoch(df["high"], df["low"], close, window=14, smooth_window=3)

        if regime not in ("ranging", "transitioning"):
            return {"signal": "SKIP", "confidence": 0, "reason": "Not a ranging regime"}

        score = 0
        reasons = []

        # RSI extremes
        if rsi.iloc[-1] < 25:
            score += 0.30
            reasons.append(f"RSI oversold ({rsi.iloc[-1]:.0f})")
        elif rsi.iloc[-1] < 35:
            score += 0.15
            reasons.append(f"RSI low ({rsi.iloc[-1]:.0f})")
        elif rsi.iloc[-1] > 75:
            score -= 0.30
            reasons.append(f"RSI overbought ({rsi.iloc[-1]:.0f})")
        elif rsi.iloc[-1] > 65:
            score -= 0.15
            reasons.append(f"RSI high ({rsi.iloc[-1]:.0f})")

        # Bollinger Band touch
        price = close.iloc[-1]
        if price <= bb.bollinger_lband().iloc[-1]:
            score += 0.25
            reasons.append("At lower Bollinger Band")
        elif price >= bb.bollinger_hband().iloc[-1]:
            score -= 0.25
            reasons.append("At upper Bollinger Band")

        # Stochastic
        if stoch.iloc[-1] < 20:
            score += 0.15
            reasons.append("Stochastic oversold")
        elif stoch.iloc[-1] > 80:
            score -= 0.15
            reasons.append("Stochastic overbought")

        if abs(score) < 0.30:
            return {"signal": "SKIP", "confidence": 0, "reason": "Weak mean reversion signal"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class BreakoutStrategy:
    name = "breakout"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr = ta.volatility.average_true_range(high, low, close, window=14)
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_width = bb.bollinger_wband()

        score = 0
        reasons = []

        # Bollinger squeeze detection (compression before breakout)
        avg_width = bb_width.rolling(50).mean().iloc[-1]
        current_width = bb_width.iloc[-1]
        is_squeezed = current_width < avg_width * 0.6

        # Price range compression
        lookback_20_high = high.iloc[-20:].max()
        lookback_20_low = low.iloc[-20:].min()
        range_pct = (lookback_20_high - lookback_20_low) / lookback_20_low

        price = close.iloc[-1]

        # Breakout above recent range
        if price > lookback_20_high and is_squeezed:
            score += 0.40
            reasons.append("Breakout above 20-bar high after squeeze")
        elif price > lookback_20_high:
            score += 0.25
            reasons.append("Breakout above 20-bar high")

        # Breakdown below recent range
        if price < lookback_20_low and is_squeezed:
            score -= 0.40
            reasons.append("Breakdown below 20-bar low after squeeze")
        elif price < lookback_20_low:
            score -= 0.25
            reasons.append("Breakdown below 20-bar low")

        # Volume confirmation (if available)
        if "volume" in df.columns:
            vol = df["volume"]
            avg_vol = vol.rolling(20).mean().iloc[-1]
            if vol.iloc[-1] > avg_vol * 1.5:
                score *= 1.3
                reasons.append("Volume expansion confirms breakout")

        # ATR expansion
        avg_atr = atr.rolling(20).mean().iloc[-1]
        if atr.iloc[-1] > avg_atr * 1.3:
            score *= 1.2
            reasons.append("ATR expanding")

        if abs(score) < 0.25:
            return {"signal": "SKIP", "confidence": 0, "reason": "No breakout detected"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class MomentumStrategy:
    name = "momentum"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        rsi = ta.momentum.rsi(close, window=14)
        macd_hist = ta.trend.macd_diff(close)
        adx = ta.trend.adx(df["high"], df["low"], close, window=14)
        roc = ta.momentum.roc(close, window=10)

        score = 0
        reasons = []

        # RSI momentum (not extreme, but moving)
        if 50 < rsi.iloc[-1] < 70 and rsi.iloc[-1] > rsi.iloc[-2]:
            score += 0.20
            reasons.append("Bullish RSI momentum")
        elif 30 < rsi.iloc[-1] < 50 and rsi.iloc[-1] < rsi.iloc[-2]:
            score -= 0.20
            reasons.append("Bearish RSI momentum")

        # MACD momentum
        if macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3]:
            score += 0.25
            reasons.append("MACD histogram accelerating up")
        elif macd_hist.iloc[-1] < macd_hist.iloc[-2] < macd_hist.iloc[-3]:
            score -= 0.25
            reasons.append("MACD histogram accelerating down")

        # Rate of change
        if roc.iloc[-1] > 0.3:
            score += 0.20
            reasons.append(f"Strong positive ROC ({roc.iloc[-1]:.2f}%)")
        elif roc.iloc[-1] < -0.3:
            score -= 0.20
            reasons.append(f"Strong negative ROC ({roc.iloc[-1]:.2f}%)")

        # ADX confirms directional movement
        if adx.iloc[-1] > 20:
            score *= 1.2
            reasons.append(f"Directional strength ADX={adx.iloc[-1]:.0f}")

        if abs(score) < 0.25:
            return {"signal": "SKIP", "confidence": 0, "reason": "Weak momentum"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


ALL_STRATEGIES = [
    TrendStrategy(),
    MeanReversionStrategy(),
    BreakoutStrategy(),
    MomentumStrategy(),
]
