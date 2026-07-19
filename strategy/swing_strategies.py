"""
Swing Trading Strategies — dedicated institutional swing system.

Completely separate from the H1 scalp strategies.
Different indicators, different logic, different timeframe behavior.

Architecture:
    GATES (must ALL pass):
        1. MA Trend Filter — 50/200 MA determines allowed direction
        2. Market Structure — HH/HL or LL/LH must be intact

    ENTRY SIGNALS (need 2+ to agree):
        3. Fair Value Gap — pullback into imbalance zone
        4. Divergence — RSI/MACD divergence confirms pullback reversal
        5. Trend Pullback — retracement to key moving averages
        6. Liquidity Sweep — stop hunt then reversal into trend
"""
import logging
import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class MATrendGate:
    """50/200 MA trend direction — hard gate. Only trade WITH the trend."""
    name = "ma_trend_gate"
    is_gate = True

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 210:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 210 bars for MA"}

        close = df["close"]
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        if pd.isna(ma50.iloc[-1]) or pd.isna(ma200.iloc[-1]):
            return {"signal": "SKIP", "confidence": 0, "reason": "MAs not computed"}

        price = float(close.iloc[-1])
        ma50_val = float(ma50.iloc[-1])
        ma200_val = float(ma200.iloc[-1])

        if ma50_val > ma200_val:
            if price > ma50_val:
                strength = 0.85
                reasons = ["Golden cross + price above 50MA — strong uptrend"]
            elif price > ma200_val:
                strength = 0.60
                reasons = ["Golden cross, price between MAs — weak uptrend"]
            else:
                return {"signal": "SKIP", "confidence": 0,
                        "reason": "Price below 200MA despite golden cross"}
            trend = "BULLISH"
        elif ma50_val < ma200_val:
            if price < ma50_val:
                strength = 0.85
                reasons = ["Death cross + price below 50MA — strong downtrend"]
            elif price < ma200_val:
                strength = 0.60
                reasons = ["Death cross, price between MAs — weak downtrend"]
            else:
                return {"signal": "SKIP", "confidence": 0,
                        "reason": "Price above 200MA despite death cross"}
            trend = "BEARISH"
        else:
            return {"signal": "SKIP", "confidence": 0, "reason": "MAs flat — no trend"}

        if len(ma50) > 5:
            slope = (float(ma50.iloc[-1]) - float(ma50.iloc[-5])) / float(ma50.iloc[-5]) * 100
            if (trend == "BULLISH" and slope > 0.1) or (trend == "BEARISH" and slope < -0.1):
                strength = min(strength + 0.05, 0.95)
                reasons.append("50MA slope confirms trend acceleration")

        signal = "BUY" if trend == "BULLISH" else "SELL"
        return {
            "signal": signal,
            "confidence": round(strength, 4),
            "trend": trend,
            "reasons": reasons,
            "strategy": self.name,
            "ma50": ma50_val,
            "ma200": ma200_val,
        }


class StructureGate:
    """Market structure — HH/HL (bullish) or LL/LH (bearish) must be intact."""
    name = "structure_gate"
    is_gate = True

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 60:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 60 bars"}

        highs = df["high"].values
        lows = df["low"].values
        n = 5

        swing_highs = []
        swing_lows = []

        for i in range(n, len(highs) - n):
            if all(highs[i] >= highs[i - j] for j in range(1, n + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, n + 1)):
                swing_highs.append({"idx": i, "price": float(highs[i])})

            if all(lows[i] <= lows[i - j] for j in range(1, n + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, n + 1)):
                swing_lows.append({"idx": i, "price": float(lows[i])})

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"signal": "SKIP", "confidence": 0, "reason": "Not enough swing points"}

        sh1, sh2 = swing_highs[-2], swing_highs[-1]
        sl1, sl2 = swing_lows[-2], swing_lows[-1]

        hh = sh2["price"] > sh1["price"]
        hl = sl2["price"] > sl1["price"]
        ll = sl2["price"] < sl1["price"]
        lh = sh2["price"] < sh1["price"]

        if hh and hl:
            signal = "BUY"
            strength = 0.80
            reasons = ["HH + HL — bullish structure intact"]
        elif ll and lh:
            signal = "SELL"
            strength = 0.80
            reasons = ["LL + LH — bearish structure intact"]
        elif hh and not hl:
            signal = "BUY"
            strength = 0.55
            reasons = ["HH but no HL — developing bullish structure"]
        elif ll and not lh:
            signal = "SELL"
            strength = 0.55
            reasons = ["LL but no LH — developing bearish structure"]
        else:
            return {"signal": "SKIP", "confidence": 0, "reason": "No clear structure trend"}

        return {
            "signal": signal,
            "confidence": round(strength, 4),
            "reasons": reasons,
            "strategy": self.name,
            "last_swing_low": sl2["price"],
            "last_swing_high": sh2["price"],
            "prev_swing_low": sl1["price"],
            "prev_swing_high": sh1["price"],
            "swing_highs": swing_highs[-4:],
            "swing_lows": swing_lows[-4:],
        }


class FairValueGapStrategy:
    """Entry at Fair Value Gaps — institutional imbalance zones in trend direction."""
    name = "fair_value_gap"
    is_gate = False

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        price = float(close.iloc[-1])

        atr = ta.volatility.average_true_range(high, low, close, window=14)
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0

        score = 0
        reasons = []

        for i in range(2, min(30, len(df) - 2)):
            idx1 = len(df) - i - 2
            idx3 = len(df) - i
            if idx1 < 0:
                break

            bar1_high = float(high.iloc[idx1])
            bar3_low = float(low.iloc[idx3])
            bar1_low = float(low.iloc[idx1])
            bar3_high = float(high.iloc[idx3])

            # Bullish FVG: gap up
            if bar1_high < bar3_low:
                gap_size = bar3_low - bar1_high
                if atr_val > 0 and gap_size > atr_val * 0.5:
                    if bar1_high <= price <= bar3_low * 1.002:
                        score += 0.55
                        reasons.append(f"Price in bullish FVG ({i} bars back, {gap_size / atr_val:.1f} ATR)")
                        if i > 5:
                            score += 0.05
                            reasons.append("Fresh untested gap")
                        break

            # Bearish FVG: gap down
            if bar1_low > bar3_high:
                gap_size = bar1_low - bar3_high
                if atr_val > 0 and gap_size > atr_val * 0.5:
                    if bar3_high * 0.998 <= price <= bar1_low:
                        score -= 0.55
                        reasons.append(f"Price in bearish FVG ({i} bars back, {gap_size / atr_val:.1f} ATR)")
                        if i > 5:
                            score -= 0.05
                            reasons.append("Fresh untested gap")
                        break

        if abs(score) < 0.30:
            return {"signal": "SKIP", "confidence": 0, "reason": "No FVG entry"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class DivergenceStrategy:
    """RSI + MACD divergence — momentum disagreeing with price at pullback."""
    name = "divergence"
    is_gate = False

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 60:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 60 bars"}

        close = df["close"]
        rsi = ta.momentum.rsi(close, window=14)
        macd_hist = ta.trend.macd_diff(close)

        if pd.isna(rsi.iloc[-1]) or pd.isna(macd_hist.iloc[-1]):
            return {"signal": "SKIP", "confidence": 0, "reason": "Indicators not ready"}

        score = 0
        reasons = []
        lookback = 40
        half = lookback // 2

        price_window = close.iloc[-lookback:]
        rsi_window = rsi.iloc[-lookback:]
        macd_window = macd_hist.iloc[-lookback:]

        p_low1 = float(price_window.iloc[:half].min())
        p_low2 = float(price_window.iloc[half:].min())
        r_low1 = float(rsi_window.iloc[:half].min())
        r_low2 = float(rsi_window.iloc[half:].min())

        p_high1 = float(price_window.iloc[:half].max())
        p_high2 = float(price_window.iloc[half:].max())
        r_high1 = float(rsi_window.iloc[:half].max())
        r_high2 = float(rsi_window.iloc[half:].max())

        # Bullish: price lower low, RSI higher low
        if p_low2 < p_low1 and r_low2 > r_low1:
            score += 0.45
            reasons.append("Bullish RSI divergence — momentum strengthening at lower price")
            if float(rsi.iloc[-1]) < 40:
                score += 0.10
                reasons.append("RSI in pullback zone")

        # Bearish: price higher high, RSI lower high
        elif p_high2 > p_high1 and r_high2 < r_high1:
            score -= 0.45
            reasons.append("Bearish RSI divergence — momentum weakening at higher price")
            if float(rsi.iloc[-1]) > 60:
                score -= 0.10
                reasons.append("RSI in overextended zone")

        # MACD confirmation
        m_low1 = float(macd_window.iloc[:half].min())
        m_low2 = float(macd_window.iloc[half:].min())
        m_high1 = float(macd_window.iloc[:half].max())
        m_high2 = float(macd_window.iloc[half:].max())

        if score > 0 and p_low2 < p_low1 and m_low2 > m_low1:
            score += 0.15
            reasons.append("MACD histogram confirms bullish divergence")
        elif score < 0 and p_high2 > p_high1 and m_high2 < m_high1:
            score -= 0.15
            reasons.append("MACD histogram confirms bearish divergence")

        if abs(score) < 0.30:
            return {"signal": "SKIP", "confidence": 0, "reason": "No divergence"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class TrendPullbackStrategy:
    """Pullback to 21 EMA or 50 MA in trend direction — the bread and butter entry."""
    name = "trend_pullback"
    is_gate = False

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 55:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 55 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ema21 = ta.trend.ema_indicator(close, window=21)
        ma50 = close.rolling(50).mean()
        atr = ta.volatility.average_true_range(high, low, close, window=14)
        rsi = ta.momentum.rsi(close, window=14)

        if pd.isna(ema21.iloc[-1]) or pd.isna(ma50.iloc[-1]) or pd.isna(atr.iloc[-1]):
            return {"signal": "SKIP", "confidence": 0, "reason": "Indicators not ready"}

        price = float(close.iloc[-1])
        ema21_val = float(ema21.iloc[-1])
        ma50_val = float(ma50.iloc[-1])
        atr_val = float(atr.iloc[-1])

        if atr_val == 0:
            return {"signal": "SKIP", "confidence": 0, "reason": "Zero ATR"}

        score = 0
        reasons = []

        # Bullish pullback
        if price > ma50_val:
            dist_21 = abs(price - ema21_val) / atr_val
            dist_50 = abs(price - ma50_val) / atr_val

            if dist_21 < 0.5 and price >= ema21_val * 0.998:
                score += 0.45
                reasons.append(f"Pullback to 21 EMA ({dist_21:.2f} ATR away)")
            elif dist_50 < 0.5 and price >= ma50_val * 0.998:
                score += 0.50
                reasons.append(f"Deep pullback to 50 MA ({dist_50:.2f} ATR away)")

            if score > 0:
                if price > float(close.iloc[-2]):
                    score += 0.10
                    reasons.append("Bullish candle — bounce starting")
                if not pd.isna(rsi.iloc[-1]) and float(rsi.iloc[-1]) < 45 and float(rsi.iloc[-1]) > float(rsi.iloc[-2]):
                    score += 0.10
                    reasons.append("RSI turning up from pullback")
                recent_high = float(high.iloc[-10:].max())
                if (recent_high - price) / atr_val > 1.5:
                    score += 0.05
                    reasons.append("Meaningful pullback depth")

        # Bearish pullback
        elif price < ma50_val:
            dist_21 = abs(price - ema21_val) / atr_val
            dist_50 = abs(price - ma50_val) / atr_val

            if dist_21 < 0.5 and price <= ema21_val * 1.002:
                score -= 0.45
                reasons.append(f"Pullback to 21 EMA ({dist_21:.2f} ATR away)")
            elif dist_50 < 0.5 and price <= ma50_val * 1.002:
                score -= 0.50
                reasons.append(f"Deep pullback to 50 MA ({dist_50:.2f} ATR away)")

            if score < 0:
                if price < float(close.iloc[-2]):
                    score -= 0.10
                    reasons.append("Bearish candle — rejection starting")
                if not pd.isna(rsi.iloc[-1]) and float(rsi.iloc[-1]) > 55 and float(rsi.iloc[-1]) < float(rsi.iloc[-2]):
                    score -= 0.10
                    reasons.append("RSI turning down from pullback")
                recent_low = float(low.iloc[-10:].min())
                if (price - recent_low) / atr_val > 1.5:
                    score -= 0.05
                    reasons.append("Meaningful pullback depth")

        if abs(score) < 0.30:
            return {"signal": "SKIP", "confidence": 0, "reason": "No pullback entry"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class LiquiditySweepStrategy:
    """Stop hunt below/above swing points then reversal — institutional trap."""
    name = "liquidity_sweep"
    is_gate = False

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 30:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 30 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        price = float(close.iloc[-1])
        curr_low = float(low.iloc[-1])
        curr_high = float(high.iloc[-1])
        curr_open = float(open_.iloc[-1])

        atr = ta.volatility.average_true_range(high, low, close, window=14)
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0

        score = 0
        reasons = []

        recent_low = float(low.iloc[-20:-1].min())
        recent_high = float(high.iloc[-20:-1].max())

        # Bullish sweep: wick below recent low, close back above
        if curr_low < recent_low and price > recent_low:
            score += 0.50
            reasons.append("Swept below recent low — liquidity grab, closed above")

            candle_range = curr_high - curr_low
            if candle_range > 0:
                lower_wick = min(curr_open, price) - curr_low
                if lower_wick / candle_range > 0.5:
                    score += 0.15
                    reasons.append("Strong rejection wick — institutions buying")

            if "volume" in df.columns:
                vol = df["volume"]
                avg_vol = float(vol.rolling(20).mean().iloc[-1])
                if avg_vol > 0 and float(vol.iloc[-1]) > avg_vol * 1.3:
                    score += 0.10
                    reasons.append("High volume on sweep")

        # Bearish sweep: wick above recent high, close back below
        elif curr_high > recent_high and price < recent_high:
            score -= 0.50
            reasons.append("Swept above recent high — liquidity grab, closed below")

            candle_range = curr_high - curr_low
            if candle_range > 0:
                upper_wick = curr_high - max(curr_open, price)
                if upper_wick / candle_range > 0.5:
                    score -= 0.15
                    reasons.append("Strong rejection wick — institutions selling")

            if "volume" in df.columns:
                vol = df["volume"]
                avg_vol = float(vol.rolling(20).mean().iloc[-1])
                if avg_vol > 0 and float(vol.iloc[-1]) > avg_vol * 1.3:
                    score -= 0.10
                    reasons.append("High volume on sweep")

        if abs(score) < 0.30:
            return {"signal": "SKIP", "confidence": 0, "reason": "No liquidity sweep"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


# Strategy instances
SWING_GATES = [MATrendGate(), StructureGate()]
SWING_ENTRIES = [
    FairValueGapStrategy(),
    DivergenceStrategy(),
    TrendPullbackStrategy(),
    LiquiditySweepStrategy(),
]
ALL_SWING_STRATEGIES = SWING_GATES + SWING_ENTRIES
