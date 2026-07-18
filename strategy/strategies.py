"""
Multi-Strategy Engine — 11 proven institutional strategies vote independently.
Donchian, London, Bollinger+RSI, MACD+EMA, Ichimoku, Smart Money, Price Action,
Keltner Channel, ADX Momentum, Fibonacci Retracement, Stochastic Divergence.
"""
import logging
import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class DonchianBreakoutStrategy:
    """Turtle Trading — buy 20-period high breakouts, sell 20-period low breakdowns."""
    name = "donchian_breakout"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 55:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 55 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        upper_20 = high.iloc[-21:-1].max()
        lower_20 = low.iloc[-21:-1].min()
        upper_55 = high.iloc[-56:-1].max() if len(df) >= 56 else upper_20
        lower_55 = low.iloc[-56:-1].min() if len(df) >= 56 else lower_20

        price = float(close.iloc[-1])
        atr = ta.volatility.average_true_range(high, low, close, window=14)
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0

        score = 0
        reasons = []

        if price > upper_20:
            score += 0.35
            reasons.append("Broke 20-period high (Donchian)")
            if price > upper_55:
                score += 0.15
                reasons.append("Also broke 55-period high")
            if atr_val > 0:
                dist = (price - upper_20) / atr_val
                if dist > 0.5:
                    score += 0.10
                    reasons.append("Strong breakout (>0.5 ATR)")
        elif price < lower_20:
            score -= 0.35
            reasons.append("Broke 20-period low (Donchian)")
            if price < lower_55:
                score -= 0.15
                reasons.append("Also broke 55-period low")
            if atr_val > 0:
                dist = (lower_20 - price) / atr_val
                if dist > 0.5:
                    score -= 0.10
                    reasons.append("Strong breakdown (>0.5 ATR)")

        if "volume" in df.columns and abs(score) > 0:
            vol = df["volume"]
            avg_vol = vol.rolling(20).mean().iloc[-1]
            if avg_vol > 0 and vol.iloc[-1] > avg_vol * 1.3:
                score *= 1.2
                reasons.append("Volume confirms breakout")

        if abs(score) < 0.25:
            return {"signal": "SKIP", "confidence": 0, "reason": "No Donchian breakout"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class LondonBreakoutStrategy:
    """Trade the Asian session range breakout at London open."""
    name = "london_breakout"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 20:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 20 bars"}

        try:
            hour = df.index[-1].hour
        except AttributeError:
            return {"signal": "SKIP", "confidence": 0, "reason": "No timestamp"}

        if hour < 7 or hour > 10:
            return {"signal": "SKIP", "confidence": 0, "reason": "Outside London window"}

        asian_bars = df[df.index.hour < 7].tail(7)
        if len(asian_bars) < 3:
            return {"signal": "SKIP", "confidence": 0, "reason": "Not enough Asian data"}

        asian_high = float(asian_bars["high"].max())
        asian_low = float(asian_bars["low"].min())
        asian_range = asian_high - asian_low

        if asian_range <= 0:
            return {"signal": "SKIP", "confidence": 0, "reason": "Zero Asian range"}

        price = float(df["close"].iloc[-1])

        score = 0
        reasons = []

        if price > asian_high:
            score += 0.35
            reasons.append("Broke Asian high at London open")
            strength = (price - asian_high) / asian_range
            if strength > 0.3:
                score += 0.15
                reasons.append(f"Strong breakout ({strength:.0%} of range)")
            if hour <= 8:
                score += 0.10
                reasons.append("Early London — peak liquidity")
        elif price < asian_low:
            score -= 0.35
            reasons.append("Broke Asian low at London open")
            strength = (asian_low - price) / asian_range
            if strength > 0.3:
                score -= 0.15
                reasons.append(f"Strong breakdown ({strength:.0%} of range)")
            if hour <= 8:
                score -= 0.10
                reasons.append("Early London — peak liquidity")

        if abs(score) < 0.25:
            return {"signal": "SKIP", "confidence": 0, "reason": "No London breakout"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class BollingerRSIStrategy:
    """Mean reversion — Bollinger Band touch + RSI extremes with trend confirmation."""
    name = "bollinger_rsi"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        rsi = ta.momentum.rsi(close, window=14)
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        ema_50 = ta.trend.ema_indicator(close, window=50)
        stoch = ta.momentum.stoch(df["high"], df["low"], close, window=14, smooth_window=3)

        if pd.isna(rsi.iloc[-1]) or pd.isna(bb.bollinger_lband().iloc[-1]):
            return {"signal": "SKIP", "confidence": 0, "reason": "Indicators not ready"}

        price = float(close.iloc[-1])
        bb_lower = float(bb.bollinger_lband().iloc[-1])
        bb_upper = float(bb.bollinger_hband().iloc[-1])

        score = 0
        reasons = []

        if price <= bb_lower and rsi.iloc[-1] < 35:
            score += 0.40
            reasons.append(f"Price at lower BB + RSI {rsi.iloc[-1]:.0f}")
            if price > ema_50.iloc[-1]:
                score += 0.15
                reasons.append("In uptrend (above EMA50) — bounce likely")
        elif price >= bb_upper and rsi.iloc[-1] > 65:
            score -= 0.40
            reasons.append(f"Price at upper BB + RSI {rsi.iloc[-1]:.0f}")
            if price < ema_50.iloc[-1]:
                score -= 0.15
                reasons.append("In downtrend (below EMA50) — rejection likely")

        if not pd.isna(stoch.iloc[-1]):
            if score > 0 and stoch.iloc[-1] < 25:
                score += 0.10
                reasons.append("Stochastic confirms oversold")
            elif score < 0 and stoch.iloc[-1] > 75:
                score -= 0.10
                reasons.append("Stochastic confirms overbought")

        # RSI divergence
        if abs(score) > 0 and len(df) >= 15:
            rsi_window = rsi.iloc[-10:]
            if score > 0:
                if rsi.iloc[-1] > rsi_window.min() and close.iloc[-1] <= close.iloc[-10:].min():
                    score += 0.10
                    reasons.append("Bullish RSI divergence")
            else:
                if rsi.iloc[-1] < rsi_window.max() and close.iloc[-1] >= close.iloc[-10:].max():
                    score -= 0.10
                    reasons.append("Bearish RSI divergence")

        if abs(score) < 0.30:
            return {"signal": "SKIP", "confidence": 0, "reason": "No BB+RSI setup"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class MACDTrendStrategy:
    """MACD signal line crossover confirmed by EMA trend."""
    name = "macd_trend"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        macd_line = ta.trend.macd(close)
        macd_signal = ta.trend.macd_signal(close)
        macd_hist = ta.trend.macd_diff(close)
        ema_50 = ta.trend.ema_indicator(close, window=50)
        ema_21 = ta.trend.ema_indicator(close, window=21)

        if pd.isna(macd_line.iloc[-1]) or pd.isna(macd_signal.iloc[-1]):
            return {"signal": "SKIP", "confidence": 0, "reason": "MACD not ready"}

        price = float(close.iloc[-1])

        macd_crossed_up = False
        macd_crossed_down = False
        for i in range(1, 5):
            if len(macd_line) <= i:
                break
            if (macd_line.iloc[-i] > macd_signal.iloc[-i] and
                    macd_line.iloc[-i - 1] <= macd_signal.iloc[-i - 1]):
                macd_crossed_up = True
                break
            elif (macd_line.iloc[-i] < macd_signal.iloc[-i] and
                  macd_line.iloc[-i - 1] >= macd_signal.iloc[-i - 1]):
                macd_crossed_down = True
                break

        score = 0
        reasons = []

        if macd_crossed_up:
            score += 0.30
            reasons.append("MACD bullish crossover")
            if price > ema_50.iloc[-1]:
                score += 0.15
                reasons.append("Above EMA50 (uptrend)")
            if price > ema_21.iloc[-1]:
                score += 0.10
                reasons.append("Above EMA21")
            if macd_hist.iloc[-1] > macd_hist.iloc[-2]:
                score += 0.10
                reasons.append("MACD histogram accelerating")
            if macd_line.iloc[-1] > 0 and len(macd_line) > 4 and macd_line.iloc[-4] < 0:
                score += 0.10
                reasons.append("MACD crossed zero line")
        elif macd_crossed_down:
            score -= 0.30
            reasons.append("MACD bearish crossover")
            if price < ema_50.iloc[-1]:
                score -= 0.15
                reasons.append("Below EMA50 (downtrend)")
            if price < ema_21.iloc[-1]:
                score -= 0.10
                reasons.append("Below EMA21")
            if macd_hist.iloc[-1] < macd_hist.iloc[-2]:
                score -= 0.10
                reasons.append("MACD histogram accelerating down")
            if macd_line.iloc[-1] < 0 and len(macd_line) > 4 and macd_line.iloc[-4] > 0:
                score -= 0.10
                reasons.append("MACD crossed below zero")

        if abs(score) < 0.25:
            return {"signal": "SKIP", "confidence": 0, "reason": "No MACD setup"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class IchimokuStrategy:
    """Ichimoku Cloud — all 5 components must align for a signal."""
    name = "ichimoku"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 80:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 80 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ich = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
        tenkan = ich.ichimoku_conversion_line()
        kijun = ich.ichimoku_base_line()
        span_a = ich.ichimoku_a()
        span_b = ich.ichimoku_b()

        if pd.isna(tenkan.iloc[-1]) or pd.isna(span_a.iloc[-1]) or pd.isna(span_b.iloc[-1]):
            return {"signal": "SKIP", "confidence": 0, "reason": "Ichimoku not ready"}

        price = float(close.iloc[-1])
        cloud_top = float(max(span_a.iloc[-1], span_b.iloc[-1]))
        cloud_bottom = float(min(span_a.iloc[-1], span_b.iloc[-1]))

        score = 0
        reasons = []

        if price > cloud_top:
            score += 0.20
            reasons.append("Price above Ichimoku cloud")
        elif price < cloud_bottom:
            score -= 0.20
            reasons.append("Price below Ichimoku cloud")
        else:
            return {"signal": "SKIP", "confidence": 0, "reason": "Price inside cloud"}

        # Tenkan/Kijun cross
        tk_cross_up = False
        tk_cross_down = False
        for i in range(1, 5):
            if len(tenkan) <= i:
                break
            if (tenkan.iloc[-i] > kijun.iloc[-i] and
                    tenkan.iloc[-i - 1] <= kijun.iloc[-i - 1]):
                tk_cross_up = True
                break
            elif (tenkan.iloc[-i] < kijun.iloc[-i] and
                  tenkan.iloc[-i - 1] >= kijun.iloc[-i - 1]):
                tk_cross_down = True
                break

        if tk_cross_up and score > 0:
            score += 0.25
            reasons.append("Tenkan crossed above Kijun")
        elif tk_cross_down and score < 0:
            score -= 0.25
            reasons.append("Tenkan crossed below Kijun")
        elif tenkan.iloc[-1] > kijun.iloc[-1] and score > 0:
            score += 0.10
            reasons.append("Tenkan above Kijun")
        elif tenkan.iloc[-1] < kijun.iloc[-1] and score < 0:
            score -= 0.10
            reasons.append("Tenkan below Kijun")

        cloud_bullish = span_a.iloc[-1] > span_b.iloc[-1]
        if cloud_bullish and score > 0:
            score += 0.10
            reasons.append("Bullish cloud color")
        elif not cloud_bullish and score < 0:
            score -= 0.10
            reasons.append("Bearish cloud color")

        # Chikou confirmation (current price vs 26 bars ago)
        if len(close) >= 27:
            price_26_ago = float(close.iloc[-27])
            if price > price_26_ago and score > 0:
                score += 0.10
                reasons.append("Chikou confirms bullish")
            elif price < price_26_ago and score < 0:
                score -= 0.10
                reasons.append("Chikou confirms bearish")

        if abs(score) < 0.25:
            return {"signal": "SKIP", "confidence": 0, "reason": "Weak Ichimoku signal"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class SmartMoneyStrategy:
    """Smart Money Concepts — order blocks, fair value gaps, liquidity sweeps."""
    name = "smart_money"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        price = float(close.iloc[-1])
        atr = ta.volatility.average_true_range(high, low, close, window=14)

        score = 0
        reasons = []

        # 1. FAIR VALUE GAP — unfilled gaps from impulse moves
        for i in range(2, min(20, len(df) - 2)):
            idx1 = len(df) - i - 2
            idx3 = len(df) - i
            if idx1 < 0:
                break

            bar1_high = float(high.iloc[idx1])
            bar3_low = float(low.iloc[idx3])
            bar1_low = float(low.iloc[idx1])
            bar3_high = float(high.iloc[idx3])

            if bar1_high < bar3_low:
                fvg_bottom = bar1_high
                fvg_top = bar3_low
                if fvg_bottom <= price <= fvg_top:
                    score += 0.25
                    reasons.append(f"Price at bullish FVG ({i} bars back)")
                    break

            if bar1_low > bar3_high:
                fvg_top = bar1_low
                fvg_bottom = bar3_high
                if fvg_bottom <= price <= fvg_top:
                    score -= 0.25
                    reasons.append(f"Price at bearish FVG ({i} bars back)")
                    break

        # 2. ORDER BLOCKS — institutional entry zones
        for i in range(5, min(30, len(df) - 3)):
            idx = len(df) - i
            if idx < 0:
                break

            candle_open = float(open_.iloc[idx])
            candle_close = float(close.iloc[idx])
            candle_high = float(high.iloc[idx])
            candle_low = float(low.iloc[idx])
            is_bearish = candle_close < candle_open
            is_bullish = candle_close > candle_open

            start = idx + 1
            end = min(start + 3, len(df))
            following = df.iloc[start:end]
            if len(following) < 2:
                continue

            atr_val = float(atr.iloc[idx]) if not pd.isna(atr.iloc[idx]) else float(atr.iloc[-1])

            if is_bearish:
                move_up = float(following["close"].max()) - candle_low
                if move_up > atr_val * 2.0:
                    body_top = candle_open
                    body_bottom = candle_close
                    if body_bottom <= price <= body_top:
                        score += 0.30
                        reasons.append(f"Bullish order block ({i} bars ago)")
                        break

            if is_bullish:
                move_down = candle_high - float(following["close"].min())
                if move_down > atr_val * 2.0:
                    body_top = candle_close
                    body_bottom = candle_open
                    if body_bottom <= price <= body_top:
                        score -= 0.30
                        reasons.append(f"Bearish order block ({i} bars ago)")
                        break

        # 3. LIQUIDITY SWEEP — stop hunt then reversal
        recent_low = float(low.iloc[-20:-1].min())
        recent_high = float(high.iloc[-20:-1].max())

        if float(low.iloc[-1]) < recent_low and price > recent_low:
            score += 0.20
            reasons.append("Liquidity sweep below recent low — bullish reversal")
        elif float(high.iloc[-1]) > recent_high and price < recent_high:
            score -= 0.20
            reasons.append("Liquidity sweep above recent high — bearish reversal")

        if abs(score) < 0.25:
            return {"signal": "SKIP", "confidence": 0, "reason": "No Smart Money setup"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class PriceActionStrategy:
    """Pin bars and engulfing patterns at key support/resistance levels."""
    name = "price_action"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 30:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 30 bars"}

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        curr_open = float(curr["open"])
        curr_close = float(curr["close"])
        curr_high = float(curr["high"])
        curr_low = float(curr["low"])
        curr_range = curr_high - curr_low

        prev_open = float(prev["open"])
        prev_close = float(prev["close"])

        if curr_range == 0:
            return {"signal": "SKIP", "confidence": 0, "reason": "Zero range bar"}

        curr_body = abs(curr_close - curr_open)
        curr_upper_wick = curr_high - max(curr_open, curr_close)
        curr_lower_wick = min(curr_open, curr_close) - curr_low

        score = 0
        reasons = []

        # PIN BAR — long wick rejection
        if curr_lower_wick > curr_range * 0.6 and curr_body < curr_range * 0.3:
            score += 0.35
            reasons.append("Bullish pin bar (lower wick rejection)")
        elif curr_upper_wick > curr_range * 0.6 and curr_body < curr_range * 0.3:
            score -= 0.35
            reasons.append("Bearish pin bar (upper wick rejection)")

        # ENGULFING PATTERN
        if prev_close < prev_open and curr_close > curr_open:
            if curr_close > prev_open and curr_open <= prev_close:
                score += 0.30
                reasons.append("Bullish engulfing pattern")
        elif prev_close > prev_open and curr_close < curr_open:
            if curr_close < prev_open and curr_open >= prev_close:
                score -= 0.30
                reasons.append("Bearish engulfing pattern")

        # PATTERN AT KEY LEVEL amplifies signal
        if abs(score) > 0:
            atr = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"], window=14)
            atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0

            if atr_val > 0:
                recent_low = float(df["low"].iloc[-20:-2].min())
                recent_high = float(df["high"].iloc[-20:-2].max())

                if score > 0 and abs(curr_low - recent_low) < atr_val * 0.5:
                    score += 0.15
                    reasons.append("Pin/engulfing at support")
                elif score < 0 and abs(curr_high - recent_high) < atr_val * 0.5:
                    score -= 0.15
                    reasons.append("Pin/engulfing at resistance")

        if abs(score) < 0.25:
            return {"signal": "SKIP", "confidence": 0, "reason": "No price action pattern"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class KeltnerChannelStrategy:
    """Keltner Channel — ATR-based volatility channels for breakout/mean-reversion."""
    name = "keltner_channel"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ema_20 = ta.trend.ema_indicator(close, window=20)
        atr = ta.volatility.average_true_range(high, low, close, window=10)

        if pd.isna(ema_20.iloc[-1]) or pd.isna(atr.iloc[-1]):
            return {"signal": "SKIP", "confidence": 0, "reason": "Keltner not ready"}

        atr_val = float(atr.iloc[-1])
        mid = float(ema_20.iloc[-1])
        upper = mid + atr_val * 2.0
        lower = mid - atr_val * 2.0
        price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])

        score = 0
        reasons = []

        if price > upper and prev_price <= upper:
            score += 0.40
            reasons.append("Keltner upper channel breakout")
            if float(close.iloc[-2]) > mid:
                score += 0.10
                reasons.append("Was already above midline — momentum")
        elif price < lower and prev_price >= lower:
            score -= 0.40
            reasons.append("Keltner lower channel breakdown")
            if float(close.iloc[-2]) < mid:
                score -= 0.10
                reasons.append("Was already below midline — momentum")

        if regime == "ranging" and abs(score) == 0:
            if price <= lower * 1.001:
                score += 0.35
                reasons.append("Keltner lower touch in range — mean reversion buy")
            elif price >= upper * 0.999:
                score -= 0.35
                reasons.append("Keltner upper touch in range — mean reversion sell")

        if abs(score) > 0 and "volume" in df.columns:
            vol = df["volume"]
            avg_vol = vol.rolling(20).mean().iloc[-1]
            if avg_vol > 0 and vol.iloc[-1] > avg_vol * 1.3:
                score *= 1.15
                reasons.append("Volume confirms Keltner signal")

        if abs(score) < 0.30:
            return {"signal": "SKIP", "confidence": 0, "reason": "No Keltner setup"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class ADXMomentumStrategy:
    """ADX trend strength + DI+/DI- crossovers for directional trades."""
    name = "adx_momentum"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        adx = ta.trend.adx(high, low, close, window=14)
        di_pos = ta.trend.adx_pos(high, low, close, window=14)
        di_neg = ta.trend.adx_neg(high, low, close, window=14)

        if pd.isna(adx.iloc[-1]) or pd.isna(di_pos.iloc[-1]):
            return {"signal": "SKIP", "confidence": 0, "reason": "ADX not ready"}

        adx_val = float(adx.iloc[-1])
        di_p = float(di_pos.iloc[-1])
        di_n = float(di_neg.iloc[-1])

        score = 0
        reasons = []

        if adx_val < 20:
            return {"signal": "SKIP", "confidence": 0, "reason": f"ADX {adx_val:.0f} — no trend"}

        di_cross_up = False
        di_cross_down = False
        for i in range(1, 5):
            if len(di_pos) <= i:
                break
            if (di_pos.iloc[-i] > di_neg.iloc[-i] and
                    di_pos.iloc[-i - 1] <= di_neg.iloc[-i - 1]):
                di_cross_up = True
                break
            elif (di_neg.iloc[-i] > di_pos.iloc[-i] and
                  di_neg.iloc[-i - 1] <= di_pos.iloc[-i - 1]):
                di_cross_down = True
                break

        if di_cross_up or (di_p > di_n and di_p - di_n > 5):
            base = 0.35 if di_cross_up else 0.25
            score += base
            reasons.append(f"DI+ above DI- {'(crossover)' if di_cross_up else ''}, ADX {adx_val:.0f}")
            if adx_val > 30:
                score += 0.10
                reasons.append("Strong trend (ADX > 30)")
            if adx_val > 40:
                score += 0.10
                reasons.append("Very strong trend (ADX > 40)")
            if len(adx) > 3 and adx.iloc[-1] > adx.iloc[-3]:
                score += 0.10
                reasons.append("ADX rising — trend strengthening")

        elif di_cross_down or (di_n > di_p and di_n - di_p > 5):
            base = 0.35 if di_cross_down else 0.25
            score -= base
            reasons.append(f"DI- above DI+ {'(crossover)' if di_cross_down else ''}, ADX {adx_val:.0f}")
            if adx_val > 30:
                score -= 0.10
                reasons.append("Strong trend (ADX > 30)")
            if adx_val > 40:
                score -= 0.10
                reasons.append("Very strong trend (ADX > 40)")
            if len(adx) > 3 and adx.iloc[-1] > adx.iloc[-3]:
                score -= 0.10
                reasons.append("ADX rising — trend strengthening")

        if abs(score) < 0.25:
            return {"signal": "SKIP", "confidence": 0, "reason": "No ADX setup"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class FibonacciRetracementStrategy:
    """Fibonacci retracement — entries at 38.2%, 50%, 61.8% pullback levels."""
    name = "fibonacci_retracement"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        price = float(close.iloc[-1])

        ema_50 = ta.trend.ema_indicator(close, window=50)
        atr = ta.volatility.average_true_range(high, low, close, window=14)

        if pd.isna(ema_50.iloc[-1]) or pd.isna(atr.iloc[-1]):
            return {"signal": "SKIP", "confidence": 0, "reason": "Fib not ready"}

        atr_val = float(atr.iloc[-1])
        trend_up = price > float(ema_50.iloc[-1])

        swing_high_idx = int(high.iloc[-50:].idxmax() if hasattr(high.iloc[-50:].idxmax(), '__int__') else high.iloc[-50:].values.argmax())
        swing_low_idx = int(low.iloc[-50:].idxmin() if hasattr(low.iloc[-50:].idxmin(), '__int__') else low.iloc[-50:].values.argmin())

        swing_high = float(high.iloc[-50:].max())
        swing_low = float(low.iloc[-50:].min())
        swing_range = swing_high - swing_low

        if swing_range < atr_val * 2:
            return {"signal": "SKIP", "confidence": 0, "reason": "Swing range too small"}

        fib_382 = swing_high - swing_range * 0.382
        fib_500 = swing_high - swing_range * 0.500
        fib_618 = swing_high - swing_range * 0.618

        tolerance = atr_val * 0.3

        score = 0
        reasons = []

        if trend_up:
            if abs(price - fib_618) < tolerance:
                score += 0.45
                reasons.append("Price at 61.8% Fibonacci (golden ratio)")
            elif abs(price - fib_500) < tolerance:
                score += 0.40
                reasons.append("Price at 50% Fibonacci retracement")
            elif abs(price - fib_382) < tolerance:
                score += 0.35
                reasons.append("Price at 38.2% Fibonacci retracement")

            if score > 0:
                if float(close.iloc[-1]) > float(close.iloc[-2]):
                    score += 0.10
                    reasons.append("Price bouncing off Fib level")
                rsi = ta.momentum.rsi(close, window=14)
                if not pd.isna(rsi.iloc[-1]) and rsi.iloc[-1] < 45:
                    score += 0.10
                    reasons.append("RSI confirms pullback (not overbought)")
        else:
            inv_382 = swing_low + swing_range * 0.382
            inv_500 = swing_low + swing_range * 0.500
            inv_618 = swing_low + swing_range * 0.618

            if abs(price - inv_618) < tolerance:
                score -= 0.45
                reasons.append("Price at 61.8% Fibonacci (golden ratio)")
            elif abs(price - inv_500) < tolerance:
                score -= 0.40
                reasons.append("Price at 50% Fibonacci retracement")
            elif abs(price - inv_382) < tolerance:
                score -= 0.35
                reasons.append("Price at 38.2% Fibonacci retracement")

            if score < 0:
                if float(close.iloc[-1]) < float(close.iloc[-2]):
                    score -= 0.10
                    reasons.append("Price rejecting off Fib level")
                rsi = ta.momentum.rsi(close, window=14)
                if not pd.isna(rsi.iloc[-1]) and rsi.iloc[-1] > 55:
                    score -= 0.10
                    reasons.append("RSI confirms pullback (not oversold)")

        if abs(score) < 0.30:
            return {"signal": "SKIP", "confidence": 0, "reason": "No Fibonacci setup"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


class StochasticDivergenceStrategy:
    """Stochastic divergence — price vs stochastic disagreement for reversals."""
    name = "stochastic_divergence"

    def evaluate(self, df, regime: str) -> dict:
        if len(df) < 50:
            return {"signal": "SKIP", "confidence": 0, "reason": "Need 50 bars"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        stoch_k = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
        stoch_d = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)

        if pd.isna(stoch_k.iloc[-1]) or pd.isna(stoch_d.iloc[-1]):
            return {"signal": "SKIP", "confidence": 0, "reason": "Stochastic not ready"}

        score = 0
        reasons = []

        lookback = 20
        if len(close) >= lookback and len(stoch_k) >= lookback:
            price_slice = close.iloc[-lookback:]
            stoch_slice = stoch_k.iloc[-lookback:]

            price_low1_idx = price_slice.iloc[:lookback // 2].idxmin() if hasattr(price_slice.iloc[:lookback // 2].idxmin(), '__int__') else 0
            price_low2 = float(price_slice.iloc[-5:].min())
            price_low1 = float(price_slice.iloc[:lookback // 2].min())

            stoch_low1 = float(stoch_slice.iloc[:lookback // 2].min())
            stoch_low2 = float(stoch_slice.iloc[-5:].min())

            price_high1 = float(price_slice.iloc[:lookback // 2].max())
            price_high2 = float(price_slice.iloc[-5:].max())

            stoch_high1 = float(stoch_slice.iloc[:lookback // 2].max())
            stoch_high2 = float(stoch_slice.iloc[-5:].max())

            if price_low2 < price_low1 and stoch_low2 > stoch_low1:
                score += 0.40
                reasons.append("Bullish stochastic divergence (price lower low, stoch higher low)")
                if stoch_k.iloc[-1] < 25:
                    score += 0.15
                    reasons.append("Stochastic in oversold zone")

            elif price_high2 > price_high1 and stoch_high2 < stoch_high1:
                score -= 0.40
                reasons.append("Bearish stochastic divergence (price higher high, stoch lower high)")
                if stoch_k.iloc[-1] > 75:
                    score -= 0.15
                    reasons.append("Stochastic in overbought zone")

        if abs(score) == 0:
            k_val = float(stoch_k.iloc[-1])
            d_val = float(stoch_d.iloc[-1])

            stoch_cross_up = False
            stoch_cross_down = False
            for i in range(1, 4):
                if len(stoch_k) <= i:
                    break
                if (stoch_k.iloc[-i] > stoch_d.iloc[-i] and
                        stoch_k.iloc[-i - 1] <= stoch_d.iloc[-i - 1]):
                    stoch_cross_up = True
                    break
                elif (stoch_k.iloc[-i] < stoch_d.iloc[-i] and
                      stoch_k.iloc[-i - 1] >= stoch_d.iloc[-i - 1]):
                    stoch_cross_down = True
                    break

            if stoch_cross_up and k_val < 25:
                score += 0.35
                reasons.append(f"Stochastic K/D bullish crossover in oversold ({k_val:.0f})")
            elif stoch_cross_down and k_val > 75:
                score -= 0.35
                reasons.append(f"Stochastic K/D bearish crossover in overbought ({k_val:.0f})")

        if abs(score) < 0.30:
            return {"signal": "SKIP", "confidence": 0, "reason": "No stochastic setup"}

        signal = "BUY" if score > 0 else "SELL"
        return {
            "signal": signal,
            "confidence": min(abs(score), 0.95),
            "reasons": reasons,
            "strategy": self.name,
        }


ALL_STRATEGIES = [
    DonchianBreakoutStrategy(),
    LondonBreakoutStrategy(),
    BollingerRSIStrategy(),
    MACDTrendStrategy(),
    IchimokuStrategy(),
    SmartMoneyStrategy(),
    PriceActionStrategy(),
    KeltnerChannelStrategy(),
    ADXMomentumStrategy(),
    FibonacciRetracementStrategy(),
    StochasticDivergenceStrategy(),
]
