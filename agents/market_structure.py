"""
Market Structure Agent — detects HH/HL/LL/LH, support/resistance, liquidity zones.
Institutional-level price action analysis.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MarketStructureAgent:
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def analyze(self, df) -> dict:
        if len(df) < self.lookback * 2:
            return {"structure": "unknown", "bias": "neutral", "confidence": 0}

        highs = df["high"].values
        lows = df["low"].values
        close = df["close"].values

        swing_highs = self._find_swing_points(highs, is_high=True)
        swing_lows = self._find_swing_points(lows, is_high=False)

        structure = self._classify_structure(swing_highs, swing_lows)

        support, resistance = self._find_sr_levels(highs, lows, close)

        liquidity_zones = self._find_liquidity_zones(highs, lows, swing_highs, swing_lows)

        price = close[-1]
        nearest_support = max([s for s in support if s < price], default=None)
        nearest_resistance = min([r for r in resistance if r > price], default=None)

        return {
            "structure": structure["type"],
            "bias": structure["bias"],
            "confidence": structure["confidence"],
            "swing_highs": swing_highs[-3:] if swing_highs else [],
            "swing_lows": swing_lows[-3:] if swing_lows else [],
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "support_levels": support[:3],
            "resistance_levels": resistance[:3],
            "liquidity_zones": liquidity_zones,
            "distance_to_support_pct": ((price - nearest_support) / price * 100) if nearest_support else None,
            "distance_to_resistance_pct": ((nearest_resistance - price) / price * 100) if nearest_resistance else None,
        }

    def _find_swing_points(self, data, is_high=True, window=5) -> list:
        points = []
        for i in range(window, len(data) - window):
            if is_high:
                if all(data[i] >= data[i - j] for j in range(1, window + 1)) and \
                   all(data[i] >= data[i + j] for j in range(1, window + 1)):
                    points.append({"index": i, "price": float(data[i])})
            else:
                if all(data[i] <= data[i - j] for j in range(1, window + 1)) and \
                   all(data[i] <= data[i + j] for j in range(1, window + 1)):
                    points.append({"index": i, "price": float(data[i])})
        return points

    def _classify_structure(self, swing_highs, swing_lows) -> dict:
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"type": "undefined", "bias": "neutral", "confidence": 0}

        recent_highs = [p["price"] for p in swing_highs[-3:]]
        recent_lows = [p["price"] for p in swing_lows[-3:]]

        hh = all(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))
        hl = all(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))
        ll = all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))
        lh = all(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))

        if hh and hl:
            return {"type": "uptrend", "bias": "bullish", "confidence": 0.80}
        elif ll and lh:
            return {"type": "downtrend", "bias": "bearish", "confidence": 0.80}
        elif hh and not hl:
            return {"type": "weakening_uptrend", "bias": "bullish", "confidence": 0.55}
        elif ll and not lh:
            return {"type": "weakening_downtrend", "bias": "bearish", "confidence": 0.55}
        else:
            return {"type": "ranging", "bias": "neutral", "confidence": 0.50}

    def _find_sr_levels(self, highs, lows, close, tolerance_pct=0.002) -> tuple:
        all_levels = list(highs[-50:]) + list(lows[-50:])

        clusters = []
        used = set()
        for i, level in enumerate(all_levels):
            if i in used:
                continue
            cluster = [level]
            for j, other in enumerate(all_levels):
                if j != i and j not in used:
                    if abs(level - other) / level < tolerance_pct:
                        cluster.append(other)
                        used.add(j)
            if len(cluster) >= 2:
                clusters.append({
                    "price": float(np.mean(cluster)),
                    "touches": len(cluster),
                })
            used.add(i)

        clusters.sort(key=lambda x: x["touches"], reverse=True)

        current = close[-1]
        support = sorted(
            [c["price"] for c in clusters if c["price"] < current],
            reverse=True,
        )
        resistance = sorted(
            [c["price"] for c in clusters if c["price"] > current],
        )

        return support, resistance

    def _find_liquidity_zones(self, highs, lows, swing_highs, swing_lows) -> list:
        zones = []

        for sh in swing_highs[-5:]:
            zones.append({
                "type": "sell_liquidity",
                "price": sh["price"],
                "description": "Stops above swing high",
            })

        for sl in swing_lows[-5:]:
            zones.append({
                "type": "buy_liquidity",
                "price": sl["price"],
                "description": "Stops below swing low",
            })

        return zones
