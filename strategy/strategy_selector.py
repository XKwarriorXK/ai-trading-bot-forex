"""
Strategy Selector — picks best strategy based on market regime.
11 proven strategies vote, selector weights by regime fitness.
Consensus-driven confidence: more strategies agreeing = exponentially higher confidence.
"""
import logging
from strategy.strategies import ALL_STRATEGIES

logger = logging.getLogger(__name__)

REGIME_WEIGHTS = {
    "trending": {
        "donchian_breakout": 0.9,
        "london_breakout": 0.7,
        "bollinger_rsi": 0.4,
        "macd_trend": 1.0,
        "ichimoku": 0.9,
        "smart_money": 0.6,
        "price_action": 0.7,
        "keltner_channel": 0.85,
        "adx_momentum": 1.0,
        "fibonacci_retracement": 0.8,
        "stochastic_divergence": 0.4,
    },
    "ranging": {
        "donchian_breakout": 0.4,
        "london_breakout": 0.5,
        "bollinger_rsi": 1.0,
        "macd_trend": 0.4,
        "ichimoku": 0.5,
        "smart_money": 0.8,
        "price_action": 0.9,
        "keltner_channel": 0.7,
        "adx_momentum": 0.4,
        "fibonacci_retracement": 0.6,
        "stochastic_divergence": 1.0,
    },
    "volatile": {
        "donchian_breakout": 0.7,
        "london_breakout": 0.8,
        "bollinger_rsi": 0.5,
        "macd_trend": 0.5,
        "ichimoku": 0.5,
        "smart_money": 0.7,
        "price_action": 0.8,
        "keltner_channel": 0.9,
        "adx_momentum": 0.6,
        "fibonacci_retracement": 0.5,
        "stochastic_divergence": 0.7,
    },
    "transitioning": {
        "donchian_breakout": 0.5,
        "london_breakout": 0.6,
        "bollinger_rsi": 0.6,
        "macd_trend": 0.6,
        "ichimoku": 0.6,
        "smart_money": 0.7,
        "price_action": 0.7,
        "keltner_channel": 0.6,
        "adx_momentum": 0.6,
        "fibonacci_retracement": 0.6,
        "stochastic_divergence": 0.6,
    },
}

CONSENSUS_BONUS = {
    2: 0.0,
    3: 0.10,
    4: 0.20,
    5: 0.30,
    6: 0.38,
    7: 0.45,
    8: 0.50,
    9: 0.55,
    10: 0.58,
    11: 0.60,
}

MIN_AGREEING = 2
MIN_CONFIDENCE = 0.35
MIN_WEIGHTED_CONFIDENCE = 0.25


class StrategySelector:
    def __init__(self):
        self.strategies = ALL_STRATEGIES

    def _calc_confidence(self, votes):
        n = len(votes)
        avg_raw = sum(v["raw_confidence"] for v in votes) / n
        max_raw = max(v["raw_confidence"] for v in votes)

        base = (avg_raw * 0.6) + (max_raw * 0.4)

        bonus = CONSENSUS_BONUS.get(n, 0.60)

        final = base + bonus

        return round(min(final, 0.95), 4), avg_raw

    def evaluate(self, df, regime: str) -> dict:
        votes = []
        for strategy in self.strategies:
            try:
                result = strategy.evaluate(df, regime)
                if result["signal"] != "SKIP":
                    weight = REGIME_WEIGHTS.get(regime, {}).get(strategy.name, 0.5)
                    weighted_conf = result["confidence"] * weight
                    votes.append({
                        "strategy": strategy.name,
                        "signal": result["signal"],
                        "raw_confidence": result["confidence"],
                        "weighted_confidence": weighted_conf,
                        "reasons": result.get("reasons", []),
                    })
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")

        if not votes:
            return {
                "signal": "SKIP",
                "confidence": 0,
                "reason": "No strategies produced signals",
                "votes": [],
            }

        buy_votes = [v for v in votes if v["signal"] == "BUY"]
        sell_votes = [v for v in votes if v["signal"] == "SELL"]

        if len(buy_votes) >= MIN_AGREEING:
            final_conf, avg_raw = self._calc_confidence(buy_votes)
            if avg_raw < MIN_CONFIDENCE:
                return {
                    "signal": "SKIP",
                    "confidence": 0,
                    "reason": f"BUY raw confidence {avg_raw:.2f} below {MIN_CONFIDENCE}",
                    "votes": votes,
                }
            if final_conf < MIN_WEIGHTED_CONFIDENCE:
                return {
                    "signal": "SKIP",
                    "confidence": 0,
                    "reason": f"BUY confidence {final_conf:.2f} below {MIN_WEIGHTED_CONFIDENCE}",
                    "votes": votes,
                }
            all_reasons = []
            for v in buy_votes:
                all_reasons.extend(v["reasons"])
            return {
                "signal": "BUY",
                "confidence": final_conf,
                "agreeing_strategies": [v["strategy"] for v in buy_votes],
                "reasons": all_reasons,
                "votes": votes,
            }

        if len(sell_votes) >= MIN_AGREEING:
            final_conf, avg_raw = self._calc_confidence(sell_votes)
            if avg_raw < MIN_CONFIDENCE:
                return {
                    "signal": "SKIP",
                    "confidence": 0,
                    "reason": f"SELL raw confidence {avg_raw:.2f} below {MIN_CONFIDENCE}",
                    "votes": votes,
                }
            if final_conf < MIN_WEIGHTED_CONFIDENCE:
                return {
                    "signal": "SKIP",
                    "confidence": 0,
                    "reason": f"SELL confidence {final_conf:.2f} below {MIN_WEIGHTED_CONFIDENCE}",
                    "votes": votes,
                }
            all_reasons = []
            for v in sell_votes:
                all_reasons.extend(v["reasons"])
            return {
                "signal": "SELL",
                "confidence": final_conf,
                "agreeing_strategies": [v["strategy"] for v in sell_votes],
                "reasons": all_reasons,
                "votes": votes,
            }

        return {
            "signal": "SKIP",
            "confidence": 0,
            "reason": f"Not enough agreement (BUY:{len(buy_votes)} SELL:{len(sell_votes)}, need {MIN_AGREEING})",
            "votes": votes,
        }
