"""
Strategy Selector — picks best strategy based on market regime.
Multiple strategies vote, selector weights by regime fitness.
"""
import logging
from strategy.strategies import ALL_STRATEGIES

logger = logging.getLogger(__name__)

REGIME_WEIGHTS = {
    "trending": {
        "trend_following": 1.0,
        "momentum": 0.8,
        "breakout": 0.5,
        "mean_reversion": 0.1,
    },
    "ranging": {
        "mean_reversion": 1.0,
        "breakout": 0.3,
        "momentum": 0.2,
        "trend_following": 0.1,
    },
    "volatile": {
        "breakout": 0.8,
        "momentum": 0.6,
        "trend_following": 0.3,
        "mean_reversion": 0.2,
    },
    "transitioning": {
        "momentum": 0.7,
        "breakout": 0.6,
        "trend_following": 0.5,
        "mean_reversion": 0.5,
    },
}

MIN_AGREEING = 2
MIN_CONFIDENCE = 0.40
MIN_WEIGHTED_CONFIDENCE = 0.30


class StrategySelector:
    def __init__(self):
        self.strategies = ALL_STRATEGIES

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
            avg_raw = sum(v["raw_confidence"] for v in buy_votes) / len(buy_votes)
            avg_weighted = sum(v["weighted_confidence"] for v in buy_votes) / len(buy_votes)
            final_conf = round(min(avg_weighted + 0.05 * (len(buy_votes) - MIN_AGREEING), 0.95), 4)
            if avg_raw >= MIN_CONFIDENCE and final_conf >= MIN_WEIGHTED_CONFIDENCE:
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
            avg_raw = sum(v["raw_confidence"] for v in sell_votes) / len(sell_votes)
            avg_weighted = sum(v["weighted_confidence"] for v in sell_votes) / len(sell_votes)
            final_conf = round(min(avg_weighted + 0.05 * (len(sell_votes) - MIN_AGREEING), 0.95), 4)
            if avg_raw >= MIN_CONFIDENCE and final_conf >= MIN_WEIGHTED_CONFIDENCE:
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
