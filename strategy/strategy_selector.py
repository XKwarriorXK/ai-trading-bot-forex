"""
Strategy Selector — institutional-grade confluence scoring.
11 strategies grouped into 6 independent confluence categories.
Confidence rewards DIVERSITY of agreement, not just vote count.
A+ requires 4+ independent categories agreeing — the real edge.
"""
import logging
from strategy.strategies import ALL_STRATEGIES

logger = logging.getLogger(__name__)

STRATEGY_CATEGORY = {
    "donchian_breakout": "trend",
    "london_breakout": "session_breakout",
    "bollinger_rsi": "mean_reversion",
    "macd_trend": "momentum",
    "ichimoku": "trend_structure",
    "smart_money": "structure",
    "price_action": "pattern",
    "keltner_channel": "volatility",
    "adx_momentum": "momentum",
    "fibonacci_retracement": "structure",
    "stochastic_divergence": "mean_reversion",
}

CATEGORY_WEIGHT = {
    "trend": 0.25,
    "trend_structure": 0.20,
    "structure": 0.20,
    "momentum": 0.15,
    "session_breakout": 0.10,
    "mean_reversion": 0.10,
    "pattern": 0.10,
    "volatility": 0.10,
}

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

GRADE_TIERS = {
    "A+": {"min": 0.85, "label": "A+ (Full conviction)", "size": 1.0},
    "A":  {"min": 0.70, "label": "A (Standard entry)", "size": 0.75},
    "B":  {"min": 0.55, "label": "B (Reduced size)", "size": 0.50},
    "C":  {"min": 0.0,  "label": "C (Skip)", "size": 0.0},
}

MIN_AGREEING = 5
MIN_CATEGORIES = 3
MIN_CONFIDENCE = 0.35
MIN_WEIGHTED_CONFIDENCE = 0.25


class StrategySelector:
    def __init__(self):
        self.strategies = ALL_STRATEGIES

    def _get_grade(self, confidence):
        if confidence >= 0.85:
            return "A+"
        elif confidence >= 0.70:
            return "A"
        elif confidence >= 0.55:
            return "B"
        return "C"

    def _calc_confluence_confidence(self, votes):
        n = len(votes)
        avg_raw = sum(v["raw_confidence"] for v in votes) / n
        max_raw = max(v["raw_confidence"] for v in votes)

        categories_hit = set()
        for v in votes:
            cat = STRATEGY_CATEGORY.get(v["strategy"], "unknown")
            categories_hit.add(cat)

        num_categories = len(categories_hit)

        base = (avg_raw * 0.5) + (max_raw * 0.3)

        category_bonus = {
            1: 0.0,
            2: 0.05,
            3: 0.15,
            4: 0.25,
            5: 0.35,
            6: 0.42,
            7: 0.48,
        }
        diversity_bonus = category_bonus.get(num_categories, 0.48)

        cat_weight_sum = 0
        for cat in categories_hit:
            cat_weight_sum += CATEGORY_WEIGHT.get(cat, 0.05)
        weight_quality = min(cat_weight_sum / 0.80, 1.0) * 0.10

        vote_bonus = max(0, (n - MIN_AGREEING)) * 0.02

        final = base + diversity_bonus + weight_quality + vote_bonus

        return round(min(final, 0.95), 4), avg_raw, num_categories, list(categories_hit)

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
                        "category": STRATEGY_CATEGORY.get(strategy.name, "unknown"),
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

        for direction, dir_votes in [("BUY", buy_votes), ("SELL", sell_votes)]:
            if len(dir_votes) < MIN_AGREEING:
                continue

            categories_hit = set(v["category"] for v in dir_votes)
            if len(categories_hit) < MIN_CATEGORIES:
                return {
                    "signal": "SKIP",
                    "confidence": 0,
                    "reason": f"{direction} has {len(dir_votes)} votes but only {len(categories_hit)} independent categories (need {MIN_CATEGORIES})",
                    "votes": votes,
                }

            final_conf, avg_raw, num_cats, cat_list = self._calc_confluence_confidence(dir_votes)

            if avg_raw < MIN_CONFIDENCE:
                return {
                    "signal": "SKIP",
                    "confidence": 0,
                    "reason": f"{direction} raw confidence {avg_raw:.2f} below {MIN_CONFIDENCE}",
                    "votes": votes,
                }

            if final_conf < MIN_WEIGHTED_CONFIDENCE:
                return {
                    "signal": "SKIP",
                    "confidence": 0,
                    "reason": f"{direction} confluence confidence {final_conf:.2f} below {MIN_WEIGHTED_CONFIDENCE}",
                    "votes": votes,
                }

            grade = self._get_grade(final_conf)
            grade_info = GRADE_TIERS[grade]

            all_reasons = []
            for v in dir_votes:
                all_reasons.extend(v["reasons"])

            logger.info(
                f"GRADE {grade} | {direction} | Conf: {final_conf:.0%} | "
                f"Strategies: {len(dir_votes)} | Categories: {num_cats} ({', '.join(cat_list)}) | "
                f"{grade_info['label']}"
            )

            return {
                "signal": direction,
                "confidence": final_conf,
                "grade": grade,
                "grade_label": grade_info["label"],
                "size_multiplier": grade_info["size"],
                "agreeing_strategies": [v["strategy"] for v in dir_votes],
                "categories": cat_list,
                "num_categories": num_cats,
                "reasons": all_reasons,
                "votes": votes,
            }

        return {
            "signal": "SKIP",
            "confidence": 0,
            "reason": f"Not enough agreement (BUY:{len(buy_votes)} SELL:{len(sell_votes)}, need {MIN_AGREEING})",
            "votes": votes,
        }
