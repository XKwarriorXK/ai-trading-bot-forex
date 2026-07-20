"""
Swing Selector — two-tier gate + entry evaluation.

Unlike the scalp selector where all 11 strategies vote equally,
the swing selector enforces a strict hierarchy:

    Tier 1 — GATES (must ALL pass):
        MA Trend Filter:     Determines allowed direction (BUY/SELL only)
        Market Structure:    Must confirm trend is healthy (HH/HL or LL/LH)

    Tier 2 — ENTRY SIGNALS (need MIN_ENTRIES to agree):
        Fair Value Gap:      Price at institutional imbalance zone
        Divergence:          RSI/MACD disagreeing with price
        Trend Pullback:      Price retracing to key moving averages
        Liquidity Sweep:     Stop hunt then reversal

Only entry signals matching the gate direction are counted.
"""
import logging
from strategy.swing_strategies import (
    MATrendGate, StructureGate, ADXGate,
    FairValueGapStrategy, DivergenceStrategy,
    TrendPullbackStrategy, LiquiditySweepStrategy,
)

logger = logging.getLogger(__name__)

SWING_CATEGORY = {
    "ma_trend_gate": "trend_gate",
    "structure_gate": "structure_gate",
    "adx_gate": "strength_gate",
    "fair_value_gap": "entry_imbalance",
    "divergence": "entry_momentum",
    "trend_pullback": "entry_retracement",
    "liquidity_sweep": "entry_reversal",
}

MIN_ENTRIES = 2


class SwingSelector:
    def __init__(self):
        self.ma_gate = MATrendGate()
        self.struct_gate = StructureGate()
        self.adx_gate = ADXGate()
        self.entry_strategies = [
            FairValueGapStrategy(),
            DivergenceStrategy(),
            TrendPullbackStrategy(),
            LiquiditySweepStrategy(),
        ]

    def evaluate(self, df, regime: str, instrument: str = None) -> dict:
        # === GATE 1: MA TREND ===
        ma = self.ma_gate.evaluate(df, regime)
        if ma["signal"] == "SKIP":
            return {"signal": "SKIP", "confidence": 0,
                    "reason": f"MA gate blocked: {ma.get('reason', 'no trend')}",
                    "votes": []}

        allowed = ma["signal"]

        # === GATE 2: MARKET STRUCTURE ===
        struct = self.struct_gate.evaluate(df, regime)
        if struct["signal"] == "SKIP":
            return {"signal": "SKIP", "confidence": 0,
                    "reason": f"Structure gate blocked: {struct.get('reason', 'no structure')}",
                    "votes": []}

        if struct["signal"] != allowed:
            return {"signal": "SKIP", "confidence": 0,
                    "reason": f"Structure ({struct['signal']}) conflicts with MA trend ({allowed})",
                    "votes": []}

        # === GATE 3: ADX TREND STRENGTH ===
        adx = self.adx_gate.evaluate(df, regime)
        if adx["signal"] == "SKIP":
            return {"signal": "SKIP", "confidence": 0,
                    "reason": f"ADX gate blocked: {adx.get('reason', 'weak trend')}",
                    "votes": []}

        # === ENTRY SIGNALS ===
        entry_votes = []
        all_reasons = list(ma.get("reasons", [])) + list(struct.get("reasons", []))

        for strat in self.entry_strategies:
            try:
                result = strat.evaluate(df, regime)
                if result["signal"] == allowed:
                    entry_votes.append({
                        "strategy": strat.name,
                        "signal": result["signal"],
                        "confidence": result["confidence"],
                        "category": SWING_CATEGORY.get(strat.name, "unknown"),
                        "reasons": result.get("reasons", []),
                    })
                    all_reasons.extend(result.get("reasons", []))
            except Exception as e:
                logger.warning(f"Swing strategy {strat.name} failed: {e}")

        if len(entry_votes) < MIN_ENTRIES:
            names = [v["strategy"] for v in entry_votes]
            return {"signal": "SKIP", "confidence": 0,
                    "reason": f"Only {len(entry_votes)} entry signals {names} (need {MIN_ENTRIES})",
                    "votes": entry_votes}

        # === CONFIDENCE ===
        gate_conf = (ma["confidence"] + struct["confidence"] + adx["confidence"]) / 3
        entry_conf = sum(v["confidence"] for v in entry_votes) / len(entry_votes)

        final_conf = gate_conf * 0.45 + entry_conf * 0.55

        if adx.get("adx", 0) >= 30:
            final_conf += 0.03

        if len(entry_votes) >= 3:
            final_conf += 0.05
        if len(entry_votes) >= 4:
            final_conf += 0.05

        final_conf = round(min(final_conf, 0.95), 4)

        if final_conf >= 0.85:
            grade = "A+"
        elif final_conf >= 0.70:
            grade = "A"
        elif final_conf >= 0.55:
            grade = "B"
        else:
            grade = "C"

        grade_labels = {
            "A+": "A+ (Full conviction swing)",
            "A": "A (Standard swing entry)",
            "B": "B (Reduced confidence)",
            "C": "C (Skip)",
        }

        categories = list(set(v["category"] for v in entry_votes))

        adx_val = adx.get("adx", 0)
        logger.info(
            f"SWING {grade} | {allowed} | Conf: {final_conf:.0%} | "
            f"Gates: MA+Structure+ADX({adx_val:.0f}) | "
            f"Entries: {len(entry_votes)} ({', '.join(v['strategy'] for v in entry_votes)}) | "
            f"{grade_labels.get(grade, grade)}"
        )

        return {
            "signal": allowed,
            "confidence": final_conf,
            "grade": grade,
            "grade_label": grade_labels.get(grade, grade),
            "size_multiplier": 1.0 if grade in ("A+", "A") else 0.0,
            "agreeing_strategies": [v["strategy"] for v in entry_votes],
            "categories": categories,
            "num_categories": len(categories),
            "reasons": all_reasons,
            "votes": entry_votes,
            "gates_passed": ["ma_trend_gate", "structure_gate", "adx_gate"],
            "structure": struct,
        }
