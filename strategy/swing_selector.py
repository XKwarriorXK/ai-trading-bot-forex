"""
Swing Selector — institutional top-down analysis.

    Daily → direction + supply/demand zones (WHERE to trade)
    H4   → gates confirm trend is intact
    H4   → 15 strategies + zone vote for entry (WHEN to enter)

Flow:
    Tier 1 — GATES (must ALL pass):
        MA Trend Filter:     50/200 MA determines direction
        Market Structure:    HH/HL or LL/LH must be intact
        ADX Strength:        ADX > 20 confirms trend exists

    Tier 2 — ENTRY SIGNALS (need MIN_ENTRIES to agree):
        Daily S/D Zone:      Price at institutional supply/demand level
        4 Swing entries:     FVG, divergence, pullback, liquidity sweep
        11 Scalp strategies: Additional confirmation from full strategy suite

Zone at an institutional level + strategies confirming = high conviction entry.
"""
import logging
from strategy.swing_strategies import (
    MATrendGate, StructureGate, ADXGate,
    FairValueGapStrategy, DivergenceStrategy,
    TrendPullbackStrategy, LiquiditySweepStrategy,
)
from strategy.strategies import ALL_STRATEGIES as SCALP_STRATEGIES

logger = logging.getLogger(__name__)

MIN_ENTRIES = 2


class SwingSelector:
    def __init__(self):
        self.ma_gate = MATrendGate()
        self.struct_gate = StructureGate()
        self.adx_gate = ADXGate()
        self.swing_entries = [
            FairValueGapStrategy(),
            DivergenceStrategy(),
            TrendPullbackStrategy(),
            LiquiditySweepStrategy(),
        ]
        self.scalp_strategies = SCALP_STRATEGIES

    def evaluate(self, df, regime: str, instrument: str = None, zone=None) -> dict:
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

        # --- Daily supply/demand zone (strongest signal) ---
        at_zone = False
        if zone:
            zone_signal = "BUY" if zone["type"] == "demand" else "SELL"
            if zone_signal == allowed:
                at_zone = True
                zone_conf = min(0.70 + zone.get("strength", 1.5) * 0.03, 0.90)
                entry_votes.append({
                    "strategy": f"daily_{zone['type']}_zone",
                    "signal": zone_signal,
                    "confidence": zone_conf,
                    "category": "zone_entry",
                    "reasons": [f"Price at daily {zone['type']} zone [{zone['bottom']:.5f}-{zone['top']:.5f}]"],
                })
                all_reasons.append(f"At daily {zone['type']} zone (strength {zone.get('strength', 0)}x ATR)")

        # --- 4 swing entry strategies ---
        for strat in self.swing_entries:
            try:
                result = strat.evaluate(df, regime)
                if result["signal"] == allowed:
                    entry_votes.append({
                        "strategy": strat.name,
                        "signal": result["signal"],
                        "confidence": result["confidence"],
                        "category": f"swing_{strat.name}",
                        "reasons": result.get("reasons", []),
                    })
                    all_reasons.extend(result.get("reasons", []))
            except Exception as e:
                logger.warning(f"Swing strategy {strat.name} failed: {e}")

        # --- 11 scalp strategies as additional confirmation ---
        scalp_confirms = 0
        for strat in self.scalp_strategies:
            try:
                result = strat.evaluate(df, regime)
                if result.get("signal") == allowed:
                    scalp_confirms += 1
            except Exception:
                pass

        if scalp_confirms >= 3:
            scalp_conf = min(0.55 + scalp_confirms * 0.03, 0.80)
            entry_votes.append({
                "strategy": f"scalp_ensemble_{scalp_confirms}",
                "signal": allowed,
                "confidence": scalp_conf,
                "category": "scalp_confirmation",
                "reasons": [f"{scalp_confirms}/11 scalp strategies confirm {allowed}"],
            })
            all_reasons.append(f"{scalp_confirms} scalp strategies confirm")

        if len(entry_votes) < MIN_ENTRIES:
            names = [v["strategy"] for v in entry_votes]
            return {"signal": "SKIP", "confidence": 0,
                    "reason": f"Only {len(entry_votes)} entry signals {names} (need {MIN_ENTRIES})",
                    "votes": entry_votes}

        # === CONFIDENCE ===
        gate_conf = (ma["confidence"] + struct["confidence"] + adx["confidence"]) / 3
        entry_conf = sum(v["confidence"] for v in entry_votes) / len(entry_votes)

        final_conf = gate_conf * 0.40 + entry_conf * 0.60

        if at_zone:
            final_conf += 0.05
        if adx.get("adx", 0) >= 30:
            final_conf += 0.03
        if len(entry_votes) >= 3:
            final_conf += 0.05
        if len(entry_votes) >= 4:
            final_conf += 0.05
        if len(entry_votes) >= 5:
            final_conf += 0.03

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
        zone_tag = f" @ {zone['type']} zone" if at_zone else ""

        logger.info(
            f"SWING {grade} | {allowed}{zone_tag} | Conf: {final_conf:.0%} | "
            f"Gates: MA+Struct+ADX({adx_val:.0f}) | "
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
            "at_zone": at_zone,
        }
