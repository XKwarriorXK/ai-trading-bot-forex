"""
Spread Filter — blocks trades when spreads are abnormally wide.
"""
import logging
from config.settings import INSTRUMENTS

logger = logging.getLogger(__name__)


class SpreadFilter:
    MAX_SPREAD_MULTIPLIER = 3.0

    def check(self, instrument: str, current_spread: float) -> dict:
        spec = INSTRUMENTS.get(instrument, {})
        avg_spread = spec.get("spread_avg", 2.0)
        pip_loc = spec.get("pip_location", -4)
        pip_value = 10 ** pip_loc

        spread_pips = current_spread / pip_value
        max_allowed = avg_spread * self.MAX_SPREAD_MULTIPLIER

        if spread_pips > max_allowed:
            return {
                "tradeable": False,
                "spread_pips": round(spread_pips, 1),
                "max_allowed": round(max_allowed, 1),
                "reason": f"Spread {spread_pips:.1f} pips > max {max_allowed:.1f} pips",
            }

        return {
            "tradeable": True,
            "spread_pips": round(spread_pips, 1),
            "spread_quality": "excellent" if spread_pips < avg_spread else
                             "normal" if spread_pips < avg_spread * 2 else "wide",
        }
