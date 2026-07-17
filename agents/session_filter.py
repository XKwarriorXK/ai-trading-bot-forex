"""
Session Filter — checks forex market hours and session overlaps.
Best trading happens during session overlaps (London/NY, Tokyo/London).
"""
import logging
from datetime import datetime, timezone
from config.settings import SESSIONS

logger = logging.getLogger(__name__)


class SessionFilter:
    INSTRUMENT_SESSIONS = {
        "EUR_USD": ["london", "new_york"],
        "GBP_USD": ["london", "new_york"],
        "USD_JPY": ["tokyo", "new_york"],
        "AUD_USD": ["sydney", "tokyo"],
        "USD_CAD": ["new_york"],
        "EUR_GBP": ["london"],
        "EUR_JPY": ["tokyo", "london"],
        "GBP_JPY": ["tokyo", "london"],
        "NZD_USD": ["sydney", "tokyo"],
        "USD_CHF": ["london", "new_york"],
    }

    def check(self, instrument: str) -> dict:
        now = datetime.now(timezone.utc)
        hour = now.hour
        active_sessions = self._get_active_sessions(hour)
        preferred = self.INSTRUMENT_SESSIONS.get(instrument, ["london", "new_york"])
        in_preferred = any(s in active_sessions for s in preferred)
        is_overlap = len(active_sessions) >= 2

        if not active_sessions:
            return {
                "tradeable": False,
                "sessions": [],
                "reason": "No major session active",
            }

        return {
            "tradeable": True,
            "sessions": active_sessions,
            "in_preferred_session": in_preferred,
            "is_overlap": is_overlap,
            "confidence_boost": 0.05 if is_overlap else 0,
        }

    def _get_active_sessions(self, hour: int) -> list:
        active = []
        for name, times in SESSIONS.items():
            if times["open"] <= times["close"]:
                if times["open"] <= hour < times["close"]:
                    active.append(name)
            else:
                if hour >= times["open"] or hour < times["close"]:
                    active.append(name)
        return active
