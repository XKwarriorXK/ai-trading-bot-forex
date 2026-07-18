"""
News/Economic Calendar Agent — reduces risk around high-impact events.
Uses calendar rules: checks month, week-of-month, and day to avoid
false positives on events that only happen monthly.
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

HIGH_IMPACT_EVENTS = {
    "FOMC": {
        "currencies": ["USD"],
        "typical_days": ["wednesday"],
        "months": [1, 3, 5, 6, 7, 9, 11, 12],
        "week_of_month": [3, 4],
    },
    "NFP": {
        "currencies": ["USD"],
        "typical_days": ["friday"],
        "week_of_month": [1],
    },
    "CPI": {
        "currencies": ["USD"],
        "typical_days": ["tuesday", "wednesday", "thursday"],
        "week_of_month": [2],
    },
    "ECB": {
        "currencies": ["EUR"],
        "typical_days": ["thursday"],
        "months": [1, 3, 4, 6, 7, 9, 10, 12],
        "week_of_month": [2, 3],
    },
    "BOE": {
        "currencies": ["GBP"],
        "typical_days": ["thursday"],
        "months": [2, 3, 5, 6, 8, 9, 11, 12],
        "week_of_month": [1, 2],
    },
}

CURRENCY_MAP = {
    "EUR_USD": ["EUR", "USD"],
    "GBP_USD": ["GBP", "USD"],
    "USD_JPY": ["USD", "JPY"],
    "AUD_USD": ["AUD", "USD"],
    "USD_CAD": ["USD", "CAD"],
    "EUR_GBP": ["EUR", "GBP"],
    "EUR_JPY": ["EUR", "JPY"],
    "GBP_JPY": ["GBP", "JPY"],
    "NZD_USD": ["NZD", "USD"],
    "USD_CHF": ["USD", "CHF"],
}


def _week_of_month(dt):
    first_day = dt.replace(day=1)
    adjusted = dt.day + first_day.weekday()
    return (adjusted - 1) // 7 + 1


class NewsAgent:
    def check_risk(self, instrument: str, timestamp=None) -> dict:
        now = timestamp if timestamp else datetime.now(timezone.utc)
        currencies = CURRENCY_MAP.get(instrument, [])
        risks = []

        day_name = now.strftime("%A").lower()
        month = now.month
        week = _week_of_month(now)

        for event_name, event in HIGH_IMPACT_EVENTS.items():
            if not any(c in currencies for c in event["currencies"]):
                continue

            if day_name not in event.get("typical_days", []):
                continue

            if "months" in event and month not in event["months"]:
                continue

            if "week_of_month" in event and week not in event["week_of_month"]:
                continue

            risks.append({
                "event": event_name,
                "risk_level": "high",
                "reason": f"{event_name} likely scheduled today (week {week})",
            })

        if not risks:
            return {
                "risk_level": "low",
                "tradeable": True,
                "events": [],
                "confidence_modifier": 0,
            }

        return {
            "risk_level": "high",
            "tradeable": True,
            "events": risks,
            "confidence_modifier": -0.10,
            "size_reduction_pct": 50,
        }
