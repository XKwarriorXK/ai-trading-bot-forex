"""
News/Economic Calendar Agent — reduces risk around high-impact events.
Uses approximate calendar for major events (FOMC, NFP, CPI, ECB, BOE, BOJ).
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

HIGH_IMPACT_EVENTS = {
    "FOMC": {
        "currencies": ["USD"],
        "typical_days": ["wednesday"],
        "months": [1, 3, 5, 6, 7, 9, 11, 12],
        "risk_hours_before": 4,
        "risk_hours_after": 2,
    },
    "NFP": {
        "currencies": ["USD"],
        "typical_days": ["friday"],
        "week_of_month": 1,
        "risk_hours_before": 2,
        "risk_hours_after": 1,
    },
    "CPI": {
        "currencies": ["USD"],
        "typical_days": ["tuesday", "wednesday", "thursday"],
        "week_of_month": 2,
        "risk_hours_before": 2,
        "risk_hours_after": 1,
    },
    "ECB": {
        "currencies": ["EUR"],
        "typical_days": ["thursday"],
        "frequency": 6,
        "risk_hours_before": 3,
        "risk_hours_after": 2,
    },
    "BOE": {
        "currencies": ["GBP"],
        "typical_days": ["thursday"],
        "frequency": 8,
        "risk_hours_before": 3,
        "risk_hours_after": 2,
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


class NewsAgent:
    def check_risk(self, instrument: str) -> dict:
        now = datetime.now(timezone.utc)
        currencies = CURRENCY_MAP.get(instrument, [])
        risks = []

        for event_name, event in HIGH_IMPACT_EVENTS.items():
            affected_currencies = event["currencies"]
            if not any(c in currencies for c in affected_currencies):
                continue

            day_name = now.strftime("%A").lower()
            if day_name in event.get("typical_days", []):
                risks.append({
                    "event": event_name,
                    "risk_level": "high",
                    "reason": f"{event_name} may be scheduled today",
                })

        if not risks:
            return {
                "risk_level": "low",
                "tradeable": True,
                "events": [],
                "confidence_modifier": 0,
            }

        high_risk = any(r["risk_level"] == "high" for r in risks)
        return {
            "risk_level": "high" if high_risk else "medium",
            "tradeable": True,
            "events": risks,
            "confidence_modifier": -0.10 if high_risk else -0.05,
            "size_reduction_pct": 50 if high_risk else 25,
        }
