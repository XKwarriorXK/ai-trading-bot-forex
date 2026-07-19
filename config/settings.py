"""
Master configuration — forex bot settings.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# -- AI Provider Config --
AI_PROVIDERS = {
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY"),
        "models": {
            "fast": "llama-3.3-70b-versatile",
            "reasoning": "llama-3.3-70b-versatile",
            "small": "llama-3.1-8b-instant",
            "gpt120": "openai/gpt-oss-120b",
            "gpt20": "openai/gpt-oss-20b",
        },
        "base_url": "https://api.groq.com/openai/v1",
    },
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "models": {
            "flash": "gemini-3.5-flash",
            "pro": "gemini-3.1-pro-preview",
        },
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}

TASK_ROUTING = {
    "market_analysis":  ["groq:gpt120", "gemini:flash", "groq:fast"],
    "trade_decision":   ["gemini:pro", "groq:gpt120", "groq:fast"],
    "risk_assessment":  ["groq:gpt120", "gemini:flash", "groq:fast"],
    "sentiment":        ["groq:gpt20", "gemini:flash", "groq:fast"],
    "reflection":       ["gemini:pro", "groq:gpt120", "groq:fast"],
    "regime_detection": ["groq:gpt20", "gemini:flash", "groq:fast"],
    "debate":           ["gemini:pro", "groq:gpt120", "groq:fast"],
    "strategy_select":  ["groq:gpt20", "gemini:flash", "groq:fast"],
    "macro_analysis":   ["gemini:pro", "groq:gpt120", "groq:fast"],
}

# L2 Fast Screening → L3 Senior Panel → L4 Final Approver
APPROVAL_CHAIN = {
    "level_2": {
        "reviewers": [
            {"provider": "groq", "model": "gpt20", "role": "trend_screener"},
            {"provider": "groq", "model": "small", "role": "momentum_screener"},
            {"provider": "groq", "model": "gpt120", "role": "risk_screener"},
        ],
        "min_pass": 2,
    },
    "level_3": {
        "reviewers": [
            {"provider": "groq", "model": "gpt120", "role": "senior_technical"},
            {"provider": "groq", "model": "gpt20", "role": "senior_structure"},
        ],
        "min_pass": 2,
    },
    "level_4": {
        "provider": "gemini",
        "model": "flash",
    },
}

CIRCUIT_BREAKER = {
    "max_failures": 20,
    "cooldown_seconds": 30,
}

TOKEN_BUDGET = {
    "max_per_hour": 100_000,
    "priority_thresholds": {
        "high": 1.0,
        "medium": 0.80,
        "low": 0.50,
    },
}


# -- OANDA Config --
OANDA = {
    "api_key": os.getenv("OANDA_API_KEY"),
    "account_id": os.getenv("OANDA_ACCOUNT_ID"),
    "environment": os.getenv("OANDA_ENVIRONMENT", "practice"),
}


# -- Trading Config --
TRADING = {
    "mode": os.getenv("TRADING_MODE", "paper"),
    "leverage": 50,
}

WATCHLIST = [
    "GBP_JPY",
    "EUR_JPY",
    "GBP_USD",
    "USD_JPY",
    "EUR_USD",
]

INSTRUMENTS = {
    "EUR_USD": {"pip_location": -4, "min_units": 1, "spread_avg": 1.2},
    "GBP_USD": {"pip_location": -4, "min_units": 1, "spread_avg": 1.5},
    "USD_JPY": {"pip_location": -2, "min_units": 1, "spread_avg": 1.3},
    "AUD_USD": {"pip_location": -4, "min_units": 1, "spread_avg": 1.4},
    "USD_CAD": {"pip_location": -4, "min_units": 1, "spread_avg": 1.8},
    "EUR_GBP": {"pip_location": -4, "min_units": 1, "spread_avg": 1.5},
    "EUR_JPY": {"pip_location": -2, "min_units": 1, "spread_avg": 2.0},
    "GBP_JPY": {"pip_location": -2, "min_units": 1, "spread_avg": 2.5},
    "NZD_USD": {"pip_location": -4, "min_units": 1, "spread_avg": 1.8},
    "USD_CHF": {"pip_location": -4, "min_units": 1, "spread_avg": 1.6},
}

TIMEFRAMES = {
    "entry": "H1",
    "trend": "H4",
    "macro": "D",
}


# -- Risk Config --
RISK = {
    "max_daily_loss_pct": 2.0,
    "max_position_pct": 2.0,
    "max_trades_per_day": 6,
    "risk_per_trade_pct": 1.0,
    "default_stop_loss_pips": 30,
    "default_take_profit_pips": 60,
    "max_open_trades": 3,
    "max_correlated_trades": 2,
}

CONFIDENCE_TIERS = {
    "C":  {"min": 0.0,  "max": 0.55, "action": "NO TRADE",   "size": 0.0},
    "B":  {"min": 0.55, "max": 0.70, "action": "SKIP",       "size": 0.0},
    "A":  {"min": 0.70, "max": 0.85, "action": "STANDARD",   "size": 0.75},
    "A+": {"min": 0.85, "max": 1.00, "action": "FULL + AI",  "size": 1.0},
}


# -- Session Config (forex market hours in UTC) --
SESSIONS = {
    "sydney":  {"open": 21, "close": 6},
    "tokyo":   {"open": 0, "close": 9},
    "london":  {"open": 7, "close": 16},
    "new_york": {"open": 12, "close": 21},
}

CORRELATION_GROUPS = {
    "usd_pairs": ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"],
    "jpy_pairs": ["USD_JPY", "EUR_JPY", "GBP_JPY"],
    "eur_pairs": ["EUR_USD", "EUR_GBP", "EUR_JPY"],
}
