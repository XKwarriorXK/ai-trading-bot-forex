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
        },
        "base_url": "https://api.groq.com/openai/v1",
    },
    "cerebras": {
        "api_key": os.getenv("CEREBRAS_API_KEY"),
        "models": {
            "fast": "llama-3.3-70b",
            "reasoning": "llama-3.3-70b",
        },
        "base_url": "https://api.cerebras.ai/v1",
    },
}

TASK_ROUTING = {
    "market_analysis":  ["cerebras:fast", "groq:fast", "groq:small"],
    "trade_decision":   ["cerebras:reasoning", "groq:reasoning", "groq:small"],
    "risk_assessment":  ["cerebras:reasoning", "groq:reasoning", "groq:small"],
    "sentiment":        ["cerebras:fast", "groq:fast", "groq:small"],
    "reflection":       ["cerebras:reasoning", "groq:reasoning", "groq:small"],
    "regime_detection": ["cerebras:fast", "groq:fast", "groq:small"],
    "debate":           ["cerebras:fast", "groq:fast", "groq:small"],
    "strategy_select":  ["cerebras:fast", "groq:fast", "groq:small"],
    "macro_analysis":   ["cerebras:fast", "groq:fast", "groq:small"],
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
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "USD_CAD",
    "EUR_GBP",
    "EUR_JPY",
    "GBP_JPY",
    "NZD_USD",
    "USD_CHF",
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
    "skip":       {"min": 0.0, "max": 0.30, "action": "NO TRADE"},
    "flag":       {"min": 0.30, "max": 0.50, "action": "LOG ONLY"},
    "cautious":   {"min": 0.50, "max": 0.65, "action": "SMALL"},
    "standard":   {"min": 0.65, "max": 0.80, "action": "STANDARD"},
    "aggressive": {"min": 0.80, "max": 1.00, "action": "FULL"},
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
