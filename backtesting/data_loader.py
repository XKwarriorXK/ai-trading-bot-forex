"""
Historical data loader — fetches forex data from OANDA for backtesting.
Chunks requests to stay under OANDA's 500-candle API limit.
"""
import logging
import pandas as pd
import oandapyV20.endpoints.instruments as inst_ep
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

CHUNK_DAYS = {
    "M1": 0.3,
    "M5": 1.5,
    "M15": 5,
    "M30": 10,
    "H1": 20,
    "H4": 80,
    "D": 400,
}


def fetch_oanda_historical(instrument: str = "EUR_USD", granularity: str = "H1",
                           days: int = 180) -> pd.DataFrame:
    try:
        from execution.oanda_client import OandaClient
        oanda = OandaClient()
        if not oanda.client:
            logger.warning("OANDA not configured, using empty data")
            return pd.DataFrame()

        all_data = []
        chunk_size = CHUNK_DAYS.get(granularity, 20)
        remaining_days = days
        end_time = datetime.now(timezone.utc)

        while remaining_days > 0:
            fetch_days = min(remaining_days, chunk_size)
            start_time = end_time - timedelta(days=fetch_days)

            params = {
                "granularity": granularity,
                "from": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            r = inst_ep.InstrumentsCandles(instrument=instrument, params=params)
            oanda.client.request(r)
            candles = r.response.get("candles", [])

            for c in candles:
                if c["complete"]:
                    mid = c["mid"]
                    all_data.append({
                        "timestamp": pd.to_datetime(c["time"]),
                        "open": float(mid["o"]),
                        "high": float(mid["h"]),
                        "low": float(mid["l"]),
                        "close": float(mid["c"]),
                        "volume": int(c["volume"]),
                    })

            remaining_days -= fetch_days
            end_time = start_time

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        logger.info(f"Loaded {len(df)} bars for {instrument}")
        return df

    except Exception as e:
        logger.error(f"Failed to load historical data: {e}")
        return pd.DataFrame()
