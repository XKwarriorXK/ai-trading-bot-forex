"""
Historical data loader — fetches forex data from OANDA for backtesting.
"""
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def fetch_oanda_historical(instrument: str = "EUR_USD", granularity: str = "H1",
                           days: int = 180) -> pd.DataFrame:
    try:
        from execution.oanda_client import OandaClient
        oanda = OandaClient()
        if not oanda.client:
            logger.warning("OANDA not configured, using empty data")
            return pd.DataFrame()

        all_data = []
        max_candles = 5000
        remaining_days = days
        end_time = datetime.now(timezone.utc)

        while remaining_days > 0:
            chunk_days = min(remaining_days, 180)
            start_time = end_time - timedelta(days=chunk_days)

            import oandapyV20.endpoints.instruments as inst_ep
            params = {
                "granularity": granularity,
                "from": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "count": max_candles,
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

            remaining_days -= chunk_days
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
