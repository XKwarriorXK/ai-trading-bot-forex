"""
OANDA REST API client — market data and order execution for forex.
Uses oandapyV20 SDK for v20 API.
"""
import logging
import pandas as pd
from datetime import datetime, timezone
from config.settings import OANDA, INSTRUMENTS

logger = logging.getLogger(__name__)

try:
    import oandapyV20
    import oandapyV20.endpoints.instruments as instruments_ep
    import oandapyV20.endpoints.orders as orders_ep
    import oandapyV20.endpoints.trades as trades_ep
    import oandapyV20.endpoints.accounts as accounts_ep
    import oandapyV20.endpoints.pricing as pricing_ep
    HAS_OANDA = True
except ImportError:
    HAS_OANDA = False
    logger.warning("oandapyV20 not installed — OANDA client unavailable")


class OandaClient:
    def __init__(self):
        if not HAS_OANDA or not OANDA["api_key"]:
            self.client = None
            self.account_id = None
            logger.warning("OANDA client not configured")
            return

        env = "practice" if OANDA["environment"] == "practice" else "live"
        self.client = oandapyV20.API(
            access_token=OANDA["api_key"],
            environment=env,
        )
        self.account_id = OANDA["account_id"]
        logger.info(f"Connected to OANDA ({env})")

    def fetch_candles(self, instrument: str = "EUR_USD", granularity: str = "H1",
                      count: int = 300) -> pd.DataFrame:
        if not self.client:
            return pd.DataFrame()
        try:
            params = {"granularity": granularity, "count": count}
            r = instruments_ep.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            candles = r.response.get("candles", [])

            rows = []
            for c in candles:
                if c["complete"]:
                    mid = c["mid"]
                    rows.append({
                        "timestamp": pd.to_datetime(c["time"]),
                        "open": float(mid["o"]),
                        "high": float(mid["h"]),
                        "low": float(mid["l"]),
                        "close": float(mid["c"]),
                        "volume": int(c["volume"]),
                    })

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch candles for {instrument}: {e}")
            return pd.DataFrame()

    def fetch_price(self, instrument: str = "EUR_USD") -> dict:
        if not self.client:
            return {}
        try:
            params = {"instruments": instrument}
            r = pricing_ep.PricingInfo(accountID=self.account_id, params=params)
            self.client.request(r)
            prices = r.response.get("prices", [])
            if prices:
                p = prices[0]
                return {
                    "instrument": instrument,
                    "bid": float(p["bids"][0]["price"]),
                    "ask": float(p["asks"][0]["price"]),
                    "spread": float(p["asks"][0]["price"]) - float(p["bids"][0]["price"]),
                    "time": p["time"],
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to fetch price for {instrument}: {e}")
            return {}

    def get_account(self) -> dict:
        if not self.client:
            return {"balance": 0, "unrealized_pl": 0, "open_trades": 0}
        try:
            r = accounts_ep.AccountDetails(accountID=self.account_id)
            self.client.request(r)
            acct = r.response.get("account", {})
            return {
                "balance": float(acct.get("balance", 0)),
                "unrealized_pl": float(acct.get("unrealizedPL", 0)),
                "nav": float(acct.get("NAV", 0)),
                "margin_used": float(acct.get("marginUsed", 0)),
                "margin_available": float(acct.get("marginAvailable", 0)),
                "open_trade_count": int(acct.get("openTradeCount", 0)),
                "open_position_count": int(acct.get("openPositionCount", 0)),
            }
        except Exception as e:
            logger.error(f"Failed to fetch account: {e}")
            return {"balance": 0, "unrealized_pl": 0, "open_trades": 0}

    def place_market_order(self, instrument: str, units: int,
                           stop_loss_price: float = None,
                           take_profit_price: float = None) -> dict:
        if not self.client:
            return {"success": False, "error": "Client not configured"}
        try:
            order_body = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(units),
                    "timeInForce": "FOK",
                }
            }
            if stop_loss_price is not None:
                order_body["order"]["stopLossOnFill"] = {
                    "price": f"{stop_loss_price:.5f}"
                }
            if take_profit_price is not None:
                order_body["order"]["takeProfitOnFill"] = {
                    "price": f"{take_profit_price:.5f}"
                }

            r = orders_ep.OrderCreate(accountID=self.account_id, data=order_body)
            self.client.request(r)
            response = r.response

            if "orderFillTransaction" in response:
                fill = response["orderFillTransaction"]
                logger.info(f"Order filled: {instrument} {units} units @ {fill.get('price')}")
                return {
                    "success": True,
                    "trade_id": fill.get("tradeOpened", {}).get("tradeID"),
                    "price": float(fill.get("price", 0)),
                    "units": int(fill.get("units", 0)),
                }
            else:
                reason = response.get("orderCancelTransaction", {}).get("reason", "Unknown")
                logger.warning(f"Order rejected: {reason}")
                return {"success": False, "error": reason}
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {"success": False, "error": str(e)}

    def close_trade(self, trade_id: str) -> dict:
        if not self.client:
            return {"success": False, "error": "Client not configured"}
        try:
            r = trades_ep.TradeClose(accountID=self.account_id, tradeID=trade_id)
            self.client.request(r)
            response = r.response
            if "orderFillTransaction" in response:
                fill = response["orderFillTransaction"]
                pl = float(fill.get("pl", 0))
                logger.info(f"Trade {trade_id} closed: P&L = {pl}")
                return {"success": True, "pl": pl, "price": float(fill.get("price", 0))}
            return {"success": False, "error": "Close failed"}
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return {"success": False, "error": str(e)}

    def get_open_trades(self) -> list:
        if not self.client:
            return []
        try:
            r = trades_ep.OpenTrades(accountID=self.account_id)
            self.client.request(r)
            trades = r.response.get("trades", [])
            return [{
                "trade_id": t["id"],
                "instrument": t["instrument"],
                "units": int(t["currentUnits"]),
                "price": float(t["price"]),
                "unrealized_pl": float(t.get("unrealizedPL", 0)),
                "open_time": t["openTime"],
            } for t in trades]
        except Exception as e:
            logger.error(f"Failed to fetch open trades: {e}")
            return []

    def health_check(self) -> dict:
        if not self.client:
            return {"connected": False, "error": "Not configured"}
        try:
            acct = self.get_account()
            return {
                "connected": True,
                "environment": OANDA["environment"],
                "balance": acct.get("balance", 0),
                "open_trades": acct.get("open_trade_count", 0),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}
