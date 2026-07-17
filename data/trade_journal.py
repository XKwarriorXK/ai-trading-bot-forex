"""
Trade Journal — logs every decision and trade for self-learning.
Stores in SQLite for analysis and pattern recognition.
"""
import logging
import sqlite3
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TradeJournal:
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                instrument TEXT,
                direction TEXT,
                units INTEGER,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                pnl REAL,
                pnl_pips REAL,
                confidence REAL,
                regime TEXT,
                strategies TEXT,
                reasons TEXT,
                session TEXT,
                duration_minutes INTEGER,
                exit_reason TEXT,
                trade_id TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                instrument TEXT,
                signal TEXT,
                confidence REAL,
                regime TEXT,
                reasons TEXT,
                executed INTEGER DEFAULT 0,
                skip_reason TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                trades_taken INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                best_trade REAL DEFAULT 0,
                worst_trade REAL DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

    def log_signal(self, instrument: str, signal: str, confidence: float,
                   regime: str, reasons: list, executed: bool = False,
                   skip_reason: str = None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO signals (timestamp, instrument, signal, confidence,
                                regime, reasons, executed, skip_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            instrument, signal, confidence, regime,
            json.dumps(reasons), 1 if executed else 0, skip_reason,
        ))
        conn.commit()
        conn.close()

    def log_trade_open(self, instrument: str, direction: str, units: int,
                       entry_price: float, stop_loss: float, take_profit: float,
                       confidence: float, regime: str, strategies: list,
                       reasons: list, session: str, trade_id: str = None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO trades (timestamp, instrument, direction, units,
                               entry_price, stop_loss, take_profit,
                               confidence, regime, strategies, reasons,
                               session, trade_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            instrument, direction, units, entry_price,
            stop_loss, take_profit, confidence, regime,
            json.dumps(strategies), json.dumps(reasons),
            session, trade_id,
        ))
        conn.commit()
        last_id = c.lastrowid
        conn.close()
        return last_id

    def log_trade_close(self, trade_db_id: int, exit_price: float, pnl: float,
                        pnl_pips: float, duration_minutes: int, exit_reason: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            UPDATE trades SET exit_price=?, pnl=?, pnl_pips=?,
                             duration_minutes=?, exit_reason=?
            WHERE id=?
        """, (exit_price, pnl, pnl_pips, duration_minutes, exit_reason, trade_db_id))
        conn.commit()
        conn.close()

        self._update_daily_stats(pnl)

    def _update_daily_stats(self, pnl: float):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT * FROM daily_stats WHERE date=?", (today,))
        row = c.fetchone()

        if row:
            trades = row[1] + 1
            wins = row[2] + (1 if pnl > 0 else 0)
            losses = row[3] + (1 if pnl < 0 else 0)
            total_pnl = row[4] + pnl
            best = max(row[6], pnl)
            worst = min(row[7], pnl)
            c.execute("""
                UPDATE daily_stats SET trades_taken=?, wins=?, losses=?,
                                      pnl=?, best_trade=?, worst_trade=?
                WHERE date=?
            """, (trades, wins, losses, total_pnl, best, worst, today))
        else:
            c.execute("""
                INSERT INTO daily_stats (date, trades_taken, wins, losses, pnl,
                                        best_trade, worst_trade)
                VALUES (?, 1, ?, ?, ?, ?, ?)
            """, (today, 1 if pnl > 0 else 0, 1 if pnl < 0 else 0,
                  pnl, pnl, pnl))

        conn.commit()
        conn.close()

    def get_performance_summary(self, days: int = 30) -> dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            SELECT * FROM trades WHERE pnl IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        """, (days * 10,))
        trades = c.fetchall()
        conn.close()

        if not trades:
            return {"total_trades": 0, "message": "No completed trades"}

        pnls = [t[9] for t in trades if t[9] is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            "total_trades": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            "total_pnl": round(sum(pnls), 2),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
            "profit_factor": round(abs(sum(wins) / sum(losses)), 2) if losses and sum(losses) != 0 else 0,
            "best_trade": round(max(pnls), 2) if pnls else 0,
            "worst_trade": round(min(pnls), 2) if pnls else 0,
        }

    def get_instrument_stats(self, instrument: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT pnl FROM trades WHERE instrument=? AND pnl IS NOT NULL
        """, (instrument,))
        rows = c.fetchall()
        conn.close()

        pnls = [r[0] for r in rows]
        if not pnls:
            return {"trades": 0}

        wins = [p for p in pnls if p > 0]
        return {
            "trades": len(pnls),
            "win_rate": round(len(wins) / len(pnls) * 100, 1),
            "total_pnl": round(sum(pnls), 2),
        }
