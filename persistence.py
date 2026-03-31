"""
SQLite persistence module for bot state, orders, and positions.
Sourced from: perpetual-s/polymarket-python-infrastructure (multi-wallet state),
              Jonmaa/btc-polymarket-bot (trade logging),
              demone456/kalshi-polymarket-bot (position tracking).

Handles: order history, position tracking, market metadata caching,
         and paper mode trade logging.
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def init_db(db_path: str) -> None:
    """Initialize the SQLite database with required tables."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,
                condition_id TEXT,
                token_id TEXT,
                side TEXT,
                price REAL,
                size REAL,
                order_type TEXT,
                status TEXT,
                paper_mode INTEGER DEFAULT 0,
                created_at REAL,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id TEXT,
                token_id TEXT,
                token_choice TEXT,
                size REAL,
                avg_price REAL,
                current_price REAL,
                pnl REAL DEFAULT 0.0,
                paper_mode INTEGER DEFAULT 0,
                opened_at REAL,
                updated_at REAL,
                UNIQUE(condition_id, token_id, paper_mode)
            );

            CREATE TABLE IF NOT EXISTS market_cache (
                condition_id TEXT PRIMARY KEY,
                question TEXT,
                category TEXT,
                data TEXT,
                cached_at REAL
            );

            CREATE TABLE IF NOT EXISTS scan_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_time REAL,
                markets_found INTEGER,
                markets_with_edge INTEGER,
                orders_placed INTEGER,
                paper_mode INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS redemptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id TEXT,
                token_id TEXT,
                amount REAL,
                tx_hash TEXT,
                status TEXT,
                created_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_orders_condition ON orders(condition_id);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS idx_positions_condition ON positions(condition_id);
        """)


@contextmanager
def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class PersistenceStore:
    """SQLite-backed persistence for the bot."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        init_db(db_path)

    def log_order(
        self,
        order_id: str,
        condition_id: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
        status: str = "PLACED",
        paper_mode: bool = False,
    ) -> None:
        now = time.time()
        with _connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO orders
                   (order_id, condition_id, token_id, side, price, size,
                    order_type, status, paper_mode, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (order_id, condition_id, token_id, side, price, size,
                 order_type, status, int(paper_mode), now, now),
            )

    def update_order_status(self, order_id: str, status: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                "UPDATE orders SET status=?, updated_at=? WHERE order_id=?",
                (status, time.time(), order_id),
            )

    def upsert_position(
        self,
        condition_id: str,
        token_id: str,
        token_choice: str,
        size: float,
        avg_price: float,
        current_price: float = 0.0,
        paper_mode: bool = False,
    ) -> None:
        now = time.time()
        pnl = (current_price - avg_price) * size if current_price > 0 else 0.0
        with _connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO positions
                   (condition_id, token_id, token_choice, size, avg_price,
                    current_price, pnl, paper_mode, opened_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(condition_id, token_id, paper_mode)
                   DO UPDATE SET
                     size=excluded.size,
                     avg_price=excluded.avg_price,
                     current_price=excluded.current_price,
                     pnl=excluded.pnl,
                     updated_at=excluded.updated_at""",
                (condition_id, token_id, token_choice, size, avg_price,
                 current_price, pnl, int(paper_mode), now, now),
            )

    def get_open_positions(self, paper_mode: bool = False) -> List[Dict]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE size > 0 AND paper_mode=?",
                (int(paper_mode),),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_total_exposure(self, paper_mode: bool = False) -> float:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(size * avg_price), 0) as total FROM positions WHERE size > 0 AND paper_mode=?",
                (int(paper_mode),),
            ).fetchone()
            return float(row["total"])

    def log_scan(
        self,
        markets_found: int,
        markets_with_edge: int,
        orders_placed: int,
        paper_mode: bool = False,
    ) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO scan_log
                   (scan_time, markets_found, markets_with_edge, orders_placed, paper_mode)
                   VALUES (?, ?, ?, ?, ?)""",
                (time.time(), markets_found, markets_with_edge, orders_placed, int(paper_mode)),
            )

    def log_redemption(
        self,
        condition_id: str,
        token_id: str,
        amount: float,
        tx_hash: str,
        status: str = "PENDING",
    ) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO redemptions
                   (condition_id, token_id, amount, tx_hash, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (condition_id, token_id, amount, tx_hash, status, time.time()),
            )

    def cache_market(self, condition_id: str, question: str, category: str, data: dict) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO market_cache
                   (condition_id, question, category, data, cached_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (condition_id, question, category, json.dumps(data), time.time()),
            )

    def get_cached_market(self, condition_id: str, max_age: float = 3600) -> Optional[Dict]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM market_cache WHERE condition_id=? AND cached_at > ?",
                (condition_id, time.time() - max_age),
            ).fetchone()
            if row:
                result = dict(row)
                result["data"] = json.loads(result["data"])
                return result
            return None

    def get_recent_orders(self, limit: int = 50, paper_mode: bool = False) -> List[Dict]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE paper_mode=? ORDER BY created_at DESC LIMIT ?",
                (int(paper_mode), limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_scan_history(self, limit: int = 20) -> List[Dict]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM scan_log ORDER BY scan_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
