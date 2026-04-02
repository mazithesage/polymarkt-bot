"""SQLite persistence — orders, positions, market cache, scan log."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

BUSY_TIMEOUT_MS = 5000
CONNECT_TIMEOUT_S = 10.0
CURRENT_SCHEMA_VERSION = 1

# Platform-aware file locking
try:
    import fcntl

    def _lock_exclusive(f):
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(f):
        fcntl.flock(f, fcntl.LOCK_UN)

except ImportError:
    # Windows fallback — msvcrt.locking is advisory but good enough
    try:
        import msvcrt

        def _lock_exclusive(f):
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

        def _unlock(f):
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass

    except ImportError:
        raise RuntimeError(
            "No file-locking module available (need fcntl on Unix or msvcrt on Windows)"
        )


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );

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
        _ensure_schema_version(conn)


def _ensure_schema_version(conn: sqlite3.Connection) -> None:
    """Check schema version and run migrations if needed."""
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)",
                      (CURRENT_SCHEMA_VERSION,))
        return
    db_version = row[0]
    if db_version < CURRENT_SCHEMA_VERSION:
        # Future migrations go here: if db_version < 2: _migrate_v1_to_v2(conn)
        conn.execute("UPDATE schema_version SET version = ?", (CURRENT_SCHEMA_VERSION,))


@contextmanager
def _connect(db_path: str):
    conn = sqlite3.connect(db_path, timeout=CONNECT_TIMEOUT_S)
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class PersistenceStore:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock_file = None
        self._acquire_lock()
        init_db(db_path)

    def _acquire_lock(self) -> None:
        lock_path = self.db_path + ".lock"
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(lock_path, "w")
        try:
            _lock_exclusive(self._lock_file)
        except OSError:
            self._lock_file.close()
            self._lock_file = None
            raise RuntimeError(
                "Another bot instance is already running (could not acquire lock "
                f"on {lock_path})"
            )

    def release_lock(self) -> None:
        if self._lock_file is not None:
            try:
                _unlock(self._lock_file)
                self._lock_file.close()
            except OSError:
                pass
            self._lock_file = None

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

    @staticmethod
    def _compute_vwap(
        old_size: float, old_avg: float, new_size: float, new_avg: float,
    ) -> tuple[float, float]:
        """Compute VWAP after adding a new fill to an existing position.

        Returns (combined_size, combined_avg_price).
        """
        total_size = old_size + new_size
        if total_size <= 0:
            return (total_size, new_avg)
        combined_avg = (old_size * old_avg + new_size * new_avg) / total_size
        return (total_size, combined_avg)

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
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT size, avg_price FROM positions "
                "WHERE condition_id=? AND token_id=? AND paper_mode=?",
                (condition_id, token_id, int(paper_mode)),
            ).fetchone()

            if row is None:
                pnl = (current_price - avg_price) * size if current_price > 0 else 0.0
                conn.execute(
                    """INSERT INTO positions
                       (condition_id, token_id, token_choice, size, avg_price,
                        current_price, pnl, paper_mode, opened_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (condition_id, token_id, token_choice, size, avg_price,
                     current_price, pnl, int(paper_mode), now, now),
                )
            else:
                total_size, vwap = self._compute_vwap(
                    row["size"], row["avg_price"], size, avg_price,
                )
                pnl = (current_price - vwap) * total_size if current_price > 0 else 0.0
                conn.execute(
                    """UPDATE positions
                       SET size=?, avg_price=?, current_price=?, pnl=?, updated_at=?
                       WHERE condition_id=? AND token_id=? AND paper_mode=?""",
                    (total_size, vwap, current_price, pnl, now,
                     condition_id, token_id, int(paper_mode)),
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
                "SELECT COALESCE(SUM(size * avg_price), 0) as total "
                "FROM positions WHERE size > 0 AND paper_mode=?",
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
