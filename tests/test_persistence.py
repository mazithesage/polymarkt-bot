"""Tests for SQLite persistence module."""

import sqlite3
import time

import pytest

from persistence import PersistenceStore, init_db


class TestInitDB:
    def test_creates_database(self, db_path):
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "orders" in table_names
        assert "positions" in table_names
        assert "market_cache" in table_names
        assert "scan_log" in table_names
        assert "redemptions" in table_names
        conn.close()

    def test_idempotent_init(self, db_path):
        init_db(db_path)
        init_db(db_path)  # Should not raise

    def test_wal_mode_enabled(self, db_path):
        store = PersistenceStore(db_path)
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        store.release_lock()
        assert mode == "wal"

    def test_busy_timeout_set(self, db_path):
        store = PersistenceStore(db_path)
        conn = sqlite3.connect(db_path)
        # busy_timeout is set per-connection in _connect, verify via a fresh _connect
        from persistence import _connect
        with _connect(db_path) as c:
            timeout = c.execute("PRAGMA busy_timeout").fetchone()[0]
        store.release_lock()
        conn.close()
        assert timeout == 5000


class TestInstanceLock:
    def test_second_instance_raises(self, db_path):
        store1 = PersistenceStore(db_path)
        with pytest.raises(RuntimeError, match="Another bot instance"):
            PersistenceStore(db_path)
        store1.release_lock()

    def test_release_allows_reacquisition(self, db_path):
        store1 = PersistenceStore(db_path)
        store1.release_lock()
        store2 = PersistenceStore(db_path)  # Should succeed
        store2.release_lock()


class TestOrderPersistence:
    def test_log_and_retrieve_order(self, db_path):
        store = PersistenceStore(db_path)
        store.log_order(
            order_id="order-123",
            condition_id="cond-1",
            token_id="token-1",
            side="BUY",
            price=0.55,
            size=10.0,
            status="PLACED",
            paper_mode=True,
        )
        orders = store.get_recent_orders(limit=10, paper_mode=True)
        assert len(orders) == 1
        assert orders[0]["order_id"] == "order-123"
        assert orders[0]["price"] == 0.55
        assert orders[0]["paper_mode"] == 1

    def test_update_order_status(self, db_path):
        store = PersistenceStore(db_path)
        store.log_order("o1", "c1", "t1", "BUY", 0.5, 10.0, status="PLACED")
        store.update_order_status("o1", "FILLED")
        orders = store.get_recent_orders(limit=10, paper_mode=False)
        assert orders[0]["status"] == "FILLED"

    def test_multiple_orders(self, db_path):
        store = PersistenceStore(db_path)
        for i in range(5):
            store.log_order(f"o{i}", "c1", "t1", "BUY", 0.5 + i * 0.01, 10.0)
        orders = store.get_recent_orders(limit=10)
        assert len(orders) == 5

    def test_order_limit(self, db_path):
        store = PersistenceStore(db_path)
        for i in range(10):
            store.log_order(f"o{i}", "c1", "t1", "BUY", 0.5, 10.0)
        orders = store.get_recent_orders(limit=3)
        assert len(orders) == 3


class TestPositionPersistence:
    def test_upsert_new_position(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.55, paper_mode=True)
        positions = store.get_open_positions(paper_mode=True)
        assert len(positions) == 1
        assert positions[0]["size"] == 10.0
        assert positions[0]["avg_price"] == 0.55

    def test_upsert_updates_existing(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.55, paper_mode=True)
        store.upsert_position("c1", "t1", "YES", 20.0, 0.60, paper_mode=True)
        positions = store.get_open_positions(paper_mode=True)
        assert len(positions) == 1
        assert positions[0]["size"] == 20.0
        assert positions[0]["avg_price"] == 0.60

    def test_total_exposure(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.50, paper_mode=True)
        store.upsert_position("c2", "t2", "NO", 20.0, 0.40, paper_mode=True)
        exposure = store.get_total_exposure(paper_mode=True)
        expected = 10.0 * 0.50 + 20.0 * 0.40  # 5 + 8 = 13
        assert exposure == pytest.approx(expected, abs=0.01)

    def test_exposure_separate_paper_live(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.50, paper_mode=True)
        store.upsert_position("c2", "t2", "NO", 20.0, 0.40, paper_mode=False)
        assert store.get_total_exposure(paper_mode=True) == pytest.approx(5.0)
        assert store.get_total_exposure(paper_mode=False) == pytest.approx(8.0)

    def test_pnl_calculation(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.50, current_price=0.70)
        positions = store.get_open_positions(paper_mode=False)
        assert positions[0]["pnl"] == pytest.approx(2.0)  # (0.70-0.50)*10


class TestScanLog:
    def test_log_scan(self, db_path):
        store = PersistenceStore(db_path)
        store.log_scan(markets_found=50, markets_with_edge=3, orders_placed=2)
        history = store.get_scan_history(limit=5)
        assert len(history) == 1
        assert history[0]["markets_found"] == 50
        assert history[0]["orders_placed"] == 2


class TestMarketCache:
    def test_cache_and_retrieve(self, db_path):
        store = PersistenceStore(db_path)
        store.cache_market("c1", "Will BTC?", "crypto", {"volume": 1000})
        cached = store.get_cached_market("c1")
        assert cached is not None
        assert cached["question"] == "Will BTC?"
        assert cached["data"]["volume"] == 1000

    def test_cache_expires(self, db_path):
        store = PersistenceStore(db_path)
        store.cache_market("c1", "Test", "other", {})
        # Retrieve with 0 max_age should return None
        cached = store.get_cached_market("c1", max_age=0)
        assert cached is None

    def test_cache_miss(self, db_path):
        store = PersistenceStore(db_path)
        assert store.get_cached_market("nonexistent") is None


class TestRedemptionLog:
    def test_log_redemption(self, db_path):
        store = PersistenceStore(db_path)
        store.log_redemption("c1", "t1", 100.0, "0xabc123", "SUCCESS")
        # Verify by checking the database directly
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM redemptions").fetchall()
        assert len(rows) == 1
        assert dict(rows[0])["tx_hash"] == "0xabc123"
        conn.close()
