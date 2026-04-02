import sqlite3
import time

import pytest
from persistence import PersistenceStore, init_db, CURRENT_SCHEMA_VERSION


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
        assert "schema_version" in table_names
        assert "ob_snapshots" in table_names
        conn.close()

    def test_idempotent_init(self, db_path):
        init_db(db_path)
        init_db(db_path)  # should not raise

    def test_wal_mode_enabled(self, db_path):
        store = PersistenceStore(db_path)
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        store.release_lock()
        assert mode == "wal"

    def test_busy_timeout_set(self, db_path):
        store = PersistenceStore(db_path)
        from persistence import _connect
        with _connect(db_path) as c:
            timeout = c.execute("PRAGMA busy_timeout").fetchone()[0]
        store.release_lock()
        assert timeout == 5000

    def test_schema_version_recorded(self, db_path):
        store = PersistenceStore(db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        conn.close()
        store.release_lock()
        assert row[0] == CURRENT_SCHEMA_VERSION


class TestInstanceLock:
    def test_second_instance_raises(self, db_path):
        store1 = PersistenceStore(db_path)
        with pytest.raises(RuntimeError, match="Another bot instance"):
            PersistenceStore(db_path)
        store1.release_lock()

    def test_release_allows_reacquisition(self, db_path):
        store1 = PersistenceStore(db_path)
        store1.release_lock()
        store2 = PersistenceStore(db_path)
        store2.release_lock()


class TestOrderPersistence:
    def test_log_and_retrieve_order(self, db_path):
        store = PersistenceStore(db_path)
        store.log_order(
            order_id="order-123", condition_id="cond-1", token_id="token-1",
            side="BUY", price=0.55, size=10.0, status="PLACED", paper_mode=True,
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
        assert len(store.get_recent_orders(limit=10)) == 5

    def test_order_limit(self, db_path):
        store = PersistenceStore(db_path)
        for i in range(10):
            store.log_order(f"o{i}", "c1", "t1", "BUY", 0.5, 10.0)
        assert len(store.get_recent_orders(limit=3)) == 3


class TestPositionPersistence:
    def test_upsert_new_position(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.55, paper_mode=True)
        positions = store.get_open_positions(paper_mode=True)
        assert len(positions) == 1
        assert positions[0]["size"] == 10.0

    def test_upsert_accumulates_with_weighted_avg_price(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.55, paper_mode=True)
        store.upsert_position("c1", "t1", "YES", 20.0, 0.60, paper_mode=True)
        positions = store.get_open_positions(paper_mode=True)
        assert len(positions) == 1
        # Size accumulates: 10 + 20 = 30
        assert positions[0]["size"] == pytest.approx(30.0)
        # VWAP: (10 * 0.55 + 20 * 0.60) / 30 = 17.5 / 30 ≈ 0.5833
        expected_avg = (10.0 * 0.55 + 20.0 * 0.60) / 30.0
        assert positions[0]["avg_price"] == pytest.approx(expected_avg, abs=0.001)

    def test_total_exposure(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.50, paper_mode=True)
        store.upsert_position("c2", "t2", "NO", 20.0, 0.40, paper_mode=True)
        expected = 10.0 * 0.50 + 20.0 * 0.40
        assert store.get_total_exposure(paper_mode=True) == pytest.approx(expected, abs=0.01)

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


class TestVWAPPositionAccumulation:
    """Verify the weighted average cost basis logic in position upserts."""

    def test_three_fills_accumulate_correctly(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.50, paper_mode=True)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.60, paper_mode=True)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.70, paper_mode=True)
        pos = store.get_open_positions(paper_mode=True)
        assert len(pos) == 1
        assert pos[0]["size"] == pytest.approx(30.0)
        # VWAP: (10*0.50 + 10*0.60 + 10*0.70) / 30 = 18/30 = 0.60
        assert pos[0]["avg_price"] == pytest.approx(0.60, abs=0.001)

    def test_pnl_uses_accumulated_avg_price(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.40, paper_mode=False)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.60, current_price=0.70, paper_mode=False)
        pos = store.get_open_positions(paper_mode=False)
        # size = 20, avg = (10*0.40 + 10*0.60) / 20 = 0.50
        # pnl = (0.70 - 0.50) * 20 = 4.0
        assert pos[0]["avg_price"] == pytest.approx(0.50, abs=0.001)
        assert pos[0]["pnl"] == pytest.approx(4.0, abs=0.01)

    def test_different_conditions_stay_separate(self, db_path):
        store = PersistenceStore(db_path)
        store.upsert_position("c1", "t1", "YES", 10.0, 0.50, paper_mode=True)
        store.upsert_position("c2", "t2", "NO", 20.0, 0.40, paper_mode=True)
        positions = store.get_open_positions(paper_mode=True)
        assert len(positions) == 2


class TestVWAPEdgeCases:
    """Edge cases for _compute_vwap static method."""

    def test_zero_total_size_returns_zero_avg(self):
        """When position fully closed (total_size=0), avg should be 0.0."""
        total, avg = PersistenceStore._compute_vwap(10.0, 0.50, -10.0, 0.60)
        assert total == 0.0
        assert avg == 0.0

    def test_negative_total_size_returns_zero(self):
        """Oversell (total_size < 0) should return (0.0, 0.0)."""
        total, avg = PersistenceStore._compute_vwap(5.0, 0.50, -10.0, 0.60)
        assert total == 0.0
        assert avg == 0.0

    def test_normal_accumulation(self):
        total, avg = PersistenceStore._compute_vwap(10.0, 0.50, 10.0, 0.60)
        assert total == 20.0
        assert avg == pytest.approx(0.55)

    def test_first_fill(self):
        total, avg = PersistenceStore._compute_vwap(0.0, 0.0, 10.0, 0.55)
        assert total == 10.0
        assert avg == pytest.approx(0.55)


class TestScanLog:
    def test_log_scan(self, db_path):
        store = PersistenceStore(db_path)
        store.log_scan(markets_found=50, markets_with_edge=3, orders_placed=2)
        history = store.get_scan_history(limit=5)
        assert len(history) == 1
        assert history[0]["markets_found"] == 50


class TestMarketCache:
    def test_cache_and_retrieve(self, db_path):
        store = PersistenceStore(db_path)
        store.cache_market("c1", "Will BTC?", "crypto", {"volume": 1000})
        cached = store.get_cached_market("c1")
        assert cached is not None
        assert cached["data"]["volume"] == 1000

    def test_cache_expires(self, db_path):
        store = PersistenceStore(db_path)
        store.cache_market("c1", "Test", "other", {})
        assert store.get_cached_market("c1", max_age=0) is None

    def test_cache_miss(self, db_path):
        store = PersistenceStore(db_path)
        assert store.get_cached_market("nonexistent") is None


class TestRedemptionLog:
    def test_log_redemption(self, db_path):
        store = PersistenceStore(db_path)
        store.log_redemption("c1", "t1", 100.0, "0xabc123", "SUCCESS")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM redemptions").fetchall()
        assert len(rows) == 1
        assert dict(rows[0])["tx_hash"] == "0xabc123"
        conn.close()


class TestOBSnapshots:
    def test_log_and_read_snapshot(self, db_path):
        store = PersistenceStore(db_path)
        store.log_ob_snapshot(
            condition_id="cond-1", token_id="tok-1",
            best_bid=0.48, best_ask=0.52, mid_price=0.50,
            spread=0.04, bid_depth=1000.0, ask_depth=800.0,
            imbalance=0.111, estimated_prob=0.506,
            market_price=0.50, had_edge=True, paper_mode=True,
        )
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM ob_snapshots").fetchall()
        conn.close()
        store.release_lock()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["condition_id"] == "cond-1"
        assert row["token_id"] == "tok-1"
        assert row["best_bid"] == pytest.approx(0.48)
        assert row["best_ask"] == pytest.approx(0.52)
        assert row["mid_price"] == pytest.approx(0.50)
        assert row["imbalance"] == pytest.approx(0.111)
        assert row["estimated_prob"] == pytest.approx(0.506)
        assert row["had_edge"] == 1
        assert row["paper_mode"] == 1
        assert row["snapshot_time"] > 0

    def test_multiple_snapshots(self, db_path):
        store = PersistenceStore(db_path)
        for i in range(5):
            store.log_ob_snapshot(
                condition_id=f"cond-{i}", token_id=f"tok-{i}",
                best_bid=0.45, best_ask=0.55, mid_price=0.50,
                spread=0.10, bid_depth=500.0, ask_depth=500.0,
                imbalance=0.0, estimated_prob=0.50,
                market_price=0.50,
            )
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM ob_snapshots").fetchone()[0]
        conn.close()
        store.release_lock()
        assert count == 5

    def test_indexes_exist(self, db_path):
        store = PersistenceStore(db_path)
        conn = sqlite3.connect(db_path)
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = {idx[0] for idx in indexes}
        conn.close()
        store.release_lock()
        assert "idx_ob_condition" in index_names
        assert "idx_ob_time" in index_names


class TestSchemaMigrationV1ToV2:
    def _create_v1_database(self, db_path):
        """Create a v1 database without ob_snapshots table."""
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (1);

            CREATE TABLE orders (
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

            CREATE TABLE positions (
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

            CREATE TABLE market_cache (
                condition_id TEXT PRIMARY KEY,
                question TEXT,
                category TEXT,
                data TEXT,
                cached_at REAL
            );

            CREATE TABLE scan_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_time REAL,
                markets_found INTEGER,
                markets_with_edge INTEGER,
                orders_placed INTEGER,
                paper_mode INTEGER DEFAULT 0
            );

            CREATE TABLE redemptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id TEXT,
                token_id TEXT,
                amount REAL,
                tx_hash TEXT,
                status TEXT,
                created_at REAL
            );

            CREATE INDEX idx_orders_condition ON orders(condition_id);
            CREATE INDEX idx_orders_status ON orders(status);
            CREATE INDEX idx_positions_condition ON positions(condition_id);
        """)
        conn.close()

    def test_migration_creates_ob_snapshots(self, db_path):
        self._create_v1_database(db_path)
        # Verify ob_snapshots does NOT exist yet
        conn = sqlite3.connect(db_path)
        tables = {t[0] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "ob_snapshots" not in tables
        conn.close()

        # Run init_db which should trigger v1→v2 migration
        init_db(db_path)

        conn = sqlite3.connect(db_path)
        tables = {t[0] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "ob_snapshots" in tables
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 2
        conn.close()

    def test_migration_preserves_existing_data(self, db_path):
        self._create_v1_database(db_path)
        # Insert some v1 data
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO orders (order_id, condition_id, token_id, side, price, size) "
            "VALUES ('o1', 'c1', 't1', 'BUY', 0.5, 10.0)"
        )
        conn.commit()
        conn.close()

        # Migrate
        init_db(db_path)

        # Verify old data still exists
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        orders = conn.execute("SELECT * FROM orders").fetchall()
        assert len(orders) == 1
        assert dict(orders[0])["order_id"] == "o1"
        conn.close()
