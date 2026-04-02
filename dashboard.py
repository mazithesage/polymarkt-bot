"""Web dashboard for the Polymarket trading bot.

Run:
    python dashboard.py --db data/bot.db --port 8050
"""

import argparse
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BUSY_TIMEOUT_MS = 5000
CONNECT_TIMEOUT_S = 10.0

app = FastAPI(title="Polymarket Bot Dashboard")

# Resolved at startup via configure()
DB_PATH: str = "data/bot.db"
PAPER_MODE: int = 1


@contextmanager
def _db():
    """Read-only DB connection (no locking, WAL-safe)."""
    conn = sqlite3.connect(DB_PATH, timeout=CONNECT_TIMEOUT_S)
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _rows_to_dicts(rows):
    return [dict(r) for r in rows]


# ── HTML ──────────────────────────────────────────────────────────────

@app.get("/")
def index():
    html = Path(__file__).parent / "static" / "dashboard.html"
    return FileResponse(html, media_type="text/html")


# ── API endpoints ─────────────────────────────────────────────────────

@app.get("/api/summary")
def api_summary(paper_mode: int = Query(default=None)):
    pm = paper_mode if paper_mode is not None else PAPER_MODE
    with _db() as conn:
        pos = conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl), 0) as total_pnl, "
            "COALESCE(SUM(size * avg_price), 0) as exposure "
            "FROM positions WHERE size > 0 AND paper_mode=?",
            (pm,),
        ).fetchone()

        profitable = conn.execute(
            "SELECT COUNT(*) as cnt FROM positions WHERE pnl > 0 AND paper_mode=?",
            (pm,),
        ).fetchone()
        total_positions = conn.execute(
            "SELECT COUNT(*) as cnt FROM positions WHERE paper_mode=?",
            (pm,),
        ).fetchone()

        last_scan = conn.execute(
            "SELECT * FROM scan_log WHERE paper_mode=? ORDER BY scan_time DESC LIMIT 1",
            (pm,),
        ).fetchone()

    win_rate = 0.0
    if total_positions and total_positions["cnt"] > 0:
        win_rate = round(profitable["cnt"] / total_positions["cnt"] * 100, 1)

    return {
        "total_pnl": round(pos["total_pnl"], 4),
        "total_exposure": round(pos["exposure"], 2),
        "open_positions": pos["cnt"],
        "win_rate": win_rate,
        "last_scan": dict(last_scan) if last_scan else None,
        "paper_mode": pm,
    }


@app.get("/api/positions")
def api_positions(paper_mode: int = Query(default=None)):
    pm = paper_mode if paper_mode is not None else PAPER_MODE
    with _db() as conn:
        rows = conn.execute(
            "SELECT p.*, mc.question, mc.category "
            "FROM positions p "
            "LEFT JOIN market_cache mc ON p.condition_id = mc.condition_id "
            "WHERE p.size > 0 AND p.paper_mode=? "
            "ORDER BY p.pnl DESC",
            (pm,),
        ).fetchall()
    return _rows_to_dicts(rows)


@app.get("/api/orders")
def api_orders(limit: int = Query(default=50, le=500), paper_mode: int = Query(default=None)):
    pm = paper_mode if paper_mode is not None else PAPER_MODE
    with _db() as conn:
        rows = conn.execute(
            "SELECT o.*, mc.question "
            "FROM orders o "
            "LEFT JOIN market_cache mc ON o.condition_id = mc.condition_id "
            "WHERE o.paper_mode=? "
            "ORDER BY o.created_at DESC LIMIT ?",
            (pm, limit),
        ).fetchall()
    return _rows_to_dicts(rows)


@app.get("/api/scans")
def api_scans(limit: int = Query(default=100, le=1000), paper_mode: int = Query(default=None)):
    pm = paper_mode if paper_mode is not None else PAPER_MODE
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM scan_log WHERE paper_mode=? ORDER BY scan_time DESC LIMIT ?",
            (pm, limit),
        ).fetchall()
    return _rows_to_dicts(rows)


@app.get("/api/ob-snapshots")
def api_ob_snapshots(
    condition_id: str = Query(...),
    limit: int = Query(default=200, le=1000),
    paper_mode: int = Query(default=None),
):
    pm = paper_mode if paper_mode is not None else PAPER_MODE
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM ob_snapshots "
            "WHERE condition_id=? AND paper_mode=? "
            "ORDER BY snapshot_time DESC LIMIT ?",
            (condition_id, pm, limit),
        ).fetchall()
    return _rows_to_dicts(rows)


@app.get("/api/ob-snapshots/latest")
def api_ob_snapshots_latest(
    limit: int = Query(default=50, le=500),
    paper_mode: int = Query(default=None),
):
    pm = paper_mode if paper_mode is not None else PAPER_MODE
    with _db() as conn:
        rows = conn.execute(
            "SELECT obs.*, mc.question "
            "FROM ob_snapshots obs "
            "LEFT JOIN market_cache mc ON obs.condition_id = mc.condition_id "
            "WHERE obs.paper_mode=? "
            "ORDER BY obs.snapshot_time DESC LIMIT ?",
            (pm, limit),
        ).fetchall()
    return _rows_to_dicts(rows)


@app.get("/api/config")
def api_config():
    """Return non-secret bot config for display."""
    try:
        from config import load_config
        _clob, _chain, bot = load_config()
        return {
            "paper_mode": bot.paper_mode,
            "scan_interval": bot.scan_interval,
            "max_position_usdc": bot.max_position_usdc,
            "kelly_fraction": bot.kelly_fraction,
            "min_edge": bot.min_edge,
            "max_markets": bot.max_markets,
            "order_type": bot.order_type,
            "db_path": bot.db_path,
            "ssvi_r2_threshold": bot.ssvi_r2_threshold,
            "slippage_spread_bps": bot.slippage_spread_bps,
            "slippage_impact_bps": bot.slippage_impact_bps,
            "min_tradeable_price": bot.min_tradeable_price,
            "max_tradeable_price": bot.max_tradeable_price,
            "max_spread": bot.max_spread,
            "min_liquidity": bot.min_liquidity,
            "bankroll_multiplier": bot.bankroll_multiplier,
            "ob_imbalance_weight": bot.ob_imbalance_weight,
        }
    except Exception as e:
        return {"error": str(e)}


def configure(db_path: str, paper_mode: bool = True) -> None:
    global DB_PATH, PAPER_MODE
    DB_PATH = db_path
    PAPER_MODE = int(paper_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Bot Dashboard")
    parser.add_argument("--db", default=os.getenv("DB_PATH", "data/bot.db"), help="SQLite DB path")
    parser.add_argument("--port", type=int, default=int(os.getenv("DASHBOARD_PORT", "8050")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--paper-mode", type=int, default=1, help="1=paper, 0=live")
    args = parser.parse_args()

    configure(args.db, paper_mode=bool(args.paper_mode))

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
