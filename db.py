from __future__ import annotations

import os
import sqlite3
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import pandas as pd

DB_PATH_DEFAULT = "data/live.db"


def connect_db(db_path: str = DB_PATH_DEFAULT) -> sqlite3.Connection:
    """
    Connect to SQLite with WAL mode enabled for concurrency.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    # Enable Write-Ahead Logging for better concurrency
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    # Enforce foreign keys if we use them (optional, but good practice)
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def init_db(con: sqlite3.Connection) -> None:
    """
    Initialize database schema (idempotent).
    """
    with con:
        # 1. Heartbeat
        # Tracks status of one or more engines (e.g. US, JP)
        con.execute("""
            CREATE TABLE IF NOT EXISTS engine_heartbeat (
                engine_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                meta_json TEXT
            );
        """)

        # 2. Live Metrics Snapshot
        # Latest calculated metrics (Z-score, betas) for the dashboard.
        # PK is just 'pair' -> one row per pair.
        con.execute("""
            CREATE TABLE IF NOT EXISTS live_metrics_snapshot (
                pair TEXT PRIMARY KEY,
                time TEXT NOT NULL,
                z REAL,
                direct_beta REAL,
                beta_30_weekly REAL,
                beta_drift_pct REAL,
                last_updated TEXT
            );
        """)

        # 3. Pair Series (History)
        # Detailed history for charts.
        # UNIQUE constraint ensures no duplicate candles for the same timeframe.
        con.execute("""
            CREATE TABLE IF NOT EXISTS pair_series (
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL, -- e.g. '15m'
                time TEXT NOT NULL,      -- Bucket start time (UTC ISO)
                z REAL,
                direct_beta REAL,
                beta_30_weekly REAL,
                beta_drift_pct REAL,
                close_price_1 REAL,
                close_price_2 REAL,
                PRIMARY KEY (pair, timeframe, time)
            );
        """)
        # Index for faster range queries on history
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_pair_series_time 
            ON pair_series (time);
        """)

        # 4. Pair Parameters
        # Configurable settings per pair.
        con.execute("""
            CREATE TABLE IF NOT EXISTS pair_params (
                pair TEXT PRIMARY KEY,
                z_entry REAL NOT NULL,
                z_exit REAL NOT NULL,
                max_drift_pct REAL NOT NULL,
                alloc_pct REAL NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1
            );
        """)

        # 5. Orders
        # Stable IDs for auditability.
        con.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,       -- Broker ID or internal UUID
                client_order_id TEXT,            -- Internal stable ID
                engine_id TEXT,
                pair TEXT,
                symbol TEXT,
                direction TEXT,
                qty INTEGER,
                order_type TEXT,
                limit_price REAL,
                status TEXT,
                created_at TEXT,
                updated_at TEXT
            );
        """)

        # 6. Fills
        con.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT,
                symbol TEXT,
                qty INTEGER,
                price REAL,
                time TEXT,
                commission REAL,
                FOREIGN KEY(order_id) REFERENCES orders(order_id)
            );
        """)
        
        # 7. Open Positions
        # Snapshot of what is currently open (for easy UI display)
        con.execute("""
            CREATE TABLE IF NOT EXISTS open_positions (
                pair TEXT PRIMARY KEY,
                sym1 TEXT,
                sym2 TEXT,
                direction TEXT,
                qty1 INTEGER,
                qty2 INTEGER,
                beta_entry REAL,
                entry_time TEXT,
                entry_price1 REAL,
                entry_price2 REAL,
                entry_z REAL,
                pnl_unrealized REAL,
                last_price1 REAL,
                last_price2 REAL,
                updated_at TEXT,
                beta_drift_limit REAL DEFAULT 10.0
            );
        """)

        # 8. Closed Trades (Legacy compatible, potentially improved)
        con.execute("""
            CREATE TABLE IF NOT EXISTS closed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT,
                direction TEXT,
                entry_time TEXT,
                exit_time TEXT,
                pnl REAL,
                meta_json TEXT
            );
        """)

        # 9. PnL Summary (Account level history)
        con.execute("""
            CREATE TABLE IF NOT EXISTS pnl_summary (
                time TEXT PRIMARY KEY,  -- Snapshot time
                equity REAL,
                realized_pnl REAL,
                unrealized_pnl REAL,
                open_pos_count INTEGER,
                closed_trade_count INTEGER
            );
        """)

        # 10. Pair Series 1m (1-minute z-score audit log)
        # Separate from 15m pair_series for observability only
        con.execute("""
            CREATE TABLE IF NOT EXISTS pair_series_1m (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,           -- 1m bar timestamp (UTC ISO)
                pair TEXT NOT NULL,
                px_a REAL,                    -- Close price of sym1
                px_b REAL,                    -- Close price of sym2
                hi_a REAL,                    -- High of 1m bar (sym1) - spike detection
                lo_a REAL,                    -- Low of 1m bar (sym1)
                hi_b REAL,                    -- High of 1m bar (sym2)
                lo_b REAL,                    -- Low of 1m bar (sym2)
                vol_a INTEGER,                -- Volume (sym1)
                vol_b INTEGER,                -- Volume (sym2)
                spread_1m REAL,               -- px_a / px_b
                mean_30d_cached REAL,         -- Cached mean from slow layer
                std_30d_cached REAL,          -- Cached std from slow layer
                z_1m REAL,                    -- Computed z-score
                passed_persistence INTEGER,   -- 1 if passed 2/3 check, 0 otherwise
                log_reason TEXT,              -- 'entry' | 'exit' | 'near_action'
                engine_id TEXT,
                source TEXT DEFAULT 'polygon'
            );
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_pair_series_1m_lookup 
            ON pair_series_1m (pair, time);
        """)

        # 11. Daily Performance
        # End-of-day stats for tracking.
        con.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                realized_pnl REAL,
                total_equity REAL,
                num_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                updated_at TEXT
            );
        """)

    # 12. Event Risk tables (earnings, dividends, macro, audit log)
    from event_risk import init_event_tables
    init_event_tables(con)

    # 13. Manual Commands
    con.execute("""
        CREATE TABLE IF NOT EXISTS manual_commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command TEXT NOT NULL,
            payload TEXT, -- JSON
            status TEXT DEFAULT 'PENDING',
            created_at TEXT,
            processed_at TEXT
        );
    """)

# ==============================================================================
# WRITERS
# ==============================================================================

def add_manual_command(con: sqlite3.Connection, command: str, payload: dict = None) -> None:
    import json
    payload_json = json.dumps(payload) if payload else None
    created_at = datetime.now(timezone.utc).isoformat()
    with con:
        con.execute("""
            INSERT INTO manual_commands (command, payload, status, created_at)
            VALUES (?, ?, 'PENDING', ?)
        """, (command, payload_json, created_at))

def mark_command_processed(con: sqlite3.Connection, command_id: int, status: str = "PROCESSED") -> None:
    processed_at = datetime.now(timezone.utc).isoformat()
    with con:
        con.execute("""
            UPDATE manual_commands 
            SET status = ?, processed_at = ?
            WHERE id = ?
        """, (status, processed_at, command_id))

def get_pending_commands(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = con.execute("SELECT * FROM manual_commands WHERE status = 'PENDING' ORDER BY created_at ASC")
    rows = cur.fetchall()
    return [dict(zip([c[0] for c in cur.description], row)) for row in rows]


# ==============================================================================
# HELPERS
# ==============================================================================

def _sanitize_db_types(df: pd.DataFrame, time_cols: List[str]) -> pd.DataFrame:
    """
    Convert pandas/numpy types to SQLite-safe native types (str, int, float, None).
    Specifically handles Timestamps -> ISO8601 strings.
    """
    if df.empty:
        return df
        
    out = df.copy()
    
    # 1. Handle Time Columns (Timestamp -> ISO String)
    for c in time_cols:
        if c in out.columns:
            # Coerce to datetime (UTC)
            series = pd.to_datetime(out[c], errors="coerce", utc=True)
            # Convert to ISO string, mapping NaT to None
            # map(lambda x: x.isoformat if not null) is slow but safe. 
            # optimized: dt.strftime for non-nulls, but handling NaT is tricky with strftime alone.
            out[c] = series.apply(lambda x: x.isoformat().replace("+00:00", "Z") if pd.notnull(x) else None)

    # 2. Handle generic object columns that might be numpy types
    # (Optional sanity check, but strictly typed columns usually fine)
    
    return out


# ==============================================================================
# WRITERS
# ==============================================================================

def pulse_heartbeat(con: sqlite3.Connection, engine_id: str, status: str, error: Optional[str] = None, meta: Optional[dict] = None) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(meta) if meta else None
    with con:
        con.execute("""
            INSERT OR REPLACE INTO engine_heartbeat (engine_id, timestamp, status, error_message, meta_json)
            VALUES (?, ?, ?, ?, ?)
        """, (engine_id, now_iso, status, error, meta_json))


def save_snapshot(con: sqlite3.Connection, metrics_df: pd.DataFrame) -> None:
    """
    Upsert latest metrics to live_metrics_snapshot.
    metrics_df must have: pair, time, z, ...
    """
    if metrics_df is None or metrics_df.empty:
        return
    
    # Ensure columns exist
    df = metrics_df.copy()
    if "last_updated" not in df.columns:
        df["last_updated"] = datetime.now(timezone.utc).isoformat()
        
    required_cols = ["pair", "time", "z", "direct_beta", "beta_30_weekly", "beta_drift_pct", "last_updated"]
    for c in required_cols:
        if c not in df.columns:
            # fill missing with None/NaN if permissive, or raise
            df[c] = None
    
    # Sanitize types (convert timestamps to strings)
    df = _sanitize_db_types(df, time_cols=["time", "last_updated"])
            
    data = list(df[required_cols].itertuples(index=False, name=None))
    
    with con:
        con.executemany("""
            INSERT OR REPLACE INTO live_metrics_snapshot 
            (pair, time, z, direct_beta, beta_30_weekly, beta_drift_pct, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data)


def save_history(con: sqlite3.Connection, series_df: pd.DataFrame, timeframe: str = "15m") -> None:
    """
    Append to pair_series. UNIQUE constraint (pair, timeframe, time) prevents duplicates.
    """
    if series_df is None or series_df.empty:
        return

    df = series_df.copy()
    df["timeframe"] = timeframe
    
    # Optional columns logic could go here
    cols = ["pair", "timeframe", "time", "z", "direct_beta", "beta_30_weekly", "beta_drift_pct"]
    extra_cols = ["close_price_1", "close_price_2"]
    
    final_cols = cols + [c for c in extra_cols if c in df.columns]
    
    # Ensure they exist in DF
    for c in cols:
        if c not in df.columns:
            df[c] = None

    # Sanitize types (convert timestamps to strings)
    df = _sanitize_db_types(df, time_cols=["time"])

    placeholders = ",".join(["?"] * len(final_cols))
    col_names = ",".join(final_cols)
    
    data = list(df[final_cols].itertuples(index=False, name=None))
    
    with con:
        con.executemany(f"""
            INSERT OR REPLACE INTO pair_series ({col_names})
            VALUES ({placeholders})
        """, data)


def save_open_positions(con: sqlite3.Connection, positions_df: pd.DataFrame, engine_id: str = "US") -> None:
    with con:
        # Safe Delete: Only remove positions managed by THIS engine
        # But we need to handle the case where engine_id is new.
        # If schema has engine_id, use it. If not, fallback (or assume migration done).
        # We assume migration is done via script.
        try:
            con.execute("DELETE FROM open_positions WHERE engine_id = ?", (engine_id,))
        except sqlite3.OperationalError:
             # Fallback if migration not run yet (shouldn't happen if we run script first)
             con.execute("DELETE FROM open_positions")

        if positions_df is not None and not positions_df.empty:
             records = []
             now_iso = datetime.now(timezone.utc).isoformat()
             
             for _, row in positions_df.iterrows():
                 # Helper to safe-get and sanitize
                 def _get(key, typ, default):
                     val = row.get(key, default)
                     if pd.isna(val): return None
                     if typ == "time" and val:
                         try:
                             return pd.Timestamp(val).isoformat()
                         except:
                             return str(val)
                     return val

                 records.append((
                     str(row.get("pair")),
                     str(row.get("sym1", "")),
                     str(row.get("sym2", "")),
                     str(row.get("direction", "")),
                     int(row.get("qty1", 0)),
                     int(row.get("qty2", 0)),
                     float(row.get("beta_entry", 0.0)),
                     _get("entry_time", "time", None),
                     float(row.get("entry_price1", 0.0)),
                     float(row.get("entry_price2", 0.0)),
                     float(row.get("entry_z", 0.0)),
                     float(row.get("pnl_unrealized", 0.0)),
                     float(row.get("last_price1", 0.0)) if pd.notnull(row.get("last_price1")) else None,
                     float(row.get("last_price2", 0.0)) if pd.notnull(row.get("last_price2")) else None,
                     now_iso,
                     engine_id,
                     float(row.get("beta_drift_limit", 10.0))
                 ))
             
             # Insert with engine_id
             con.executemany("""
                 INSERT INTO open_positions 
                 (pair, sym1, sym2, direction, qty1, qty2, beta_entry, entry_time, entry_price1, entry_price2, entry_z, pnl_unrealized, last_price1, last_price2, updated_at, engine_id, beta_drift_limit)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
             """, records)


def save_pnl_snapshot(con: sqlite3.Connection, equity: float, realized: float, unrealized: float, open_count: int, closed_count: int) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with con:
        con.execute("""
            INSERT OR REPLACE INTO pnl_summary (time, equity, realized_pnl, unrealized_pnl, open_pos_count, closed_trade_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (now_iso, equity, realized, unrealized, open_count, closed_count))


def log_order(con: sqlite3.Connection, order: Dict[str, Any]) -> None:
    # order dict -> columns
    # Safe insert
    with con:
        con.execute("""
            INSERT OR REPLACE INTO orders (order_id, client_order_id, engine_id, pair, symbol, direction, qty, order_type, limit_price, status, created_at, updated_at)
            VALUES (:order_id, :client_order_id, :engine_id, :pair, :symbol, :direction, :qty, :order_type, :limit_price, :status, :created_at, :updated_at)
        """, order)


def upsert_pair_params(con: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    
    # We expect specific columns
    required = ["pair", "z_entry", "z_exit", "max_drift_pct", "max_drift_delta", "alloc_pct", "enabled"]
    subset = df[required].copy()
    
    data = list(subset.itertuples(index=False, name=None))
    with con:
        con.executemany("""
            INSERT OR REPLACE INTO pair_params (pair, z_entry, z_exit, max_drift_pct, max_drift_delta, alloc_pct, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data)

# ==============================================================================
# READERS
# ==============================================================================

def get_latest_snapshot(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM live_metrics_snapshot", con)

def get_pair_history(con: sqlite3.Connection, pair: str, timeframe: str = "15m", limit: int = 500) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT * FROM (
            SELECT * FROM pair_series 
            WHERE pair = ? AND timeframe = ? 
            ORDER BY time DESC
            LIMIT ?
        ) ORDER BY time ASC
    """, con, params=(pair, timeframe, limit))

def get_open_positions(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM open_positions", con)

def get_pnl_history(con: sqlite3.Connection, limit: int = 100) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM pnl_summary ORDER BY time DESC LIMIT ?", con, params=(limit,))

def get_heartbeat(con: sqlite3.Connection, engine_id: Optional[str] = None) -> List[Dict]:
    q = "SELECT * FROM engine_heartbeat"
    params = ()
    if engine_id:
        q += " WHERE engine_id = ?"
        params = (engine_id,)
    q += " ORDER BY timestamp DESC"
    
    df = pd.read_sql_query(q, con, params=params)
    return df.to_dict(orient="records")

def get_pair_params(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM pair_params", con)


# ==============================================================================
# 1-MINUTE SIGNAL LOG (Audit / Observability)
# ==============================================================================

def save_1m_signal_logs(con: sqlite3.Connection, records: List[Dict[str, Any]]) -> None:
    """
    Batch insert 1-minute z-score evaluations for auditability.
    Called by the fast engine during signal evaluation.
    
    Each record should contain:
        time, pair, px_a, px_b, hi_a, lo_a, hi_b, lo_b, vol_a, vol_b,
        spread_1m, mean_30d_cached, std_30d_cached, z_1m, passed_persistence,
        log_reason, engine_id, source
    """
    if not records:
        return
    
    with con:
        con.executemany("""
            INSERT INTO pair_series_1m 
            (time, pair, px_a, px_b, hi_a, lo_a, hi_b, lo_b, vol_a, vol_b,
             spread_1m, mean_30d_cached, std_30d_cached, z_1m, passed_persistence,
             log_reason, engine_id, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(
            r.get("time"),
            r.get("pair"),
            r.get("px_a"),
            r.get("px_b"),
            r.get("hi_a"),
            r.get("lo_a"),
            r.get("hi_b"),
            r.get("lo_b"),
            r.get("vol_a"),
            r.get("vol_b"),
            r.get("spread_1m"),
            r.get("mean_30d_cached"),
            r.get("std_30d_cached"),
            r.get("z_1m"),
            1 if r.get("passed_persistence") else 0,
            r.get("log_reason", "monitoring"),
            r.get("engine_id", "US_1M"),
            r.get("source", "polygon")
        ) for r in records])


def get_1m_series(con: sqlite3.Connection, pair: str, start_time: str = None, end_time: str = None, limit: int = 500) -> pd.DataFrame:
    """
    Retrieve 1-minute z-score history for a pair within a time window.
    Used by dashboard for trade analysis visualization.
    """
    query = "SELECT * FROM pair_series_1m WHERE pair = ?"
    params = [pair]
    
    if start_time:
        query += " AND time >= ?"
        params.append(start_time)
    if end_time:
        query += " AND time <= ?"
        params.append(end_time)
    
    query += " ORDER BY time ASC LIMIT ?"
    params.append(limit)
    
    return pd.read_sql_query(query, con, params=params)


def update_position_limits(con: sqlite3.Connection, updates: List[Dict[str, Any]]) -> None:
    """
    Update beta_drift_limit for specific open positions.
    updates: list of dicts with 'pair' and 'beta_drift_limit' keys.
    """
    if not updates:
        return
        
    with con:
        con.executemany("""
            UPDATE open_positions 
            SET beta_drift_limit = :beta_drift_limit
            WHERE pair = :pair
        """, updates)


def save_daily_performance(con: sqlite3.Connection, date: str, realized_pnl: float, total_equity: float, num_trades: int, wins: int, losses: int) -> None:
    """
    Upsert daily performance stats.
    date: YYYY-MM-DD
    """
    updated_at = datetime.now(timezone.utc).isoformat()
    with con:
        con.execute("""
            INSERT OR REPLACE INTO daily_performance 
            (date, realized_pnl, total_equity, num_trades, wins, losses, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (date, realized_pnl, total_equity, num_trades, wins, losses, updated_at))


def get_daily_performance(con: sqlite3.Connection, limit: int = 30) -> pd.DataFrame:
    """
    Get daily performance stats for dashboard.
    """
    return pd.read_sql_query("SELECT * FROM daily_performance ORDER BY date DESC LIMIT ?", con, params=(limit,))
