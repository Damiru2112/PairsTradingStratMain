# event_risk.py
"""
Event Risk Layer for Pairs Trading Engine.

Provides compute_pair_event_state() which returns multiplier, entry_blocked,
and active/upcoming event info for a given pair.

Data sources:
  - Earnings: yfinance (.earnings_dates)
  - Dividends: Polygon /v3/reference/dividends
  - Macro: static data/macro_calendar.json
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

log = logging.getLogger("event_risk")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

EARNINGS_MULTIPLIER = 1.5
DIVIDEND_MULTIPLIER = 1.3
MACRO_MULTIPLIER = 1.2

# Window bounds (trading days relative to event date, inclusive)
EARNINGS_WINDOW = (-1, +2)   # days -1, 0, +1, +2
DIVIDEND_WINDOW = (-1, +1)   # days -1, 0, +1
MACRO_WINDOW = (0, 0)        # day 0 only

Z_EXIT_CAP = 1.5  # Exit threshold hard cap

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "Uo5696bmc67zfyX23fbjBqfEk7nPYkWu")
MACRO_CALENDAR_PATH = Path(__file__).parent / "data" / "macro_calendar.json"

# NYSE calendar (singleton)
_nyse_cal = None


def _get_nyse_cal():
    global _nyse_cal
    if _nyse_cal is None:
        _nyse_cal = mcal.get_calendar("XNYS")
    return _nyse_cal


# ---------------------------------------------------------------------------
# NYSE TRADING-DAY HELPERS
# ---------------------------------------------------------------------------

def trading_day_offset(ref_date: date, current_date: date) -> int:
    """
    Compute signed trading-day offset from ref_date to current_date.
    Positive = current_date is AFTER ref_date.
    Uses NYSE calendar.
    """
    nyse = _get_nyse_cal()

    if current_date == ref_date:
        return 0

    if current_date > ref_date:
        days = nyse.valid_days(
            start_date=pd.Timestamp(ref_date),
            end_date=pd.Timestamp(current_date),
        )
        # valid_days is inclusive; subtract 1 because we want offset (0 for same day)
        return len(days) - 1
    else:
        days = nyse.valid_days(
            start_date=pd.Timestamp(current_date),
            end_date=pd.Timestamp(ref_date),
        )
        return -(len(days) - 1)


def next_n_trading_days(from_date: date, n: int = 10) -> List[date]:
    """Return next n NYSE trading days starting from from_date (inclusive if trading day)."""
    nyse = _get_nyse_cal()
    end = from_date + timedelta(days=n * 2 + 5)  # generous buffer
    days = nyse.valid_days(
        start_date=pd.Timestamp(from_date),
        end_date=pd.Timestamp(end),
    )
    return [d.date() for d in days[:n]]


def current_trading_date(dt_et: datetime) -> date:
    """
    Determine the 'current trading date' in US/Eastern.
    Simply uses the date portion of the Eastern-localized datetime.
    """
    if dt_et.tzinfo is None:
        dt_et = dt_et.replace(tzinfo=EASTERN)
    return dt_et.astimezone(EASTERN).date()


# ---------------------------------------------------------------------------
# DATA FETCHING: EARNINGS (yfinance)
# ---------------------------------------------------------------------------

def fetch_earnings_events(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch upcoming/recent earnings dates for a ticker using yfinance.
    Returns list of {"ticker": str, "report_date": "YYYY-MM-DD"}.
    """
    """
    Fetch upcoming/recent earnings dates for a ticker using yfinance via subprocess.
    Returns list of {"ticker": str, "report_date": "YYYY-MM-DD"}.
    """
    import subprocess
    import sys
    
    script_path = Path(__file__).parent / "scripts" / "fetch_ticker_earnings.py"
    
    try:
        # Run the standalone script with a timeout
        result = subprocess.run(
            [sys.executable, str(script_path), ticker],
            capture_output=True,
            text=True,
            timeout=3  # reduced timeout for faster warmup
        )
        
        if result.returncode != 0:
            log.warning(f"Earnings subprocess failed for {ticker}: {result.stderr.strip()}")
            return []
            
        data = json.loads(result.stdout)
        return data
        
    except subprocess.TimeoutExpired:
        log.warning(f"Timeout fetching earnings for {ticker} (subprocess)")
        return []
    except json.JSONDecodeError:
        log.warning(f"Failed to parse earnings JSON for {ticker}")
        return []
    except Exception as e:
        log.warning(f"Subprocess earnings fetch failed for {ticker}: {e}")
        return []


# ---------------------------------------------------------------------------
# DATA FETCHING: DIVIDENDS (Polygon)
# ---------------------------------------------------------------------------

def fetch_dividend_events(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch dividend ex-dates for a ticker from Polygon (with pagination).
    Returns list of {"ticker": str, "ex_date": "YYYY-MM-DD", "amount": float}.
    """
    import requests

    results = []
    url = f"https://api.polygon.io/v3/reference/dividends"
    params = {
        "ticker": ticker,
        "order": "desc",
        "limit": 50,
        "apiKey": POLYGON_API_KEY,
    }

    try:
        while url:
            resp = requests.get(url, params=params, timeout=10)
            if not resp.ok:
                log.warning(f"Polygon dividends request failed for {ticker}: {resp.status_code}")
                break
            data = resp.json()
            for item in data.get("results", []):
                ex = item.get("ex_dividend_date")
                if ex:
                    results.append({
                        "ticker": ticker,
                        "ex_date": ex,
                        "amount": item.get("cash_amount", 0.0),
                    })
            # Pagination
            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": POLYGON_API_KEY}
            else:
                break
    except Exception as e:
        log.warning(f"Polygon dividend fetch failed for {ticker}: {e}")

    return results


# ---------------------------------------------------------------------------
# DATA FETCHING: MACRO (local file)
# ---------------------------------------------------------------------------

def load_macro_calendar(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load macro calendar from JSON file.
    Returns list of {"date": "YYYY-MM-DD", "type": str, "label": str}.
    """
    p = path or MACRO_CALENDAR_PATH
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load macro calendar from {p}: {e}")
        return []


# ---------------------------------------------------------------------------
# SQLite SCHEMA + REFRESH
# ---------------------------------------------------------------------------

def init_event_tables(con: sqlite3.Connection) -> None:
    """Create event tables (idempotent)."""
    with con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS events_earnings (
                ticker TEXT NOT NULL,
                report_date TEXT NOT NULL,
                updated_at TEXT,
                PRIMARY KEY (ticker, report_date)
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS events_dividends (
                ticker TEXT NOT NULL,
                ex_date TEXT NOT NULL,
                amount REAL,
                updated_at TEXT,
                PRIMARY KEY (ticker, ex_date)
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS events_macro (
                date TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS event_risk_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                ts TEXT NOT NULL,
                multiplier REAL,
                entry_blocked INTEGER,
                active_events_json TEXT,
                z_entry_eff REAL,
                z_exit_eff REAL
            );
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_risk_log_pair_ts
            ON event_risk_log (pair, ts);
        """)

        # 5. Overrides
        con.execute("""
            CREATE TABLE IF NOT EXISTS event_risk_overrides (
                pair TEXT PRIMARY KEY,
                multiplier REAL,
                entry_blocked INTEGER, -- 0 or 1. If NULL, use calculated.
                updated_at TEXT
            );
        """)


def refresh_event_data(con: sqlite3.Connection, tickers: List[str]) -> None:
    """
    Fetch earnings + dividends for all tickers, load macro calendar.
    Upserts into SQLite. Call once per day at engine startup.
    """
    init_event_tables(con)
    now_iso = datetime.now(UTC).isoformat()

    # --- Earnings (yfinance) ---
    # --- Earnings (yfinance via subprocess) ---
    for ticker in tickers:
        try:
            events = fetch_earnings_events(ticker)
            if events:
                with con:
                    con.executemany("""
                        INSERT OR REPLACE INTO events_earnings (ticker, report_date, updated_at)
                        VALUES (?, ?, ?)
                    """, [(e["ticker"], e["report_date"], now_iso) for e in events])
                log.info(f"Refreshed {len(events)} earnings events for {ticker}")
        except Exception as e:
            log.warning(f"Failed to refresh earnings for {ticker}: {e}")

    # --- Dividends (Polygon) ---
    for ticker in tickers:
        try:
            events = fetch_dividend_events(ticker)
            if events:
                with con:
                    con.executemany("""
                        INSERT OR REPLACE INTO events_dividends (ticker, ex_date, amount, updated_at)
                        VALUES (?, ?, ?, ?)
                    """, [(e["ticker"], e["ex_date"], e["amount"], now_iso) for e in events])
                log.info(f"Refreshed {len(events)} dividend events for {ticker}")
        except Exception as e:
            log.warning(f"Failed to refresh dividends for {ticker}: {e}")

    # --- Macro (local file) ---
    try:
        macro_events = load_macro_calendar()
        if macro_events:
            with con:
                con.executemany("""
                    INSERT OR REPLACE INTO events_macro (date, type, label)
                    VALUES (?, ?, ?)
                """, [(m["date"], m["type"], m.get("label", "")) for m in macro_events])
            log.info(f"Loaded {len(macro_events)} macro calendar events")
    except Exception as e:
        log.warning(f"Failed to load macro calendar: {e}")


# ---------------------------------------------------------------------------
# CORE API: compute_pair_event_state
# ---------------------------------------------------------------------------

def _get_event_override(con: sqlite3.Connection, pair: str) -> Optional[Dict[str, Any]]:
    try:
        row = con.execute("SELECT multiplier, entry_blocked FROM event_risk_overrides WHERE pair = ?", (pair,)).fetchone()
        if row:
            return {"multiplier": row[0], "entry_blocked": row[1]}
    except sqlite3.OperationalError:
        # Table might not exist yet if migration pending
        return None
    return None


def set_event_override(con: sqlite3.Connection, pair: str, multiplier: Optional[float] = None, entry_blocked: Optional[bool] = None) -> None:
    """
    Upsert override. passing None for a field means 'use calculated' (set to NULL in DB?).
    Actually simpler: The UI passes current state.
    If we want to clear override, we delete row.
    """
    now_iso = datetime.now(UTC).isoformat()
    
    # Check if row exists to partial update? No, UI sends full state.
    # If both None, delete?
    if multiplier is None and entry_blocked is None:
        with con:
            con.execute("DELETE FROM event_risk_overrides WHERE pair = ?", (pair,))
        return

    with con:
        # We use INSERT OR REPLACE. 
        # But we need to handle partials if we want. For now assume full replace.
        # Store booleans as integers (0, 1). None -> NULL.
        blk = 1 if entry_blocked is True else (0 if entry_blocked is False else None)
        con.execute("""
            INSERT OR REPLACE INTO event_risk_overrides (pair, multiplier, entry_blocked, updated_at)
            VALUES (?, ?, ?, ?)
        """, (pair, multiplier, blk, now_iso))


def _query_ticker_earnings(con: sqlite3.Connection, ticker: str, window_start: date, window_end: date) -> List[Dict]:
    """Query earnings events for a ticker within a date range."""
    rows = con.execute("""
        SELECT report_date FROM events_earnings
        WHERE ticker = ? AND report_date >= ? AND report_date <= ?
        ORDER BY report_date
    """, (ticker, window_start.isoformat(), window_end.isoformat())).fetchall()
    return [{"date": r[0], "type": "earnings"} for r in rows]


def _query_ticker_dividends(con: sqlite3.Connection, ticker: str, window_start: date, window_end: date) -> List[Dict]:
    """Query dividend events for a ticker within a date range."""
    rows = con.execute("""
        SELECT ex_date FROM events_dividends
        WHERE ticker = ? AND ex_date >= ? AND ex_date <= ?
        ORDER BY ex_date
    """, (ticker, window_start.isoformat(), window_end.isoformat())).fetchall()
    return [{"date": r[0], "type": "dividend"} for r in rows]


def _query_macro_events(con: sqlite3.Connection, window_start: date, window_end: date) -> List[Dict]:
    """Query macro events within a date range."""
    rows = con.execute("""
        SELECT date, type, label FROM events_macro
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """, (window_start.isoformat(), window_end.isoformat())).fetchall()
    return [{"date": r[0], "type": "macro", "subtype": r[1], "label": r[2]} for r in rows]


def compute_pair_event_state(
    pair: str,
    current_dt_et: datetime,
    con: sqlite3.Connection,
) -> Dict[str, Any]:
    """
    Main API: computes event-risk state for a pair at a given time.

    Args:
        pair: "SYM1-SYM2"
        current_dt_et: current datetime (should be US/Eastern aware or naive=treated as ET)
        con: SQLite connection with event tables populated

    Returns:
        {
            "multiplier": float,
            "entry_blocked": bool,
            "active_events": [...],
            "next_events_10d": [...],
        }
    """
    parts = pair.split("-")
    if len(parts) != 2:
        return {"multiplier": 1.0, "entry_blocked": False, "active_events": [], "next_events_10d": []}

    sym1, sym2 = parts[0], parts[1]
    today = current_trading_date(current_dt_et)

    # Search window: cover the largest possible window around today
    # Earnings [-1,+2] means we need events from today-2 to today+1
    # (if event is 2 days ago, today is day +2 relative to event)
    # To be safe, query ±5 trading days worth of calendar days
    query_start = today - timedelta(days=10)
    query_end = today + timedelta(days=20)  # 10 trading days forward ≈ 14 calendar days

    active_events = []
    multipliers = []
    entry_blocked = False

    # --- Process per-leg events ---
    for leg_ticker in [sym1, sym2]:
        # Earnings
        earnings = _query_ticker_earnings(con, leg_ticker, query_start, query_end)
        for ev in earnings:
            ev_date = date.fromisoformat(ev["date"])
            offset = trading_day_offset(ev_date, today)
            # offset > 0 means event is in the past (today is after event)
            # offset < 0 means event is in the future (today is before event)
            # We want: window is [-1, +2] around event
            # offset == 0: event day = today
            # offset == 1: event was 1 trading day ago
            # offset == -1: event is tomorrow

            if EARNINGS_WINDOW[0] <= -offset <= EARNINGS_WINDOW[1]:
                # Inside active window
                days_to_event = -offset  # positive = future, negative = past
                window_label = f"earnings({'day 0' if offset == 0 else f'{-offset:+d}'})"

                active_events.append({
                    "type": "earnings",
                    "leg": leg_ticker,
                    "event_date": ev["date"],
                    "days_to_event": days_to_event,
                    "window_label": window_label,
                })
                multipliers.append(EARNINGS_MULTIPLIER)

                if offset == 0:
                    entry_blocked = True  # Event day

        # Dividends
        dividends = _query_ticker_dividends(con, leg_ticker, query_start, query_end)
        for ev in dividends:
            ev_date = date.fromisoformat(ev["date"])
            offset = trading_day_offset(ev_date, today)

            if DIVIDEND_WINDOW[0] <= -offset <= DIVIDEND_WINDOW[1]:
                days_to_event = -offset
                window_label = f"dividend({'day 0' if offset == 0 else f'{-offset:+d}'})"

                active_events.append({
                    "type": "dividend",
                    "leg": leg_ticker,
                    "event_date": ev["date"],
                    "days_to_event": days_to_event,
                    "window_label": window_label,
                })
                multipliers.append(DIVIDEND_MULTIPLIER)

                if offset == 0:
                    entry_blocked = True  # Ex-date day

    # --- Process macro events (market-wide, not per-leg) ---
    macro_events = _query_macro_events(con, query_start, query_end)
    for ev in macro_events:
        ev_date = date.fromisoformat(ev["date"])
        offset = trading_day_offset(ev_date, today)

        if MACRO_WINDOW[0] <= -offset <= MACRO_WINDOW[1]:
            days_to_event = -offset
            window_label = f"macro({ev.get('subtype', '')} day 0)"

            active_events.append({
                "type": "macro",
                "leg": "ALL",
                "event_date": ev["date"],
                "days_to_event": days_to_event,
                "window_label": window_label,
            })
            multipliers.append(MACRO_MULTIPLIER)

            if offset == 0:
                entry_blocked = True  # Macro event day — blocks all

    # --- Compute final multiplier ---
    multiplier = max(multipliers) if multipliers else 1.0

    # --- Check Overrides ---
    override = _get_event_override(con, pair)
    is_overridden = False
    if override:
        if override["multiplier"] is not None:
            multiplier = float(override["multiplier"])
            is_overridden = True
        if override["entry_blocked"] is not None:
            entry_blocked = bool(override["entry_blocked"])
            is_overridden = True

    # --- Next 10 trading days events (for dashboard) ---
    next_10_days = next_n_trading_days(today, 10)
    next_10_end = next_10_days[-1] if next_10_days else today + timedelta(days=15)
    next_events_10d = []

    for leg_ticker in [sym1, sym2]:
        for ev in _query_ticker_earnings(con, leg_ticker, today, next_10_end):
            ev_date = date.fromisoformat(ev["date"])
            offset = trading_day_offset(ev_date, today)
            next_events_10d.append({
                "type": "earnings", "leg": leg_ticker,
                "event_date": ev["date"], "trading_days_to": -offset,
            })

        for ev in _query_ticker_dividends(con, leg_ticker, today, next_10_end):
            ev_date = date.fromisoformat(ev["date"])
            offset = trading_day_offset(ev_date, today)
            next_events_10d.append({
                "type": "dividend", "leg": leg_ticker,
                "event_date": ev["date"], "trading_days_to": -offset,
            })

    for ev in _query_macro_events(con, today, next_10_end):
        ev_date = date.fromisoformat(ev["date"])
        offset = trading_day_offset(ev_date, today)
        next_events_10d.append({
            "type": "macro", "leg": "ALL",
            "event_date": ev["date"], "trading_days_to": -offset,
            "label": ev.get("label", ""),
        })

    return {
        "multiplier": multiplier,
        "entry_blocked": entry_blocked,
        "active_events": active_events,
        "next_events_10d": sorted(next_events_10d, key=lambda x: x.get("trading_days_to", 99)),
        "is_overridden": is_overridden
    }


# ---------------------------------------------------------------------------
# AUDIT LOGGING
# ---------------------------------------------------------------------------

def log_event_risk(
    con: sqlite3.Connection,
    pair: str,
    ts: str,
    multiplier: float,
    entry_blocked: bool,
    active_events: List[Dict],
    z_entry_eff: float,
    z_exit_eff: float,
) -> None:
    """Log event risk evaluation to audit table."""
    try:
        with con:
            con.execute("""
                INSERT INTO event_risk_log
                (pair, ts, multiplier, entry_blocked, active_events_json, z_entry_eff, z_exit_eff)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pair, ts, multiplier,
                1 if entry_blocked else 0,
                json.dumps(active_events),
                z_entry_eff, z_exit_eff,
            ))
    except Exception as e:
        log.debug(f"Failed to log event risk: {e}")
