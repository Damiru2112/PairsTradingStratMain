# event_risk.py
"""
Event Risk Layer for Pairs Trading Engine.

Two-layer architecture:
  Layer 1: compute_pair_event_state() — pure event detection (no direction awareness)
  Layer 2: resolve_event_trade_constraints() — direction-aware blocking/warnings

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

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
MACRO_CALENDAR_PATH = Path(__file__).parent / "data" / "macro_calendar.json"

# Fallback constants (used when event_type_config rows are missing)
_FALLBACK_CONFIGS = {
    "earnings": {"multiplier": EARNINGS_MULTIPLIER, "entry_blocked": True},
    "dividend": {"multiplier": DIVIDEND_MULTIPLIER, "entry_blocked": False},
    "macro":    {"multiplier": MACRO_MULTIPLIER,    "entry_blocked": True},
}

# NYSE calendar (singleton)
_nyse_cal = None


def parse_pair_symbols(pair: str) -> Tuple[str, str]:
    """Canonical pair parser. Returns (sym1, sym2) from 'SYM1-SYM2'."""
    parts = pair.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid pair format: {pair!r}, expected 'SYM1-SYM2'")
    return parts[0], parts[1]


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

        # 6. Event type configuration (configurable multipliers & blocking policy)
        con.execute("""
            CREATE TABLE IF NOT EXISTS event_type_config (
                event_type TEXT PRIMARY KEY,
                multiplier REAL NOT NULL,
                entry_blocked INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT
            );
        """)
        # Seed defaults if table is empty
        existing = con.execute("SELECT COUNT(*) FROM event_type_config").fetchone()[0]
        if existing == 0:
            now_iso = datetime.now(UTC).isoformat()
            con.executemany(
                "INSERT INTO event_type_config (event_type, multiplier, entry_blocked, updated_at) VALUES (?, ?, ?, ?)",
                [
                    ("earnings", EARNINGS_MULTIPLIER, 1, now_iso),
                    ("dividend", DIVIDEND_MULTIPLIER, 0, now_iso),  # dividends: allow by default
                    ("macro",    MACRO_MULTIPLIER,    1, now_iso),
                ],
            )


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
# EVENT TYPE CONFIG (configurable multipliers & blocking policy)
# ---------------------------------------------------------------------------

def get_event_type_configs(con: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """
    Read event type configs from DB.
    Returns {event_type: {"multiplier": float, "entry_blocked": bool}}.
    Falls back to _FALLBACK_CONFIGS with a warning for missing rows.
    """
    configs = {}
    try:
        rows = con.execute("SELECT event_type, multiplier, entry_blocked FROM event_type_config").fetchall()
        for r in rows:
            configs[r[0]] = {"multiplier": float(r[1]), "entry_blocked": bool(r[2])}
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet

    # Fill in any missing event types from fallbacks
    for etype, fallback in _FALLBACK_CONFIGS.items():
        if etype not in configs:
            log.warning(
                "event_type_config missing for '%s', using fallback multiplier=%.1f, entry_blocked=%s",
                etype, fallback["multiplier"], fallback["entry_blocked"],
            )
            configs[etype] = dict(fallback)

    return configs


def set_event_type_config(
    con: sqlite3.Connection,
    event_type: str,
    multiplier: float,
    entry_blocked: bool,
) -> None:
    """Upsert event type configuration."""
    now_iso = datetime.now(UTC).isoformat()
    with con:
        con.execute("""
            INSERT OR REPLACE INTO event_type_config (event_type, multiplier, entry_blocked, updated_at)
            VALUES (?, ?, ?, ?)
        """, (event_type, multiplier, 1 if entry_blocked else 0, now_iso))


# ---------------------------------------------------------------------------
# CORE API: compute_pair_event_state (Layer 1 — Pure Detection)
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
    Layer 1 — Pure event detection. No direction awareness.

    Args:
        pair: "SYM1-SYM2"
        current_dt_et: current datetime (should be US/Eastern aware or naive=treated as ET)
        con: SQLite connection with event tables populated

    Returns:
        {
            "multiplier": float,
            "base_entry_blocked": bool,
            "base_blocking_reasons": List[str],
            "active_events": [...],
            "next_events_10d": [...],
            "is_overridden": bool,
        }
    """
    try:
        sym1, sym2 = parse_pair_symbols(pair)
    except ValueError:
        return {
            "multiplier": 1.0, "base_entry_blocked": False,
            "base_blocking_reasons": [], "active_events": [],
            "next_events_10d": [], "is_overridden": False,
        }

    today = current_trading_date(current_dt_et)

    # Read configurable multipliers & blocking policy from DB
    configs = get_event_type_configs(con)

    # Search window: cover the largest possible window around today
    query_start = today - timedelta(days=10)
    query_end = today + timedelta(days=20)

    active_events = []
    multipliers = []
    base_entry_blocked = False
    base_blocking_reasons: List[str] = []

    # --- Process per-leg events ---
    for leg_ticker in [sym1, sym2]:
        # Earnings
        earnings_cfg = configs.get("earnings", _FALLBACK_CONFIGS["earnings"])
        earnings = _query_ticker_earnings(con, leg_ticker, query_start, query_end)
        for ev in earnings:
            ev_date = date.fromisoformat(ev["date"])
            offset = trading_day_offset(ev_date, today)

            if EARNINGS_WINDOW[0] <= -offset <= EARNINGS_WINDOW[1]:
                days_to_event = -offset
                window_label = f"earnings({'day 0' if offset == 0 else f'{-offset:+d}'})"

                active_events.append({
                    "type": "earnings",
                    "leg": leg_ticker,
                    "event_date": ev["date"],
                    "days_to_event": days_to_event,
                    "window_label": window_label,
                    "config_multiplier": earnings_cfg["multiplier"],
                    "config_entry_blocked": earnings_cfg["entry_blocked"],
                })
                multipliers.append(earnings_cfg["multiplier"])

                if offset == 0 and earnings_cfg["entry_blocked"]:
                    base_entry_blocked = True
                    base_blocking_reasons.append(f"earnings(day 0, {leg_ticker})")

        # Dividends
        dividend_cfg = configs.get("dividend", _FALLBACK_CONFIGS["dividend"])
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
                    "config_multiplier": dividend_cfg["multiplier"],
                    "config_entry_blocked": dividend_cfg["entry_blocked"],
                })
                multipliers.append(dividend_cfg["multiplier"])

                # Dividend blocking only applies if config says so (default: False)
                if offset == 0 and dividend_cfg["entry_blocked"]:
                    base_entry_blocked = True
                    base_blocking_reasons.append(f"dividend(day 0, {leg_ticker})")

    # --- Process macro events (market-wide, not per-leg) ---
    macro_cfg = configs.get("macro", _FALLBACK_CONFIGS["macro"])
    macro_events = _query_macro_events(con, query_start, query_end)
    for ev in macro_events:
        ev_date = date.fromisoformat(ev["date"])
        offset = trading_day_offset(ev_date, today)

        if MACRO_WINDOW[0] <= -offset <= MACRO_WINDOW[1]:
            days_to_event = -offset
            subtype = ev.get("subtype", "")
            window_label = f"macro({subtype} day 0)"

            active_events.append({
                "type": "macro",
                "leg": "ALL",
                "event_date": ev["date"],
                "days_to_event": days_to_event,
                "window_label": window_label,
                "config_multiplier": macro_cfg["multiplier"],
                "config_entry_blocked": macro_cfg["entry_blocked"],
            })
            multipliers.append(macro_cfg["multiplier"])

            if offset == 0 and macro_cfg["entry_blocked"]:
                base_entry_blocked = True
                base_blocking_reasons.append(f"macro({subtype} day 0)")

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
            base_entry_blocked = bool(override["entry_blocked"])
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
        "base_entry_blocked": base_entry_blocked,
        "base_blocking_reasons": base_blocking_reasons,
        "active_events": active_events,
        "next_events_10d": sorted(next_events_10d, key=lambda x: x.get("trading_days_to", 99)),
        "is_overridden": is_overridden,
    }


# ---------------------------------------------------------------------------
# LAYER 2: resolve_event_trade_constraints (Direction-Aware)
# ---------------------------------------------------------------------------

def resolve_event_trade_constraints(
    pair: str,
    event_state: Dict[str, Any],
    z_now: Optional[float] = None,
    current_position: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Layer 2 — Direction-aware blocking and warnings.

    Takes raw event state from compute_pair_event_state() and resolves
    trade constraints based on z_now direction and current position.

    Args:
        pair: "SYM1-SYM2"
        event_state: output of compute_pair_event_state()
        z_now: current z-score (>0 → short sym1, <0 → short sym2, None/0 → unknown)
        current_position: {"direction": "SHORT_SPREAD"|"LONG_SPREAD"} or None

    Returns:
        {
            "effective_multiplier": float,
            "effective_entry_blocked": bool,
            "dividend_short_leg_blocked": bool,
            "intended_short_leg": str | None,
            "current_short_leg": str | None,
            "hold_warning": bool,
            "hold_blocked": bool,
            "blocking_reasons": List[str],
            "warning_reasons": List[str],
            "active_events": List[Dict],
        }
    """
    try:
        sym1, sym2 = parse_pair_symbols(pair)
    except ValueError:
        return {
            "effective_multiplier": 1.0, "effective_entry_blocked": False,
            "dividend_short_leg_blocked": False,
            "intended_short_leg": None, "current_short_leg": None,
            "hold_warning": False, "hold_blocked": False,
            "blocking_reasons": [], "warning_reasons": [],
            "active_events": [],
        }

    # Seed blocking reasons from Layer 1
    blocking_reasons: List[str] = list(event_state.get("base_blocking_reasons", []))
    warning_reasons: List[str] = []

    effective_entry_blocked = event_state.get("base_entry_blocked", False)
    effective_multiplier = event_state.get("multiplier", 1.0)
    dividend_short_leg_blocked = False

    # Determine intended short leg from z_now
    intended_short_leg: Optional[str] = None
    if z_now is not None and z_now != 0:
        # z > 0 → SHORT_SPREAD → sym1 is shorted
        # z < 0 → LONG_SPREAD → sym2 is shorted
        intended_short_leg = sym1 if z_now > 0 else sym2

    # Determine current short leg from existing position
    current_short_leg: Optional[str] = None
    if current_position:
        direction = current_position.get("direction", "")
        if direction == "SHORT_SPREAD":
            current_short_leg = sym1
        elif direction == "LONG_SPREAD":
            current_short_leg = sym2

    # Process dividend events for direction-aware blocking
    hold_warning = False
    hold_blocked = False
    active_dividend_events = [
        e for e in event_state.get("active_events", []) if e.get("type") == "dividend"
    ]

    for div_ev in active_dividend_events:
        div_leg = div_ev.get("leg", "")

        # Entry blocking: dividend on would-be-short leg
        if intended_short_leg is not None:
            if div_leg == intended_short_leg:
                dividend_short_leg_blocked = True
                effective_entry_blocked = True
                blocking_reasons.append(f"dividend({div_leg} short leg)")
        elif active_dividend_events:
            # z_now is None or 0 — can't determine direction
            warning_reasons.append(
                f"dividend active on {div_leg}, direction unknown (z_now={'None' if z_now is None else z_now})"
            )

        # Hold warning: dividend on current short leg in existing position
        if current_short_leg is not None and div_leg == current_short_leg:
            hold_warning = True
            warning_reasons.append(
                f"dividend on short leg {div_leg} in open position"
            )

    return {
        "effective_multiplier": effective_multiplier,
        "effective_entry_blocked": effective_entry_blocked,
        "dividend_short_leg_blocked": dividend_short_leg_blocked,
        "intended_short_leg": intended_short_leg,
        "current_short_leg": current_short_leg,
        "hold_warning": hold_warning,
        "hold_blocked": hold_blocked,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "active_events": event_state.get("active_events", []),
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
