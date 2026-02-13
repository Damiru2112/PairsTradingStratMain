from __future__ import annotations

import os
import requests
import pandas as pd
import time
import logging
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, Optional, Tuple, Any

# Setup Logging
log = logging.getLogger("data_polygon")

# Constants
API_KEY = os.getenv("POLYGON_API_KEY", "Uo5696bmc67zfyX23fbjBqfEk7nPYkWu")
BASE_URL = "https://api.polygon.io"
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

def _to_ms(ts: pd.Timestamp | datetime | str) -> int:
    """Convert timestamp to millisecond integer for Polygon API."""
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC) # Assume UTC if not specified
    return int(ts.timestamp() * 1000)

def get_aggs(
    ticker: str, 
    multiplier: int, 
    timespan: str, 
    start: str | pd.Timestamp, 
    end: str | pd.Timestamp, 
    limit: int = 50000
) -> pd.DataFrame:
    """
    Fetch aggregates from Polygon.io.
    
    Args:
        ticker: Symbol name (e.g. 'AAPL')
        multiplier: Size of the timespan multiplier (e.g. 15)
        timespan: Size of the time window (e.g. 'minute')
        start: Start time (str YYYY-MM-DD or Timestamp)
        end: End time (str YYYY-MM-DD or Timestamp)
    
    Returns:
        pd.DataFrame: Indexed by 'date' (Eastern Time), columns=['open', 'high', 'low', 'close', 'volume', 'vwap']
    """
    
    # Polygon expects YYYY-MM-DD or millis. Since we want precise range, we use millis if possible,
    # but the v2/aggs endpoint path uses YYYY-MM-DD. 
    # Actually, Polygon v2/aggs docs say: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    # from/to can be YYYY-MM-DD or Unix MS Timestamp.
    
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_ms}/{end_ms}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": limit,
        "apiKey": API_KEY,
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error(f"Polygon API Error for {ticker}: {e}")
        return pd.DataFrame()

    results = data.get("results", [])
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    
    # Rename Polygon columns to standard
    # v: volume, vw: vwap, o: open, c: close, h: high, l: low, t: timestamp (ms), n: distinct_transactions
    col_map = {
        "v": "volume",
        "vw": "vwap",
        "o": "open",
        "c": "close",
        "h": "high",
        "l": "low",
        "t": "date",
        "n": "transactions"
    }
    df = df.rename(columns=col_map)
    
    # Process Timestamp
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    
    # Convert to Eastern Time (standard for this system)
    df = df.set_index("date").sort_index()
    df.index = df.index.tz_convert(EASTERN)
    
    # Filter columns
    return df[["open", "high", "low", "close", "volume"]]


# ------------------------------------------------------------------------------
# CSV PERSISTENCE
# ------------------------------------------------------------------------------
DATA_DIR = "data/ticker_data"

def _save_to_csv(symbol: str, new_df: pd.DataFrame):
    """
    Save or append data to CSV. 
    Matches index (datetime) to avoid duplicates.
    """
    if new_df.empty:
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    
    # Clean up new_df index name for saving
    df_to_save = new_df.copy()
    if df_to_save.index.name != "time":
        df_to_save.index.name = "time"
        
    if os.path.exists(file_path):
        try:
            # Read existing
            existing_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if existing_df.index.tz is None:
                # Naive to UTC to Eastern (assuming file saved as eastern ISO)
                # But to_csv usually saves ISO with offset if present.
                # If read back as naive, localize to Eastern to match new_df
                existing_df.index = existing_df.index.tz_localize(EASTERN)
            else:
                 existing_df.index = existing_df.index.tz_convert(EASTERN)

            # Combine
            combined = pd.concat([existing_df, df_to_save])
            # Drop duplicates on index, keep last
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            combined.to_csv(file_path)
            
        except Exception as e:
            log.error(f"[CSV Error] Failed to append {symbol}: {e}")
            # Fallback: Just overwrite if append fails? Or skip?
            # Let's try to overwrite if read failed (maybe corrupt)
            # df_to_save.to_csv(file_path) 
    else:
        # Create new
        df_to_save.to_csv(file_path)


def fetch_many_symbols(
    symbols: Iterable[str],
    start_utc: str,
    end_utc: str,
    bar_size: str = "15 mins", # Ignored, hardcoded to 15m for now as per requirement coverage
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    mimics Import_IBKR_multi.fetch_many_symbols but uses Polygon.
    """
    store = {}
    
    # Parse bar size roughly
    # We assume '15 mins' for now based on system use.
    multiplier = 15
    timespan = "minute"
    
    if "day" in bar_size.lower():
        multiplier = 1
        timespan = "day"
    elif "hour" in bar_size.lower():
        multiplier = 1
        timespan = "hour"
        
    t0 = time.time()
    if verbose:
        print(f"=== POLYGON BULK FETCH: {len(list(symbols))} symbols | {start_utc} -> {end_utc} ===")

    for i, sym in enumerate(symbols, 1):
        if verbose:
            print(f"[{i}] {sym} ...", end="\r")
            
        df = get_aggs(sym, multiplier, timespan, start_utc, end_utc)
        
        # SAVE TO CSV (Full OHLCV)
        if not df.empty:
            _save_to_csv(sym, df)
        
        # Format for engine: needs column name == symbol
        if not df.empty:
            df_engine = df[["close"]].rename(columns={"close": sym})
            store[sym] = df_engine
        else:
            store[sym] = pd.DataFrame()
            
        # Rate limit basic sleep (5 calls/min on free? User provided a key so assume standard/premium)
        # We'll do a tiny sleep to be safe.
        time.sleep(0.05)

    if verbose:
        print(f"\nDone in {time.time() - t0:.1f}s")
        
    return store

def fetch_latest_closed_15m_close(
    symbol: str, 
    lookback_days: int = 2
) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    """
    Fetches the latest completed 15m bar close.
    """
    end = datetime.now(UTC)
    start = end - timedelta(days=lookback_days)
    
    df = get_aggs(symbol, 15, "minute", start, end)
    
    if df.empty:
        return None, None
    
    # SAVE TO CSV
    _save_to_csv(symbol, df)
        
    # Get last row
    last_ts = df.index[-1]
    last_close = float(df["close"].iloc[-1])
    
    return last_ts, last_close


def fetch_since_timestamp(
    symbol: str, 
    since_ts: pd.Timestamp,
    multiplier: int = 15,
) -> pd.DataFrame:
    """
    Fetches all bars from `since_ts` until NOW.
    Used for gap filling.
    
    Args:
        symbol: Ticker symbol
        since_ts: Start timestamp (inclusive)
        multiplier: Bar size in minutes (default 15)
    
    Returns:
        DataFrame indexed by bar close time (UTC normalized to Eastern)
    """
    end = datetime.now(UTC)
    
    # Check if since_ts is timezone aware
    if since_ts.tzinfo is None:
        since_ts = since_ts.tz_localize(UTC)
    else:
        since_ts = since_ts.astimezone(UTC)
        
    start = since_ts # Query from last known time
    # Polygon range is inclusive, so we might get the last known one again. 
    # Logic in get_aggs handles start/end.
    # We rely on unique index de-duplication in _save_to_csv and engine cache.

    df = get_aggs(symbol, multiplier, "minute", start, end)
    
    if not df.empty:
        _save_to_csv(symbol, df)
        
    return df


# ------------------------------------------------------------------------------
# 1-MINUTE BAR FUNCTIONS
# ------------------------------------------------------------------------------
# Note: Polygon free tier has 15-minute delay. These functions return CLOSED bars only.
# Bar timestamps represent bar CLOSE time (end of interval), normalized to UTC.

def fetch_latest_closed_1m_close(
    symbol: str, 
    lookback_hours: int = 24
) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    """
    Fetches the latest completed 1-minute bar close.
    
    Returns:
        Tuple of (bar_close_ts, close_price) or (None, None) if no data.
        bar_close_ts is timezone-aware (Eastern time).
    """
    end = datetime.now(UTC)
    start = end - timedelta(hours=lookback_hours)
    
    df = get_aggs(symbol, 1, "minute", start, end)
    
    if df.empty:
        return None, None
        
    # Get last row (most recent closed bar)
    last_ts = df.index[-1]
    last_close = float(df["close"].iloc[-1])
    
    return last_ts, last_close

def fetch_latest_closed_1m_bar(
    symbol: str, 
    lookback_hours: int = 24
) -> Optional[Dict[str, Any]]:
    """
    Fetches the latest completed 1-minute bar with full OHLCV data.
    
    Returns:
        Dict with keys: ts, open, high, low, close, volume
        Or None if no data available.
    """
    end = datetime.now(UTC)
    start = end - timedelta(hours=lookback_hours)
    
    df = get_aggs(symbol, 1, "minute", start, end)
    
    if df.empty:
        return None
    
    row = df.iloc[-1]
    return {
        "ts": df.index[-1],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0
    }



def fetch_since_timestamp_1m(
    symbol: str, 
    since_ts: pd.Timestamp
) -> pd.DataFrame:
    """
    Fetches all 1-minute bars from `since_ts` until NOW.
    Used for gap filling in 1-minute mode.
    
    Args:
        symbol: Ticker symbol
        since_ts: Start timestamp (inclusive), should be timezone-aware
    
    Returns:
        DataFrame with columns [open, high, low, close, volume]
        Indexed by bar close time (Eastern timezone)
    """
    return fetch_since_timestamp(symbol, since_ts, multiplier=1)


def fetch_many_symbols_1m(
    symbols: Iterable[str],
    start_utc: str,
    end_utc: str,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Bulk fetch 1-minute bars for multiple symbols.
    
    Returns:
        Dict mapping symbol -> DataFrame with single 'close' column renamed to symbol
    """
    store = {}
    
    t0 = time.time()
    if verbose:
        print(f"=== POLYGON 1M FETCH: {len(list(symbols))} symbols | {start_utc} -> {end_utc} ===")

    for i, sym in enumerate(symbols, 1):
        if verbose:
            print(f"[{i}] {sym} ...", end="\r")
            
        df = get_aggs(sym, 1, "minute", start_utc, end_utc)
        
        # Format for engine: needs column name == symbol
        if not df.empty:
            df_engine = df[["close"]].rename(columns={"close": sym})
            store[sym] = df_engine
        else:
            store[sym] = pd.DataFrame()
            
        # Rate limit
        time.sleep(0.05)

    if verbose:
        print(f"\nDone in {time.time() - t0:.1f}s")
        
    return store

