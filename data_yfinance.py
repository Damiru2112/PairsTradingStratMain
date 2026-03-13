"""
data_yfinance.py

Yahoo Finance data fetcher for Japanese equity pairs.
Uses yfinance library for historical and near-real-time 1m OHLCV data.

Key features:
- Linear interpolation for missing bars (yfinance gaps)
- Mid-day break awareness (11:30-12:30 JST — no bars expected)
- Gap detection: too many consecutive missing bars triggers retry/warning
- All data returned with 15-minute delay buffer so interpolation can settle

JP tickers on yfinance use the .T suffix (e.g. 7203.T for Toyota).
"""
from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

log = logging.getLogger("data_yfinance")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_DELAY_S = 5.0
# If more than this fraction of bars are missing, flag as bad data
MAX_GAP_FRACTION = 0.30  # 30%
# Consecutive missing bars threshold before warning
MAX_CONSECUTIVE_GAPS = 10


# ---------------------------------------------------------------------------
# JP MARKET HOURS HELPERS
# ---------------------------------------------------------------------------

def _is_jp_trading_minute(ts: pd.Timestamp) -> bool:
    """
    Check if a given timestamp (UTC or JST) falls within TSE trading hours.
    Morning: 09:00–11:30 JST, Afternoon: 12:30–15:00 JST.
    The mid-day break 11:30–12:30 is NOT trading time.
    """
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    jst = ts.tz_convert("Asia/Tokyo")

    if jst.weekday() >= 5:  # Weekend
        return False

    hour, minute = jst.hour, jst.minute
    t = hour * 60 + minute  # minutes since midnight

    morning_open = 9 * 60       # 09:00
    morning_close = 11 * 60 + 30  # 11:30
    afternoon_open = 12 * 60 + 30  # 12:30
    afternoon_close = 15 * 60    # 15:00

    return (morning_open <= t < morning_close) or (afternoon_open <= t < afternoon_close)


def _build_expected_1m_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Build the expected 1-minute bar timestamps between start and end,
    excluding mid-day break (11:30–12:30 JST) and outside market hours.
    """
    full_range = pd.date_range(start=start, end=end, freq="1min")
    # Filter to trading minutes only
    mask = [_is_jp_trading_minute(ts) for ts in full_range]
    return full_range[mask]


# ---------------------------------------------------------------------------
# INTERPOLATION & GAP HANDLING
# ---------------------------------------------------------------------------

def _interpolate_gaps(df: pd.DataFrame, max_consecutive: int = MAX_CONSECUTIVE_GAPS) -> Tuple[pd.DataFrame, int, int]:
    """
    Linearly interpolate missing bars in a 1m OHLCV DataFrame.

    For close/open/high/low: linear interpolation.
    For volume: fill with 0 (no volume during gap).

    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns.
        max_consecutive: Max consecutive NaN bars before flagging.

    Returns:
        (interpolated_df, total_gaps_filled, max_consecutive_gap_found)
    """
    if df.empty:
        return df, 0, 0

    # Count gaps before interpolation
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if not price_cols:
        return df, 0, 0

    # Use 'close' as the primary gap indicator
    is_gap = df["close"].isna() if "close" in df.columns else pd.Series(False, index=df.index)
    total_gaps = int(is_gap.sum())

    # Find max consecutive gap run
    max_consec = 0
    current_run = 0
    for val in is_gap:
        if val:
            current_run += 1
            max_consec = max(max_consec, current_run)
        else:
            current_run = 0

    # Interpolate price columns linearly
    out = df.copy()
    for col in price_cols:
        if col in out.columns:
            out[col] = out[col].interpolate(method="linear", limit_direction="forward")

    # Fill volume with 0
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0).astype(int)

    # Forward-fill any remaining edge NaNs (start of series)
    out = out.ffill().bfill()

    return out, total_gaps, max_consec


def _reindex_to_trading_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex a DataFrame to include all expected trading minutes,
    inserting NaN rows for missing bars. Excludes mid-day break.
    """
    if df.empty:
        return df

    expected_idx = _build_expected_1m_index(df.index.min(), df.index.max())
    # Only keep expected minutes that are within the data range
    return df.reindex(expected_idx)


# ---------------------------------------------------------------------------
# FETCH WITH RETRY + INTERPOLATION
# ---------------------------------------------------------------------------

def _fetch_single_symbol(
    sym: str,
    period: str = "5d",
    interval: str = "1m",
    start: str = None,
    end: str = None,
    retries: int = MAX_RETRIES,
) -> Optional[pd.DataFrame]:
    """
    Fetch data for a single symbol with retry logic.
    Returns cleaned DataFrame or None on total failure.
    """
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(sym)
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
            else:
                df = ticker.history(period=period, interval=interval, auto_adjust=True)

            if df.empty:
                log.warning(f"[{sym}] Empty response (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    _time.sleep(RETRY_DELAY_S)
                continue

            # Normalize
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep]

            # Drop exact duplicate timestamps
            df = df[~df.index.duplicated(keep="last")]

            return df

        except Exception as e:
            log.error(f"[{sym}] Fetch error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                _time.sleep(RETRY_DELAY_S)

    log.error(f"[{sym}] All {retries} fetch attempts failed.")
    return None


def _clean_and_interpolate(
    df: pd.DataFrame,
    sym: str,
) -> Tuple[Optional[pd.DataFrame], dict]:
    """
    Reindex to trading minutes, interpolate gaps, check quality.

    Returns:
        (cleaned_df or None if too bad, stats_dict)
    """
    if df is None or df.empty:
        return None, {"total_bars": 0, "gaps_filled": 0, "max_consecutive_gap": 0, "quality": "empty"}

    # Reindex to expected trading minutes (adds NaN for missing bars)
    df_reindexed = _reindex_to_trading_minutes(df)

    total_expected = len(df_reindexed)
    if total_expected == 0:
        return None, {"total_bars": 0, "gaps_filled": 0, "max_consecutive_gap": 0, "quality": "empty"}

    # Interpolate
    df_clean, gaps_filled, max_consec = _interpolate_gaps(df_reindexed)

    gap_fraction = gaps_filled / total_expected if total_expected > 0 else 0

    stats = {
        "total_bars": total_expected,
        "gaps_filled": gaps_filled,
        "gap_fraction": round(gap_fraction, 4),
        "max_consecutive_gap": max_consec,
        "quality": "good",
    }

    if gap_fraction > MAX_GAP_FRACTION:
        stats["quality"] = "bad"
        log.warning(f"[{sym}] Bad data quality: {gaps_filled}/{total_expected} bars missing "
                    f"({gap_fraction:.1%}), max consecutive gap: {max_consec}")
    elif max_consec > MAX_CONSECUTIVE_GAPS:
        stats["quality"] = "warning"
        log.warning(f"[{sym}] Large gap: {max_consec} consecutive bars missing")
    else:
        log.info(f"[{sym}] Data OK: {gaps_filled} gaps interpolated out of {total_expected} bars")

    return df_clean, stats


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def fetch_many_symbols(
    symbols: List[str],
    days: int = 7,
    interval: str = "1m",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical OHLCV for multiple JP symbols via yfinance.
    Applies interpolation for missing bars and respects mid-day break.

    yfinance limits for 1m: last 7 days only.

    Returns: {symbol: DataFrame} with DatetimeIndex (UTC),
             columns [open, high, low, close, volume], gaps interpolated.
    """
    results = {}
    period = f"{min(days, 7)}d"  # yfinance caps 1m at 7 days

    for sym in symbols:
        raw = _fetch_single_symbol(sym, period=period, interval=interval)
        if raw is None:
            continue

        cleaned, stats = _clean_and_interpolate(raw, sym)

        if cleaned is not None and stats["quality"] != "bad":
            results[sym] = cleaned
        elif cleaned is not None and stats["quality"] == "bad":
            # Still return it but log the issue — engine can decide to skip
            log.warning(f"[{sym}] Returning bad-quality data ({stats['gap_fraction']:.1%} gaps)")
            results[sym] = cleaned

    return results


def fetch_many_symbols_15m(
    symbols: List[str],
    days: int = 60,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch 15m historical data for warmup/slow metrics.
    yfinance allows up to ~60 days of 15m data.
    Interpolation is lighter here since 15m gaps are less common.

    Returns: {symbol: DataFrame} with DatetimeIndex (UTC)
    """
    results = {}
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=min(days, 57))  # yfinance enforces ~60d; use 57 for safety margin

    for sym in symbols:
        raw = _fetch_single_symbol(
            sym,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="15m",
        )
        if raw is None:
            continue

        # Light interpolation for 15m (don't reindex to trading minutes)
        if not raw.empty:
            price_cols = [c for c in ["open", "high", "low", "close"] if c in raw.columns]
            for col in price_cols:
                raw[col] = raw[col].interpolate(method="linear", limit=4).ffill().bfill()
            if "volume" in raw.columns:
                raw["volume"] = raw["volume"].fillna(0).astype(int)

        results[sym] = raw
        log.info(f"Fetched {len(raw)} 15m bars for {sym}")

    return results


def fetch_latest_1m_bars(
    symbols: List[str],
    delay_minutes: int = 15,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch today's 1m bars for all symbols, applying a 15-minute delay.
    Only bars older than (now - delay_minutes) are returned,
    giving time for interpolation to settle.

    Returns: {symbol: DataFrame} with all 1m bars up to the delay cutoff,
             gaps interpolated.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=delay_minutes)
    results = {}

    for sym in symbols:
        raw = _fetch_single_symbol(sym, period="1d", interval="1m")
        if raw is None:
            continue

        # Apply delay: only keep bars before cutoff
        raw = raw[raw.index <= cutoff]
        if raw.empty:
            continue

        cleaned, stats = _clean_and_interpolate(raw, sym)
        if cleaned is not None:
            results[sym] = cleaned

    return results


def fetch_latest_close_delayed(
    symbols: List[str],
    delay_minutes: int = 15,
) -> Dict[str, float]:
    """
    Fetch the latest close price for each symbol, respecting a delay.
    Returns the close price of the bar at (now - delay_minutes).

    Returns: {symbol: close_price}
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=delay_minutes)
    prices = {}

    for sym in symbols:
        raw = _fetch_single_symbol(sym, period="1d", interval="1m")
        if raw is None:
            continue

        # Only bars before cutoff
        delayed = raw[raw.index <= cutoff]
        if delayed.empty:
            continue

        prices[sym] = float(delayed["close"].iloc[-1])

    return prices


def fetch_latest_1m_bar_delayed(
    symbols: List[str],
    delay_minutes: int = 15,
) -> Dict[str, dict]:
    """
    Fetch the latest 1m bar (full OHLCV) for each symbol, with delay.

    Returns: {symbol: {open, high, low, close, volume, time}}
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=delay_minutes)
    bars = {}

    for sym in symbols:
        raw = _fetch_single_symbol(sym, period="1d", interval="1m")
        if raw is None:
            continue

        delayed = raw[raw.index <= cutoff]
        if delayed.empty:
            continue

        # Interpolate this slice
        cleaned, _ = _clean_and_interpolate(delayed, sym)
        if cleaned is None or cleaned.empty:
            continue

        last = cleaned.iloc[-1]
        ts = cleaned.index[-1]

        bars[sym] = {
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "close": float(last["close"]),
            "volume": int(last.get("volume", 0)),
            "time": ts.isoformat(),
        }

    return bars


def check_data_health(symbols: List[str]) -> Dict[str, dict]:
    """
    Quick health check: fetch 1d of 1m data and report gap stats per symbol.
    Useful for diagnostics.

    Returns: {symbol: {total_bars, gaps_filled, gap_fraction, max_consecutive_gap, quality}}
    """
    report = {}
    for sym in symbols:
        raw = _fetch_single_symbol(sym, period="1d", interval="1m", retries=1)
        if raw is None:
            report[sym] = {"quality": "fetch_failed"}
            continue
        _, stats = _clean_and_interpolate(raw, sym)
        report[sym] = stats
    return report
