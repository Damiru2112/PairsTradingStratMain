# run_live_realtime.py
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from db import connect_db, init_db, read_pair_params
from persist import persist_state

from Import_IBKR_multi import (
    IBKRConfig,
    connect_ibkr,
    unique_symbols_from_pairs,
    fetch_many_symbols,   # warmup only
    build_pair_frames,
)

from utils.pairs import parse_pairs
from price_cache import PriceCache
from live_metrics import build_live_metrics_table
from portfolio import PaperPortfolio

from strategy.live_trader import process_pair_barclose

# ib_insync realtime bars
from ib_insync import Stock, util


# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("run_live_realtime")


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "data/live.db")

# Pull enabled pairs from DB (Streamlit controls enabled/disabled)
RAW_PAIRS = None  # keep None to run from DB

# Warmup history (startup only). This can be small.
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "14"))
WARMUP_BAR_SIZE = os.getenv("WARMUP_BAR_SIZE", "15 mins")  # warmup only

# Optional: limit symbols while debugging (e.g. WARMUP_MAX_SYMBOLS=6)
WARMUP_MAX_SYMBOLS = int(os.getenv("WARMUP_MAX_SYMBOLS", "0"))

# Realtime bars are always 5s; we aggregate to 1m closes for strategy.
AGG_MINUTES = int(os.getenv("AGG_MINUTES", "1"))  # 1 = 1-min cadence, 5 = 5-min cadence, etc.

# Persist throttling: write to SQLite at most every N seconds (even if many bar events come in)
PERSIST_MIN_SECONDS = int(os.getenv("PERSIST_MIN_SECONDS", "30"))

RUN_ONCE = os.getenv("RUN_ONCE") == "1"


# ------------------------------------------------------------
# PORTFOLIO
# ------------------------------------------------------------
portfolio = PaperPortfolio(starting_equity=100_000)


def _fmt_pair(a: str, b: str) -> str:
    return f"{a}-{b}"


def _load_pairs_from_db(con) -> list[str]:
    df = read_pair_params(con)
    if df is None or df.empty:
        return []
    df = df.copy()
    df["enabled"] = df["enabled"].astype(int)
    return df[df["enabled"] == 1]["pair"].astype(str).tolist()


def _params_map_from_db(con) -> dict[str, dict]:
    df = read_pair_params(con)
    if df is None or df.empty:
        return {}
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        out[str(row["pair"])] = {
            "z_entry": float(row["z_entry"]),
            "z_exit": float(row["z_exit"]),
            "max_drift_pct": float(row["max_drift_pct"]),
            "alloc_pct": float(row["alloc_pct"]),
            "enabled": int(row["enabled"]),
        }
    return out


@dataclass
class MinuteBucket:
    """Tracks aggregation state for a single symbol."""
    cur_bucket_start: Optional[pd.Timestamp] = None
    last_close: Optional[float] = None


def _bucket_start(ts_utc: datetime, minutes: int) -> pd.Timestamp:
    """
    Map a bar timestamp to an aggregation bucket start, in UTC, aligned to `minutes`.
    """
    t = pd.Timestamp(ts_utc, tz="UTC")
    # Align to minute boundary first
    t = t.floor("min")
    if minutes == 1:
        return t
    # Align to N-minute boundary
    m = (t.minute // minutes) * minutes
    return t.replace(minute=m, second=0, microsecond=0)


def main() -> None:
    # ------------------------------------------------------------
    # DB CONNECT + INIT
    # ------------------------------------------------------------
    con = connect_db(DB_PATH)
    init_db(con)
    log.info("✅ DB ready: %s", os.path.abspath(DB_PATH))

    # ------------------------------------------------------------
    # PAIRS
    # ------------------------------------------------------------
    if RAW_PAIRS is None:
        raw_pairs = _load_pairs_from_db(con)
        if not raw_pairs:
            log.error("pair_params empty or no enabled pairs. Run: python seed_params.py")
            return
    else:
        raw_pairs = RAW_PAIRS

    PAIRS = parse_pairs(raw_pairs)
    if not PAIRS:
        log.error("No pairs parsed. Exiting.")
        return

    symbols = unique_symbols_from_pairs(PAIRS)
    if WARMUP_MAX_SYMBOLS > 0:
        symbols = symbols[:WARMUP_MAX_SYMBOLS]
        log.warning("WARMUP_MAX_SYMBOLS=%d -> limiting symbols to: %s", WARMUP_MAX_SYMBOLS, symbols)

    log.info("Pairs (%d): %s", len(raw_pairs), raw_pairs)
    log.info("Symbols (%d): %s", len(symbols), symbols)

    # ------------------------------------------------------------
    # IBKR CONNECT
    # ------------------------------------------------------------
    cfg = IBKRConfig(
        client_id=int(os.getenv("IB_CLIENT_ID", "70")),
        sleep_seconds=float(os.getenv("IB_SLEEP_S", "0.8")),
        timeout=int(os.getenv("IB_TIMEOUT", "60")),
    )

    log.info("Connecting to IBKR...")
    ib = connect_ibkr(cfg)
    log.info("✅ Connected to IBKR")

    # ------------------------------------------------------------
    # Cache + state
    # ------------------------------------------------------------
    cache = PriceCache()
    last_closed_count = 0
    last_persist_ts = 0.0

    # Aggregation state per symbol
    buckets: Dict[str, MinuteBucket] = {s: MinuteBucket() for s in symbols}

    # Realtime bar subscriptions map (symbol -> RealTimeBarList)
    rtb_streams = {}

    def run_strategy_and_persist(event_ts: pd.Timestamp) -> None:
        """
        This is the old 'while loop body' minus the historical polling and sleep.
        It runs whenever our aggregator emits a new aggregated close.
        """
        nonlocal last_closed_count, last_persist_ts

        # 1) build pair frames from cache
        symbol_frames_live = {sym: cache.data[sym].to_frame() for sym in symbols if sym in cache.data}
        pair_frames = build_pair_frames(symbol_frames_live, PAIRS, how="inner")
        if not pair_frames:
            log.info("No pair_frames yet (not enough aligned data).")
            return

        # 2) params from DB each run (Streamlit edits apply immediately)
        params_map = _params_map_from_db(con)

        # 3) trade decisions
        traded = 0
        skipped = 0
        for (a, b), df in pair_frames.items():
            pair_name = _fmt_pair(a, b)
            p = params_map.get(pair_name)
            if p is None or int(p["enabled"]) != 1:
                skipped += 1
                continue

            try:
                process_pair_barclose(
                    portfolio,
                    df,
                    a,
                    b,
                    z_entry=float(p["z_entry"]),
                    z_exit=float(p["z_exit"]),
                    max_drift_pct=float(p["max_drift_pct"]),
                    alloc_pct=float(p["alloc_pct"]),
                )
                traded += 1
            except Exception as e:
                log.exception("process_pair_barclose failed for %s: %s", pair_name, e)

        # 4) metrics
        metrics = build_live_metrics_table(pair_frames)

        # 5) latest prices for unrealized
        latest_prices = {}
        for sym in symbols:
            try:
                latest_prices[sym] = float(cache.data[sym].iloc[-1])
            except Exception:
                pass

        # 6) heartbeat
        equity = portfolio.starting_equity
        pnl_df = portfolio.pnl_df()
        if pnl_df is not None and not pnl_df.empty and "equity" in pnl_df.columns:
            equity = float(pnl_df.iloc[-1]["equity"])

        log.info(
            "💓 %s | traded=%d skipped=%d | metrics_rows=%d | open=%d | closed=%d | equity=%.2f",
            event_ts.isoformat(),
            traded,
            skipped,
            0 if metrics is None else len(metrics),
            len(portfolio.open_positions_df()),
            len(portfolio.closed_trades_df()),
            equity,
        )

        # 7) persist (throttled)
        now = time.time()
        if now - last_persist_ts < PERSIST_MIN_SECONDS:
            return

        closed_all = portfolio.closed_trades_df()
        new_closed = (
            closed_all.iloc[last_closed_count:].copy()
            if (closed_all is not None and not closed_all.empty)
            else closed_all
        )

        persist_state(con, metrics, portfolio, new_closed)
        last_closed_count = 0 if closed_all is None else len(closed_all)
        last_persist_ts = now
        log.info("✅ Persisted to SQLite")

        if RUN_ONCE:
            log.info("RUN_ONCE=1 -> exiting after first strategy run.")
            # Stop event loop safely
            ib.disconnect()

    def on_realtime_bar(sym: str, bar) -> None:
        """
        Called on every 5-second realtime bar for a symbol.
        Aggregates into AGG_MINUTES close and then triggers strategy.
        """
        try:
            b = buckets[sym]

            # bar.time is a datetime (UTC) in ib_insync
            bucket_start = _bucket_start(bar.time, AGG_MINUTES)
            close = float(bar.close)

            if b.cur_bucket_start is None:
                # first bar seen
                b.cur_bucket_start = bucket_start
                b.last_close = close
                return

            if bucket_start == b.cur_bucket_start:
                # still within same bucket, update last close
                b.last_close = close
                return

            # bucket advanced -> emit previous bucket close
            prev_bucket_ts = b.cur_bucket_start
            prev_close = b.last_close

            if prev_bucket_ts is not None and prev_close is not None:
                # feed aggregated close into your existing cache updater
                cache.update_close(sym, prev_bucket_ts, float(prev_close))

                # trigger strategy run on any aggregated close (inner-join will handle alignment)
                run_strategy_and_persist(prev_bucket_ts)

            # start new bucket
            b.cur_bucket_start = bucket_start
            b.last_close = close

        except Exception as e:
            log.exception("on_realtime_bar error for %s: %s", sym, e)

    try:
        # ------------------------------------------------------------
        # Warmup (startup only)
        # ------------------------------------------------------------
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=HISTORY_DAYS)

        log.info("Warmup: fetching %d days history (%s -> %s)", HISTORY_DAYS, start.isoformat(), end.isoformat())
        t0 = time.time()

        start_str = start.strftime("%Y%m%d-%H:%M:%S")
        end_str = end.strftime("%Y%m%d-%H:%M:%S")

        symbol_frames = fetch_many_symbols(
            ib=ib,
            symbols=symbols,
            start_utc=start_str,
            end_utc=end_str,
            cfg=cfg,
            bar_size=WARMUP_BAR_SIZE,     # warmup only
            what_to_show="TRADES",
            use_rth=True,
            max_retries=3,
        )

        ok = 0
        for sym in symbols:
            df = symbol_frames.get(sym)
            if df is None or df.empty:
                log.warning("Warmup empty for %s", sym)
                continue
            cache.seed(sym, df)
            ok += 1

        log.info("✅ Warmup done: %d/%d symbols seeded in %.1fs", ok, len(symbols), time.time() - t0)

        # ------------------------------------------------------------
        # Realtime subscriptions (once)
        # ------------------------------------------------------------
        # NOTE: util.startLoop() is for notebooks. In scripts it is harmless but not required.
        # We keep it to avoid event-loop issues in some environments.
        util.startLoop()

        for sym in symbols:
            c = Stock(sym, "SMART", "USD")
            stream = ib.reqRealTimeBars(
                c,
                barSize=5,
                whatToShow="TRADES",
                useRTH=True,
            )
            rtb_streams[sym] = stream

            # Attach callback
            stream.updateEvent += (lambda bars, sym=sym: on_realtime_bar(sym, bars[-1]))

        log.info("✅ Subscribed reqRealTimeBars(5s) for %d symbols. Aggregating to %d-min closes.",
                 len(symbols), AGG_MINUTES)

        if RUN_ONCE:
            log.info("RUN_ONCE=1 -> waiting for first aggregated close then exiting.")

        # Keep process alive
        ib.run()

    finally:
        try:
            for s, stream in rtb_streams.items():
                try:
                    ib.cancelRealTimeBars(stream)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            ib.disconnect()
        except Exception:
            pass
        try:
            con.close()
        except Exception:
            pass
        log.info("Stopped.")


if __name__ == "__main__":
    main()
