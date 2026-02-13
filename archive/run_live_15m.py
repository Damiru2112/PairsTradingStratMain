from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from db import connect_db, init_db, read_pair_params
from persist import persist_state

from Import_IBKR_multi import (
    IBKRConfig,
    connect_ibkr,
    unique_symbols_from_pairs,
    fetch_many_symbols,   # now has progress + retry + pacing
    build_pair_frames,
)
from live_metrics import build_pair_series_table  # add import
from utils.pairs import parse_pairs
from data_live import fetch_latest_closed_15m_close
from price_cache import PriceCache
from live_metrics import build_live_metrics_table
from portfolio import PaperPortfolio
from time_utils import sleep_until_next_15m_close

from strategy.live_trader import process_pair_barclose


# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("run_live_15m")


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "data/live.db")

# Pull enabled pairs from DB (Streamlit controls enabled/disabled)
RAW_PAIRS = None  # keep None to run from DB

# Reduced default warmup days for speed while debugging
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "14"))

# Optional: limit warmup symbols while debugging (e.g. WARMUP_MAX_SYMBOLS=6)
WARMUP_MAX_SYMBOLS = int(os.getenv("WARMUP_MAX_SYMBOLS", "0"))

RUN_ONCE = os.getenv("RUN_ONCE") == "1"



# Build historical pair frames from warmup cache


# ------------------------------------------------------------
# PORTFOLIO
# ------------------------------------------------------------
portfolio = PaperPortfolio(starting_equity=100_000)


def _fmt_pair(a: str, b: str) -> str:
    return f"{a}-{b}"


def _load_pairs_from_db(con) -> list[str]:
    """
    Pull enabled pairs from pair_params.
    """
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


def main() -> None:
    # ------------------------------------------------------------
    # DB CONNECT + INIT
    # ------------------------------------------------------------
    con = connect_db(DB_PATH)
    init_db(con)
    log.info("✅ DB ready: %s", os.path.abspath(DB_PATH))

    # ------------------------------------------------------------
    # PAIRS (from DB unless RAW_PAIRS provided)
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
        log.warning("WARMUP_MAX_SYMBOLS=%d -> limiting warmup to: %s", WARMUP_MAX_SYMBOLS, symbols)

    log.info("Pairs (%d): %s", len(raw_pairs), raw_pairs)
    log.info("Symbols (%d): %s", len(symbols), symbols)

    # ------------------------------------------------------------
    # IBKR CONNECT
    # ------------------------------------------------------------
    # Paper TWS -> 7497, Live TWS -> 7496
    cfg = IBKRConfig(
        client_id=int(os.getenv("IB_CLIENT_ID", "70")),
        sleep_seconds=float(os.getenv("IB_SLEEP_S", "0.8")),
        timeout=int(os.getenv("IB_TIMEOUT", "60")),
    )

    log.info("Connecting to IBKR...")
    ib = connect_ibkr(cfg)
    log.info("✅ Connected to IBKR")

    last_closed_count = 0

    try:
        # ------------------------------------------------------------
        # 1) Seed history (warmup)
        # ------------------------------------------------------------
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=HISTORY_DAYS)

        log.info("Warmup: fetching %d days history (%s -> %s)", HISTORY_DAYS, start.isoformat(), end.isoformat())
        t0 = time.time()

        # IB wants UTC format: yyyymmdd-hh:mm:ss
        start_str = start.strftime("%Y%m%d-%H:%M:%S")
        end_str = end.strftime("%Y%m%d-%H:%M:%S")

        symbol_frames = fetch_many_symbols(
            ib=ib,
            symbols=symbols,
            start_utc=start_str,
            end_utc=end_str,
            cfg=cfg,
            bar_size="15 mins",
            what_to_show="TRADES",
            use_rth=True,
            max_retries=3,
        )

        cache = PriceCache()
        ok = 0
        for sym in symbols:
            df = symbol_frames.get(sym)
            if df is None or df.empty:
                log.warning("Warmup empty for %s", sym)
                continue
            cache.seed(sym, df)
            ok += 1

        log.info("✅ Warmup done: %d/%d symbols seeded in %.1fs", ok, len(symbols), time.time() - t0)

        symbol_frames_live = {sym: cache.data[sym].to_frame() for sym in symbols if sym in cache.data}
        pair_frames_hist = build_pair_frames(symbol_frames_live, PAIRS, how="inner")

        # Seed pair_series with full history (so UI has real lines immediately)
        series_df = build_pair_series_table(pair_frames_hist)
        log.info("Seeding pair_series with %d rows of history...", len(series_df))
        from db import write_pair_series
        write_pair_series(con, series_df)
        log.info("✅ pair_series seeded")


        # ------------------------------------------------------------
        # Loop (bar-close driven)
        # ------------------------------------------------------------
        loop_i = 0
        while True:
            loop_i += 1
            log.info("🔁 Loop #%d | updating latest closed 15m bars...", loop_i)

            # 2) update latest closed bar for each symbol
            updated = 0
            for i, sym in enumerate(symbols, start=1):
                try:
                    t, close = fetch_latest_closed_15m_close(ib, sym)
                    cache.update_close(sym, t, close)
                    updated += 1
                except Exception as e:
                    log.exception("Failed latest close for %s (%d/%d): %s", sym, i, len(symbols), e)

            log.info("✅ Updated %d/%d symbols", updated, len(symbols))

            # 3) build pair frames from cache
            symbol_frames_live = {sym: cache.data[sym].to_frame() for sym in symbols if sym in cache.data}
            pair_frames = build_pair_frames(symbol_frames_live, PAIRS, how="inner")

            if not pair_frames:
                log.warning("No pair_frames yet (not enough aligned data).")
                if RUN_ONCE:
                    break
                sleep_until_next_15m_close(buffer_s=8)
                continue

            # 4) params from DB each loop (Streamlit edits take effect immediately)
            params_map = _params_map_from_db(con)

            # 4.5) trade decisions
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

            log.info("Trading decisions: ran=%d skipped=%d", traded, skipped)

            # 5) metrics
            metrics = build_live_metrics_table(pair_frames)

            # 6) latest prices for unrealized
            latest_prices = {}
            for sym in symbols:
                try:
                    latest_prices[sym] = float(cache.data[sym].iloc[-1])
                except Exception:
                    pass

            # 7) heartbeat
            equity = portfolio.starting_equity
            pnl_df = portfolio.pnl_df()
            if pnl_df is not None and not pnl_df.empty and "equity" in pnl_df.columns:
                equity = float(pnl_df.iloc[-1]["equity"])

            log.info(
                "💓 Heartbeat | metrics_rows=%d | open_pos=%d | closed=%d | equity=%.2f",
                0 if metrics is None else len(metrics),
                len(portfolio.open_positions_df()),
                len(portfolio.closed_trades_df()),
                equity,
            )

            # Print tables occasionally
            if loop_i == 1 or loop_i % 4 == 0:
                print("\nLIVE METRICS\n", metrics)
                print("\nOPEN POSITIONS\n", portfolio.open_positions_df())
                print("\nPNL SUMMARY\n", portfolio.pnl_df())
                print("\nUNREALIZED\n", portfolio.mark_to_market_unrealized(latest_prices))

                closed = portfolio.closed_trades_df()
                if not closed.empty:
                    print("\nCLOSED TRADES (last 5)\n", closed.tail(5))

            # 8) persist
            closed_all = portfolio.closed_trades_df()
            new_closed = (
                closed_all.iloc[last_closed_count:].copy()
                if (closed_all is not None and not closed_all.empty)
                else closed_all
            )

            persist_state(con, metrics, portfolio, new_closed)
            last_closed_count = 0 if closed_all is None else len(closed_all)
            log.info("✅ Persisted to SQLite")

            if RUN_ONCE:
                log.info("RUN_ONCE=1 -> exiting after one loop.")
                break

            log.info("Sleeping until next 15m close...")
            sleep_until_next_15m_close(buffer_s=8)

    finally:
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
