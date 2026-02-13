# US_realtime_engine.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
from ib_insync import Stock, util

from db import connect_db, init_db, read_pair_params
from utils.pairs import parse_pairs
from Import_IBKR_multi import IBKRConfig, connect_ibkr, unique_symbols_from_pairs


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("US_realtime_engine")


# ----------------------------
# Settings
# ----------------------------
DB_PATH = os.getenv("DB_PATH", "data/live.db")
USE_RTH = os.getenv("USE_RTH", "1") == "1"


def load_enabled_pairs_from_db(con) -> list[str]:
    """Return list like ['WMT-COST', 'KO-PEP'] for enabled pairs."""
    df = read_pair_params(con)
    if df is None or df.empty:
        return []
    df = df.copy()
    df["enabled"] = df["enabled"].astype(int)
    return df[df["enabled"] == 1]["pair"].astype(str).tolist()


@dataclass
class LastBar:
    ts_utc: Optional[pd.Timestamp] = None
    close: Optional[float] = None


def main() -> None:
    # 1) DB: load enabled pairs -> symbols
    con = connect_db(DB_PATH)
    init_db(con)

    raw_pairs = load_enabled_pairs_from_db(con)
    if not raw_pairs:
        log.error("No enabled pairs in DB. Run: python seed_params.py")
        return

    pairs = parse_pairs(raw_pairs)
    symbols = unique_symbols_from_pairs(pairs)

    log.info("Enabled pairs (%d): %s", len(raw_pairs), raw_pairs)
    log.info("Symbols (%d): %s", len(symbols), symbols)

    # 2) IBKR connect
    cfg = IBKRConfig(
        client_id=int(os.getenv("IB_CLIENT_ID", "70")),
        sleep_seconds=float(os.getenv("IB_SLEEP_S", "0.8")),
        timeout=int(os.getenv("IB_TIMEOUT", "60")),
    )

    log.info("Connecting to IBKR...")
    ib = connect_ibkr(cfg)
    log.info("✅ Connected to IBKR")

    # 3) Subscribe to 5s bars
    # util.startLoop() helps in some environments; safe to keep.
    util.startLoop()

    streams = {}
    last_seen: Dict[str, LastBar] = {s: LastBar() for s in symbols}

    def on_bar(sym: str, bar) -> None:
        """
        Called on every 5-second bar.
        bar.time is UTC datetime from ib_insync.
        """
        ts = pd.Timestamp(bar.time, tz="UTC")
        close = float(bar.close)

        last_seen[sym].ts_utc = ts
        last_seen[sym].close = close

        # Print a light heartbeat sometimes (not every bar)
        # Example: only log when seconds == 0 (minute boundary)
        if ts.second == 0:
            log.info("BAR %s | %s close=%.4f", sym, ts.isoformat(), close)

    for sym in symbols:
        contract = Stock(sym, "SMART", "USD")
        stream = ib.reqRealTimeBars(
            contract,
            barSize=5,
            whatToShow="TRADES",
            useRTH=USE_RTH,
        )
        streams[sym] = stream

        # IMPORTANT: bind sym default in lambda to avoid late-binding bug
        stream.updateEvent += (lambda bars, sym=sym: on_bar(sym, bars[-1]))

    log.info("✅ Subscribed to 5s realtime bars for %d symbols (USE_RTH=%s).", len(symbols), USE_RTH)

    # 4) Run forever
    try:
        ib.run()
    finally:
        # cleanup
        for sym, stream in streams.items():
            try:
                ib.cancelRealTimeBars(stream)
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
