"""
run_engine_jp.py

Japanese Pairs Trading Engine (1-Minute).
Mirrors run_engine_1m.py but uses yfinance for data and JP-specific DB tables.

Architecture:
- Fast Layer (1m): Fetches 1m bars via yfinance, computes spread, uses cached mean/std for z-score
- Slow Layer (15m): Refreshes when new 15m bar available, computes betas/drift

Usage:
    python run_engine_jp.py --engine-id JP_1M
"""
from __future__ import annotations

import argparse
import os
import time
import logging
import threading
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Set
from collections import deque

import pandas as pd

# Local imports
from db import connect_db, init_db, init_jp_db, pulse_heartbeat
import db
from data_yfinance import (
    fetch_many_symbols,
    fetch_many_symbols_15m,
    fetch_latest_close_delayed,
    fetch_latest_1m_bar_delayed,
    fetch_latest_1m_bars,
)
from telegram_notifier import notifier
from utils.pairs import parse_pairs, build_pair_frames, unique_symbols_from_pairs
from price_cache import PriceCache
from slow_metrics_cache import SlowMetricsCache
from live_metrics import build_live_metrics_table, build_pair_series_table
from portfolio_jp import PaperPortfolio
from strategy.live_trader import process_pair_barclose_fast
from jp_market_time import sleep_for_next_bar, sleep_short
from analytics.zscore import compute_zscore_30d_weekly
from analytics.betas import compute_direct_beta, compute_beta_30d_weekly

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("engine_jp")

# Config
DB_PATH = os.getenv("DB_PATH", "data/live.db")
HISTORY_DAYS = int(os.getenv("JP_HISTORY_DAYS", "60"))  # yfinance 15m limit ~60 days
JP_STARTING_EQUITY = float(os.getenv("JP_STARTING_EQUITY", "10000000"))  # ¥10M


# ------------------------------------------------------------------------------
# HEARTBEAT
# ------------------------------------------------------------------------------
class HeartbeatThread(threading.Thread):
    def __init__(self, db_path: str, engine_id: str, interval_s: float = 10.0):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.engine_id = engine_id
        self.interval_s = interval_s
        self.running = True
        self.status = "Starting"
        self.error: Optional[str] = None
        self.meta: Dict = {}

    def set_status(self, status: str, error: Optional[str] = None, meta: Optional[dict] = None):
        self.status = status
        if error:
            self.error = error
        if meta:
            self.meta.update(meta)

    def run(self):
        con = connect_db(self.db_path)
        try:
            while self.running:
                try:
                    pulse_heartbeat(con, self.engine_id, self.status, error=self.error, meta=self.meta)
                except Exception as e:
                    log.error(f"Heartbeat write failed: {e}")
                time.sleep(self.interval_s)
        finally:
            con.close()

    def stop(self):
        self.running = False


# ------------------------------------------------------------------------------
# JP MARKET HOURS
# ------------------------------------------------------------------------------
def is_jp_market_open() -> bool:
    """Check if TSE is currently in trading hours."""
    import pytz
    jst = pytz.timezone("Asia/Tokyo")
    now = datetime.now(jst)

    if now.weekday() >= 5:  # Weekend
        return False

    # Morning session: 09:00–11:30, Afternoon: 12:30–15:00 JST
    morning_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    morning_close = now.replace(hour=11, minute=30, second=0, microsecond=0)
    afternoon_open = now.replace(hour=12, minute=30, second=0, microsecond=0)
    afternoon_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

    return (morning_open <= now <= morning_close) or (afternoon_open <= now <= afternoon_close)


def seconds_until_jp_market_open() -> float:
    """
    Seconds until next TSE session open. Returns 0 if market is open.
    Handles mid-day break: if between 11:30-12:30, returns seconds until 12:30.
    """
    import pytz
    jst = pytz.timezone("Asia/Tokyo")
    now = datetime.now(jst)

    if is_jp_market_open():
        return 0.0

    t = now.hour * 60 + now.minute

    # During mid-day break (11:30 - 12:30) → wait for afternoon session
    if now.weekday() < 5 and 11 * 60 + 30 <= t < 12 * 60 + 30:
        afternoon_open = now.replace(hour=12, minute=30, second=0, microsecond=0)
        return max(0.0, (afternoon_open - now).total_seconds())

    # Before morning open on a weekday
    if now.weekday() < 5 and t < 9 * 60:
        morning_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        return max(0.0, (morning_open - now).total_seconds())

    # After 15:00 or weekend → next 09:00 JST
    next_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now >= next_open:
        next_open += timedelta(days=1)

    # Skip weekends
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)

    return max(0.0, (next_open - now).total_seconds())


# ------------------------------------------------------------------------------
# ENGINE
# ------------------------------------------------------------------------------
class JPTradingEngine:
    def __init__(self, engine_id: str):
        self.engine_id = engine_id

        # Database
        self.con = connect_db(DB_PATH)
        init_db(self.con)
        init_jp_db(self.con)

        # Caches
        self.cache_1m = PriceCache()
        self.cache_15m = PriceCache()
        self.slow_metrics = SlowMetricsCache()

        # State
        self.portfolio = PaperPortfolio(starting_equity=JP_STARTING_EQUITY)
        self.symbols: List[str] = []
        self.pairs: List[tuple] = []
        self.heartbeat_thread: Optional[HeartbeatThread] = None

        # Timestamp tracking
        self._last_processed_1m_ts: Dict[str, pd.Timestamp] = {}
        self._last_slow_bar_ts: Optional[pd.Timestamp] = None
        self._last_slow_refresh_wall: Optional[datetime] = None

        # Persistence filter: ring buffer of last 3 z-scores per pair
        self._z_history: Dict[str, deque] = {}

        # 1m bar cache
        self._bar_cache_1m: Dict[str, dict] = {}

        # Persistence tracking
        self._persisted_closed_count = 0

        # EOD State
        self._pnl_recorded_today = False
        self._last_equity_snapshot_ts: Optional[datetime] = None
        self._last_unrealized_pnl: float = 0.0

    def start(self):
        log.info(f"Starting JP Engine | ID: {self.engine_id}")
        notifier.notify(f"🇯🇵 JP Engine Starting | ID: {self.engine_id}")

        # Start Heartbeat
        self.heartbeat_thread = HeartbeatThread(DB_PATH, self.engine_id)
        self.heartbeat_thread.start()

        try:
            self._run_setup()
            self.heartbeat_thread.set_status("Running")
            self._run_polling_loop()
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        except Exception as e:
            log.exception("Fatal error in JP engine loop")
            notifier.notify(f"🇯🇵💀 Fatal JP Engine Error: {e}")
            self.heartbeat_thread.set_status("Error", error=str(e))
            time.sleep(5)
        finally:
            self.cleanup()

    def _load_params(self):
        """Load enabled JP pairs from DB."""
        df = db.jp_get_pair_params(self.con)
        if df is None or df.empty:
            log.warning("No JP pair parameters found in DB.")
            return [], []

        enabled = df[df["enabled"] == 1]
        raw_pairs = enabled["pair"].tolist()
        pairs_tuples = parse_pairs(raw_pairs)
        symbols = unique_symbols_from_pairs(pairs_tuples)
        return pairs_tuples, symbols

    def _params_as_dict(self) -> dict:
        df = db.jp_get_pair_params(self.con)
        out = {}
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                out[str(row["pair"])] = row.to_dict()
        return out

    def _run_setup(self):
        """Initial setup: load pairs, warmup caches, compute slow metrics."""

        # 1. Load Pairs
        self.pairs, self.symbols = self._load_params()
        if not self.symbols:
            log.error("No enabled JP symbols found. Exiting setup.")
            return

        # 2. Warmup 15m cache via yfinance
        self.heartbeat_thread.set_status("Warmup-15m")
        log.info(f"Warming up {len(self.symbols)} JP symbols (15m) for {HISTORY_DAYS} days...")

        frames_15m = fetch_many_symbols_15m(
            symbols=self.symbols,
            days=HISTORY_DAYS,
        )

        for sym, df in frames_15m.items():
            if not df.empty:
                self.cache_15m.seed(sym, df["close"])

        # 3. Compute initial slow metrics
        self.heartbeat_thread.set_status("Warmup-SlowMetrics")
        log.info("Computing initial JP slow metrics...")
        self._update_slow_metrics()

        if self.slow_metrics.is_empty():
            log.error("Failed to compute JP slow metrics. Check data.")
            notifier.notify("⚠️ JP slow metrics warmup failed")
        else:
            log.info(f"JP slow metrics computed for {len(self.slow_metrics.get_all_pairs())} pairs")

        # 4. Persist initial history to DB
        log.info("Persisting initial JP history to DB...")
        try:
            symbol_frames_warm = {s: self.cache_15m.data[s].to_frame() for s in self.symbols if s in self.cache_15m.data}
            if symbol_frames_warm:
                pair_frames_warm = build_pair_frames(symbol_frames_warm, self.pairs, how="inner")
                history_df = build_pair_series_table(pair_frames_warm)
                if not history_df.empty:
                    db.jp_save_history(self.con, history_df, timeframe="15m")
                    log.info(f"Persisted {len(history_df)} JP historical bars to DB.")
        except Exception as e:
            log.error(f"Failed to persist JP initial history: {e}")

        # 5. Restore positions from DB
        self._restore_positions_from_db()
        self._restore_pnl_from_db()

        # 6. Initial snapshot persist
        try:
            symbol_frames = {s: self.cache_15m.data[s].to_frame() for s in self.symbols if s in self.cache_15m.data}
            pair_frames = build_pair_frames(symbol_frames, self.pairs, how="inner")
            metrics_df = build_live_metrics_table(pair_frames)
            db.jp_save_snapshot(self.con, metrics_df)

            latest_prices = self.cache_15m.get_latest_prices()
            mtm_df = self.portfolio.mark_to_market_unrealized(latest_prices)
            unrealized_pnl = 0.0
            if not mtm_df.empty and "unrealized_pnl" in mtm_df.columns:
                unrealized_pnl = mtm_df["unrealized_pnl"].sum()
                self._last_unrealized_pnl = unrealized_pnl

            db.jp_save_pnl_snapshot(
                self.con, self.portfolio.equity(),
                self.portfolio.realized_pnl, unrealized_pnl,
                len(self.portfolio.positions), len(self.portfolio.closed),
            )
            log.info("JP initial snapshot persisted.")
        except Exception as e:
            log.error(f"Failed to persist JP initial snapshot: {e}")

        self.heartbeat_thread.set_status("Running")
        log.info("JP Engine setup complete.")

    def _update_slow_metrics(self):
        """Recompute 15m-based slow metrics (mean, std, betas)."""
        symbol_frames = {s: self.cache_15m.data[s].to_frame() for s in self.symbols if s in self.cache_15m.data}
        pair_frames = build_pair_frames(symbol_frames, self.pairs, how="inner")

        for (s1, s2), pf in pair_frames.items():
            try:
                pair_key = f"{s1}-{s2}"
                z_series = compute_zscore_30d_weekly(pf, s1, s2)
                if z_series is None or z_series.empty:
                    continue

                spread = pf[s1] / pf[s2]
                mean_series = spread.rolling(30 * 26).mean()
                std_series = spread.rolling(30 * 26).std()

                last_mean = mean_series.dropna().iloc[-1] if not mean_series.dropna().empty else None
                last_std = std_series.dropna().iloc[-1] if not std_series.dropna().empty else None

                if last_mean is not None and last_std is not None:
                    self.slow_metrics.update(pair_key, {
                        "mean": float(last_mean),
                        "std": float(last_std),
                        "z": float(z_series.dropna().iloc[-1]) if not z_series.dropna().empty else 0.0,
                    })
            except Exception as e:
                log.warning(f"Slow metrics failed for {s1}-{s2}: {e}")

    def _restore_positions_from_db(self):
        """Restore open positions from jp_open_positions table."""
        try:
            pos_df = db.jp_get_open_positions(self.con)
            if pos_df.empty:
                log.info("No JP positions to restore.")
                return

            from portfolio_jp import Position
            count = 0
            for _, row in pos_df.iterrows():
                try:
                    pos = Position(
                        pair=row["pair"], sym1=row["sym1"], sym2=row["sym2"],
                        direction=row["direction"],
                        qty1=int(row["qty1"]), qty2=int(row["qty2"]),
                        beta_entry=float(row["beta_entry"]),
                        entry_time=pd.Timestamp(row["entry_time"]),
                        entry_price1=float(row["entry_price1"]),
                        entry_price2=float(row["entry_price2"]),
                        entry_z=float(row["entry_z"]),
                        beta_drift_limit=float(row.get("beta_drift_limit", 10.0)),
                    )
                    self.portfolio.enter(pos)
                    count += 1
                except Exception as e:
                    log.warning(f"Failed to restore JP position {row.get('pair')}: {e}")

            if count > 0:
                log.info(f"Restored {count} JP open position(s) from DB.")
        except Exception as e:
            log.error(f"Failed to restore JP positions: {e}")

    def _restore_pnl_from_db(self):
        """Restore realized PnL from jp_pnl_summary."""
        try:
            pnl_df = db.jp_get_pnl_history(self.con, limit=1)
            if not pnl_df.empty:
                realized = float(pnl_df.iloc[0].get("realized_pnl", 0))
                self.portfolio.realized_pnl = realized
                log.info(f"Restored JP realized PnL: ¥{realized:,.0f}")
        except Exception as e:
            log.warning(f"Failed to restore JP PnL: {e}")

    def _run_polling_loop(self):
        """Main 1-minute polling loop."""
        log.info("JP Engine entering polling loop.")

        while True:
            try:
                # Check market hours
                if not is_jp_market_open():
                    wait_s = seconds_until_jp_market_open()
                    if wait_s > 300:
                        log.info(f"JP market closed. Sleeping {wait_s/3600:.1f}h until next open.")
                        self.heartbeat_thread.set_status("Waiting-MarketOpen")
                        time.sleep(min(wait_s, 3600))  # Sleep max 1h at a time
                        continue
                    elif wait_s > 0:
                        time.sleep(wait_s)
                        continue

                self.heartbeat_thread.set_status("Running")

                # Fetch latest 1m prices (15-min delayed so interpolation settles)
                latest_prices = fetch_latest_close_delayed(self.symbols, delay_minutes=15)
                if not latest_prices:
                    log.warning("No JP 1m data received (delayed). Retrying...")
                    time.sleep(15)
                    continue

                # Update 1m cache
                for sym, price in latest_prices.items():
                    self.cache_1m.update(sym, price)

                # Check if slow metrics need refresh (every 15m)
                now = datetime.now(timezone.utc)
                if (self._last_slow_refresh_wall is None or
                        (now - self._last_slow_refresh_wall).total_seconds() > 900):
                    log.info("Refreshing JP slow metrics (15m)...")
                    frames_15m = fetch_many_symbols_15m(self.symbols, days=HISTORY_DAYS)
                    for sym, df in frames_15m.items():
                        if not df.empty:
                            self.cache_15m.seed(sym, df["close"])
                    self._update_slow_metrics()
                    self._last_slow_refresh_wall = now

                    # Persist 15m history
                    try:
                        symbol_frames = {s: self.cache_15m.data[s].to_frame() for s in self.symbols if s in self.cache_15m.data}
                        pair_frames = build_pair_frames(symbol_frames, self.pairs, how="inner")
                        metrics_df = build_live_metrics_table(pair_frames)
                        history_df = build_pair_series_table(pair_frames)
                        db.jp_save_snapshot(self.con, metrics_df)
                        db.jp_save_history(self.con, history_df, timeframe="15m")
                    except Exception as e:
                        log.error(f"Failed to persist JP 15m data: {e}")

                # Run fast strategy cycle for each pair
                params_dict = self._params_as_dict()
                for s1, s2 in self.pairs:
                    pair_key = f"{s1}-{s2}"
                    p1 = latest_prices.get(s1)
                    p2 = latest_prices.get(s2)
                    if p1 is None or p2 is None:
                        continue

                    # Compute 1m z-score using cached slow metrics
                    slow = self.slow_metrics.get(pair_key)
                    if slow is None:
                        continue

                    spread = p1 / p2
                    mean_cached = slow.get("mean")
                    std_cached = slow.get("std")
                    if mean_cached is None or std_cached is None or std_cached == 0:
                        continue

                    z_1m = (spread - mean_cached) / std_cached

                    # Persistence filter (2/3 bars must exceed threshold)
                    if pair_key not in self._z_history:
                        self._z_history[pair_key] = deque(maxlen=3)
                    self._z_history[pair_key].append(z_1m)

                    pair_params = params_dict.get(pair_key, {})
                    z_entry = pair_params.get("z_entry", 3.5)
                    z_exit = pair_params.get("z_exit", 1.0)

                    # Check entry/exit (simplified — mirrors US engine logic)
                    is_open = pair_key in self.portfolio.positions

                    if is_open:
                        # Check exit
                        if abs(z_1m) <= z_exit:
                            pos = self.portfolio.positions[pair_key]
                            self.portfolio.exit(
                                pair_key,
                                exit_time=pd.Timestamp.now(tz="UTC"),
                                exit_price1=p1, exit_price2=p2,
                                exit_z=z_1m,
                            )
                            log.info(f"JP EXIT: {pair_key} z={z_1m:.2f}")
                            notifier.notify(f"🇯🇵 EXIT {pair_key} z={z_1m:.2f} P&L=¥{self.portfolio.closed[-1].pnl:,.0f}")
                    else:
                        # Check entry (persistence: 2/3 bars above threshold)
                        z_ring = list(self._z_history[pair_key])
                        passed_count = sum(1 for z in z_ring if abs(z) >= z_entry)
                        if len(z_ring) >= 2 and passed_count >= 2:
                            # Determine direction
                            direction = "SHORT_SPREAD" if z_1m > 0 else "LONG_SPREAD"

                            # Beta-adjusted sizing
                            direct_beta = p1 / p2
                            alloc = pair_params.get("alloc_pct", 0.05) * self.portfolio.equity()
                            leg1_notional = alloc / (1 + direct_beta)
                            leg2_notional = leg1_notional * direct_beta
                            qty1 = max(1, int(leg1_notional / p1))
                            qty2 = max(1, int(leg2_notional / p2))

                            from portfolio_jp import Position
                            pos = Position(
                                pair=pair_key, sym1=s1, sym2=s2,
                                direction=direction,
                                qty1=qty1, qty2=qty2,
                                beta_entry=direct_beta,
                                entry_time=pd.Timestamp.now(tz="UTC"),
                                entry_price1=p1, entry_price2=p2,
                                entry_z=z_1m,
                            )
                            self.portfolio.enter(pos)
                            log.info(f"JP ENTRY: {pair_key} {direction} z={z_1m:.2f} qty={qty1}/{qty2}")
                            notifier.notify(f"🇯🇵 ENTRY {pair_key} {direction} z={z_1m:.2f}")

                # Persist state
                self._persist_state()

                # Sleep until next bar
                sleep_for_next_bar(interval_s=60.0, buffer_s=5.0)

            except Exception as e:
                log.exception(f"Error in JP polling loop: {e}")
                self.heartbeat_thread.set_status("Warning", error=str(e))
                time.sleep(30)

    def _persist_state(self):
        """Persist portfolio state and metrics to JP DB tables."""
        try:
            # Mark-to-market
            latest_prices = self.cache_1m.get_latest_prices()
            mtm_df = self.portfolio.mark_to_market_unrealized(latest_prices)
            unrealized_pnl = 0.0
            if not mtm_df.empty and "unrealized_pnl" in mtm_df.columns:
                unrealized_pnl = mtm_df["unrealized_pnl"].sum()
                self._last_unrealized_pnl = unrealized_pnl

            # Positions
            pos_df = self.portfolio.open_positions_df()
            if not mtm_df.empty and not pos_df.empty:
                pnl_map = mtm_df.set_index("pair")
                for col in ["unrealized_pnl", "last_price1", "last_price2"]:
                    if col in pnl_map.columns:
                        mapped_col = "pnl_unrealized" if col == "unrealized_pnl" else col
                        pos_df[mapped_col] = pos_df["pair"].map(pnl_map[col]).fillna(0.0 if "pnl" in col else pd.NA)

            db.jp_save_open_positions(self.con, pos_df, engine_id=self.engine_id)

            # PnL summary
            db.jp_save_pnl_snapshot(
                self.con, self.portfolio.equity(),
                self.portfolio.realized_pnl, unrealized_pnl,
                len(self.portfolio.positions), len(self.portfolio.closed),
            )

            # Closed trades (new ones only)
            new_closed = self.portfolio.closed[self._persisted_closed_count:]
            if new_closed:
                from dataclasses import asdict
                for ct in new_closed:
                    ct_dict = asdict(ct)
                    ct_dict["meta_json"] = json.dumps({"engine_id": self.engine_id})
                    with self.con:
                        self.con.execute("""
                            INSERT INTO jp_closed_trades
                            (pair, direction, entry_time, exit_time, pnl, meta_json,
                             commission, borrow_cost, slippage, total_cost)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            ct_dict["pair"], ct_dict["direction"],
                            str(ct_dict["entry_time"]), str(ct_dict["exit_time"]),
                            ct_dict["pnl"], ct_dict["meta_json"],
                            ct_dict["commission"], ct_dict["borrow_cost"],
                            ct_dict["slippage"], ct_dict["total_cost"],
                        ))
                self._persisted_closed_count = len(self.portfolio.closed)

            # Equity snapshot (every 15 min)
            now = datetime.now(timezone.utc)
            if (self._last_equity_snapshot_ts is None or
                    (now - self._last_equity_snapshot_ts).total_seconds() > 900):
                db.jp_save_equity_snapshot(
                    self.con, self.portfolio.equity(),
                    self.portfolio.realized_pnl, unrealized_pnl,
                    len(self.portfolio.positions),
                )
                self._last_equity_snapshot_ts = now

        except Exception as e:
            log.error(f"Failed to persist JP state: {e}")

    def cleanup(self):
        log.info("JP Engine shutting down...")
        if self.heartbeat_thread:
            self.heartbeat_thread.set_status("Offline")
            self.heartbeat_thread.stop()
        try:
            self.con.close()
        except:
            pass
        log.info("JP Engine stopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-id", default="JP_1M", help="Identifier for this engine instance")
    args = parser.parse_args()

    engine = JPTradingEngine(engine_id=args.engine_id)
    engine.start()


if __name__ == "__main__":
    main()
