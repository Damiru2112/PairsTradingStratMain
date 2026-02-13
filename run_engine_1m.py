"""
run_engine_1m.py

1-Minute Pairs Trading Engine.
Runs strategy at 1-minute resolution while using cached 15m analytics.

Architecture:
- Fast Layer (1m): Fetches 1m bars, computes spread, uses cached mean/std for z-score
- Slow Layer (15m): Refreshes when new 15m bar available, computes betas/drift

Z-Score Consistency Note:
    Fast z-score uses 1m spread with 15m-distribution mean/std.
    This preserves original 30d rolling methodology calibration.
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

import pandas as pd

# Local imports
from db import connect_db, init_db, pulse_heartbeat, get_pair_params
import db
from persist import persist_state
from data_polygon import (
    fetch_many_symbols, 
    fetch_latest_closed_1m_close,
    fetch_latest_closed_1m_bar,
    fetch_since_timestamp_1m,
)
from telegram_notifier import notifier
from utils.pairs import parse_pairs, build_pair_frames
from price_cache import PriceCache
from slow_metrics_cache import SlowMetricsCache
from live_metrics import build_live_metrics_table, build_pair_series_table
from portfolio import PaperPortfolio
from strategy.live_trader import process_pair_barclose_fast
from jp_market_time import sleep_for_next_bar, sleep_short

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("engine_1m")

# Config
DB_PATH = os.getenv("DB_PATH", "data/live.db")
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "75"))

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
# ENGINE
# ------------------------------------------------------------------------------
class TradingEngine1m:
    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        
        # Database
        self.con = connect_db(DB_PATH)
        init_db(self.con)
        
        # Caches
        self.cache_1m = PriceCache()       # Fast: 1-minute prices
        self.cache_15m = PriceCache()      # Slow: 15-minute prices for slow metrics
        self.slow_metrics = SlowMetricsCache()
        
        # State
        self.portfolio = PaperPortfolio(starting_equity=100_000)
        self.symbols: List[str] = []
        self.pairs: List[tuple] = []
        self.heartbeat_thread: Optional[HeartbeatThread] = None
        
        # Timestamp tracking for deduplication and gating
        self._last_processed_1m_ts: Dict[str, pd.Timestamp] = {}  # Per-symbol
        self._last_slow_bar_ts: Optional[pd.Timestamp] = None     # Global
        self._sent_event_ids: Set[str] = self._load_sent_events()
        
        # Persistence filter: ring buffer of last 3 z-scores per pair
        from collections import deque
        self._z_history: Dict[str, deque] = {}  # {pair: deque([z1, z2, z3], maxlen=3)}
        
        # 1m bar cache: stores full OHLCV for signal logging
        self._bar_cache_1m: Dict[str, dict] = {}  # {symbol: {ts, open, high, low, close, volume}}

        # Persistence tracking
        self._persisted_closed_count = 0
        
        # EOD State
        self._pnl_recorded_today = False

    def start(self):
        log.info(f"Starting 1-Minute Engine | ID: {self.engine_id}")
        notifier.notify(f"🚀 Engine Starting (1m mode) | ID: {self.engine_id}")
        
        # Start Heartbeat
        self.heartbeat_thread = HeartbeatThread(DB_PATH, self.engine_id)
        self.heartbeat_thread.start()
        
        try:
            self._run_setup()
            self.heartbeat_thread.set_status("Running")
            self._run_1m_polling_loop()
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        except Exception as e:
            log.exception("Fatal error in engine loop")
            notifier.notify(f"💀 Fatal Engine Error: {e}")
            self.heartbeat_thread.set_status("Error", error=str(e))
            time.sleep(5)
        finally:
            self.cleanup()

    def _load_params(self):
        """Load enabled pairs from DB."""
        df = get_pair_params(self.con)
        if df is None or df.empty:
            log.warning("No pair parameters found in DB.")
            return [], []
        
        enabled = df[df["enabled"] == 1]
        raw_pairs = enabled["pair"].tolist()
        pairs_tuples = parse_pairs(raw_pairs)
        from utils.pairs import unique_symbols_from_pairs
        symbols = unique_symbols_from_pairs(pairs_tuples)
        return pairs_tuples, symbols

    def _params_as_dict(self) -> dict:
        """Helper to get O(1) param lookup."""
        df = get_pair_params(self.con)
        out = {}
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                out[str(row["pair"])] = row.to_dict()
        return out

    def _run_setup(self):
        """Initial setup: load pairs, warmup caches, compute initial slow metrics."""
        
        # 1. Load Pairs
        self.pairs, self.symbols = self._load_params()
        if not self.symbols:
            log.error("No enabled symbols found. Exiting setup.")
            return

        # 2. Warmup 15m cache (for slow metrics)
        self.heartbeat_thread.set_status("Warmup-15m")
        log.info(f"Warming up {len(self.symbols)} symbols (15m) for {HISTORY_DAYS} days...")
        
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(days=HISTORY_DAYS)
        
        frames_15m = fetch_many_symbols(
            symbols=self.symbols,
            start_utc=start_utc.strftime("%Y-%m-%d"),
            end_utc=end_utc,
            verbose=True
        )
        
        for sym, df in frames_15m.items():
            if not df.empty:
                self.cache_15m.seed(sym, df)
        
        # 3. CRITICAL: Compute initial slow metrics (blocking warmup)
        self.heartbeat_thread.set_status("Warmup-SlowMetrics")
        log.info("Computing initial slow metrics...")
        self._update_slow_metrics()
        
        if self.slow_metrics.is_empty():
            log.error("Failed to compute slow metrics. Check data availability.")
            notifier.notify("⚠️ Slow metrics warmup failed - no tradable pairs")
        else:
            log.info(f"Slow metrics computed for {len(self.slow_metrics.get_all_pairs())} pairs")
        
        # 4. Persist initial history to DB
        log.info("Persisting initial history to DB...")
        try:
            symbol_frames_warm = {s: self.cache_15m.data[s].to_frame() for s in self.symbols if s in self.cache_15m.data}
            if symbol_frames_warm:
                # Use util valid for pairs
                from utils.pairs import build_pair_frames, unique_symbols_from_pairs
                pair_frames_warm = build_pair_frames(symbol_frames_warm, self.pairs, how="inner")
                history_df = build_pair_series_table(pair_frames_warm)
                if not history_df.empty:
                    db.save_history(self.con, history_df, timeframe="15m")
                    log.info(f"Persisted {len(history_df)} historical bars to DB.")
        except Exception as e:
            log.error(f"Failed to persist initial history: {e}")
        
        # 5. CRITICAL: Restore open positions from DB (for restart resilience)
        # 5. CRITICAL: Restore open positions from DB (for restart resilience)
        self._restore_positions_from_db()
        self._restore_pnl_from_db()
        
        # 6. Initial Persist (so dashboard reflects warmup immediately)
        try:
            # We need to build a dummy metrics_df or use the one from compute_slow_metrics?
            # compute_slow_metrics returns dict of Series? No, returns DataFrame? 
            # Let's check compute_slow_metrics return type.
            # actually run_engine_1m lines 205: self.metrics_df = compute_slow_metrics(...)
            # So we use self.metrics_df
            
            # We also need positions_df and pnl.
            # logic similar to main loop:
            # We need latest prices from cache (which we just warmed up)
            # cache_15m.get_latest_prices() returns {symbol: price}
            latest_prices = self.cache_15m.get_latest_prices()
            mtm_df = self.portfolio.mark_to_market_unrealized(latest_prices)
            mtm_positions = mtm_df # Use the DF returned by mark_to_market
            
            total_equity = self.portfolio.equity() # + unrealized? 
            # Portfolio equity() returns realized only? Let's check. 
            # portfolio.py L47: return self.starting_equity + self.realized_pnl
            # So we need to add unrealized.
            unrealized_pnl = mtm_df["unrealized_pnl"].sum() if not mtm_df.empty else 0.0
            
            # Note: persist_state expects positions_df to be the output of mark_to_market_unrealized (with 'unrealized_pnl' column)
            
            
            persist_state(
                self.con, 
                self.metrics_df, 
                self.portfolio, 
                None, # No new closed trades in setup
                self.engine_id,
                positions_df=mtm_positions,
                unrealized_pnl=unrealized_pnl
            )
            log.info("Initial state persisted.")
        except Exception as e:
            log.error(f"Failed to persist initial state: {e}")

        notifier.notify(f"✅ Warmup complete (1m mode). Tracking {len(self.symbols)} symbols.")

    def _update_slow_metrics(self):
        """
        Refresh slow metrics from 15m cache.
        Called at startup and when new 15m bar is available.
        """
        symbol_frames = {s: self.cache_15m.data[s].to_frame() for s in self.symbols if s in self.cache_15m.data}
        pair_frames = build_pair_frames(symbol_frames, self.pairs, how="inner")
        
        if not pair_frames:
            return
        
        for (sym1, sym2), df in pair_frames.items():
            pair_name = f"{sym1}-{sym2}"
            self.slow_metrics.update(pair_name, sym1, sym2, df)
        
        # Update last slow bar timestamp
        for s in self.symbols:
            if s in self.cache_15m.data and not self.cache_15m.data[s].empty:
                ts = self.cache_15m.data[s].index[-1]
                if self._last_slow_bar_ts is None or ts > self._last_slow_bar_ts:
                    self._last_slow_bar_ts = ts

    def _should_refresh_slow(self) -> bool:
        """Check if a new 15m bar is available (based on timestamp, not wall-clock)."""
        # Sample one symbol
        if not self.cache_15m.data:
            return False
        
        sample_sym = next(iter(self.cache_15m.data))
        current_latest = self.cache_15m.data[sample_sym].index[-1] if not self.cache_15m.data[sample_sym].empty else None
        
        if current_latest is None:
            return False
        
        if self._last_slow_bar_ts is None:
            return True
        
        return current_latest > self._last_slow_bar_ts

    def _run_1m_polling_loop(self):
        """
        Main loop - runs every ~60 seconds.
        Fetches 1m bars, runs fast strategy, optionally refreshes slow metrics.
        """
        import pytz
        ny_tz = pytz.timezone('US/Eastern')
        was_market_open = False
        
        def is_market_open():
            now_ny = datetime.now(ny_tz)
            is_weekday = now_ny.weekday() < 5
            market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
            return is_weekday and (market_open <= now_ny <= market_close)

        was_market_open = is_market_open()
        
        while True:
            # 1. Market status transitions
            current_open = is_market_open()
            if current_open and not was_market_open:
                notifier.notify("🔔 <b>Market Open</b> (US)")
                self._pnl_recorded_today = False  # Reset for new day
            elif not current_open and was_market_open:
                notifier.notify("🏁 <b>Market Closed</b> (US)")
                # Trigger EOD P&L processing immediately on close if not done
                if not self._pnl_recorded_today:
                    self._process_end_of_day()
            was_market_open = current_open
            

            
            # 2. Fetch 1m data
            new_bars_found = self._update_1m_prices()
            
            # 3. Check if slow metrics need refresh (new 15m bar)
            if self._should_refresh_slow():
                log.info("New 15m bar detected, refreshing slow metrics...")
                self._refresh_15m_cache()
                self._update_slow_metrics()
            
            # 4. Run fast strategy cycle (only if new bars AND market open)
            if new_bars_found:
                if current_open:
                    self._run_fast_strategy_cycle()
                else:
                    log.debug("Market closed - skipping strategy execution.")
            else:
                log.debug("No new 1m bars, skipping strategy cycle")
            
            # 5. Sleep before next poll
            sleep_for_next_bar(interval_s=60, buffer_s=5)

    def _update_1m_prices(self) -> bool:
        """
        Fetch latest 1m bars for all symbols.
        Returns True if at least one new bar was found.
        Uses timestamp gating to avoid reprocessing same bars.
        Also caches full bar OHLCV for signal logging.
        """
        new_bars_found = False
        
        for sym in self.symbols:
            try:
                # Get last known timestamp for this symbol
                last_known_ts = self._last_processed_1m_ts.get(sym)
                
                # Fetch full bar with OHLCV
                bar = fetch_latest_closed_1m_bar(sym)
                
                if bar is None:
                    continue
                
                bar_ts = bar["ts"]
                close = bar["close"]
                
                # Timestamp gating: only process if newer
                if last_known_ts is not None and bar_ts <= last_known_ts:
                    continue
                
                # New bar found - update caches
                self.cache_1m.update_close(sym, bar_ts, close)
                self._last_processed_1m_ts[sym] = bar_ts
                self._bar_cache_1m[sym] = bar  # Store full OHLCV
                new_bars_found = True
                
            except Exception as e:
                log.warning(f"Failed 1m update for {sym}: {e}")
        
        if new_bars_found:
            log.info(f"1m update: new bars fetched")
            # Update heartbeat metadata for data freshness tracking
            if self.heartbeat_thread:
                self.heartbeat_thread.meta.update({
                    "last_data_update_ts": datetime.now(timezone.utc).isoformat(),
                    "updated_symbol_count": len([s for s in self.symbols if s in self._last_processed_1m_ts]),
                    "total_symbols": len(self.symbols)
                })
        
        return new_bars_found

    def _refresh_15m_cache(self):
        """Refresh 15m cache when new 15m bar available."""
        from data_polygon import fetch_since_timestamp
        
        for sym in self.symbols:
            try:
                last_known_t, _ = self.cache_15m.get_last(sym)
                if last_known_t:
                    new_bars = fetch_since_timestamp(sym, last_known_t, multiplier=15)
                    if not new_bars.empty:
                        for ts, row in new_bars.iterrows():
                            self.cache_15m.update_close(sym, ts, float(row["close"]))
            except Exception as e:
                log.warning(f"Failed 15m update for {sym}: {e}")

    def _run_fast_strategy_cycle(self):
        """
        Fast strategy loop: check trades, mark PnL at 1-minute resolution.
        Uses cached slow metrics for z-score computation.
        
        Logging Rules:
        - Always log at entry/exit (mandatory)
        - Log when abs(z_1m) >= z_entry - 0.3 (near_action)
        """
        params_map = self._params_as_dict()
        trade_actions = []
        signal_logs = []  # Batch logging
        
        NEAR_ACTION_BUFFER = 0.3  # Log when z is within this of entry threshold
        
        for (sym1, sym2) in self.pairs:
            pair_name = f"{sym1}-{sym2}"
            p = params_map.get(pair_name)
            
            if not p or int(p.get("enabled", 1)) != 1:
                continue
            
            # Get slow metrics (required for fast z-score)
            slow = self.slow_metrics.get(pair_name)
            if slow is None:
                log.debug(f"No slow metrics for {pair_name}, skipping")
                continue
            
            try:
                # Get current 1m prices
                ts1, price1 = self.cache_1m.get_last(sym1)
                ts2, price2 = self.cache_1m.get_last(sym2)
                
                if price1 is None or price2 is None:
                    continue
                
                # Use most recent timestamp
                bar_ts = max(ts1, ts2) if ts1 and ts2 else (ts1 or ts2)
                
                # Compute current spread
                current_spread = price1 / price2
                
                # Compute fast z-score using cached mean/std
                z_now = self.slow_metrics.compute_fast_zscore(pair_name, current_spread)
                if z_now is None:
                    continue
                
                # --- PERSISTENCE FILTER (2/3 minutes) ---
                from collections import deque
                if pair_name not in self._z_history:
                    self._z_history[pair_name] = deque(maxlen=3)
                
                self._z_history[pair_name].append(z_now)
                z_entry_thresh = float(p["z_entry"])
                z_exit_thresh = float(p["z_exit"])
                
                # Count how many of last 3 evaluations exceeded entry threshold
                qualifying_count = sum(1 for z in self._z_history[pair_name] if abs(z) >= z_entry_thresh)
                passed_persistence = qualifying_count >= 2
                
                # Determine if position exists (for exit detection)
                in_position = pair_name in self.portfolio.positions
                would_exit = in_position and abs(z_now) <= z_exit_thresh
                would_enter = not in_position and abs(z_now) >= z_entry_thresh and passed_persistence
                
                # --- DETERMINE LOG REASON ---
                log_reason = None
                if would_enter:
                    log_reason = "entry"
                elif would_exit:
                    log_reason = "exit"
                elif abs(z_now) >= z_entry_thresh - NEAR_ACTION_BUFFER:
                    log_reason = "near_action"
                
                # --- BUILD LOG RECORD (only if should log) ---
                if log_reason:
                    bar_ts_iso = bar_ts.isoformat() if bar_ts else datetime.now(timezone.utc).isoformat()
                    
                    # Get OHLCV from bar cache
                    bar_a = self._bar_cache_1m.get(sym1, {})
                    bar_b = self._bar_cache_1m.get(sym2, {})
                    
                    signal_logs.append({
                        "time": bar_ts_iso,
                        "pair": pair_name,
                        "px_a": price1,
                        "px_b": price2,
                        "hi_a": bar_a.get("high"),
                        "lo_a": bar_a.get("low"),
                        "hi_b": bar_b.get("high"),
                        "lo_b": bar_b.get("low"),
                        "vol_a": bar_a.get("volume"),
                        "vol_b": bar_b.get("volume"),
                        "spread_1m": current_spread,
                        "mean_30d_cached": slow.spread_mean_30d if slow else None,
                        "std_30d_cached": slow.spread_std_30d if slow else None,
                        "z_1m": z_now,
                        "passed_persistence": passed_persistence,
                        "log_reason": log_reason,
                        "engine_id": self.engine_id,
                        "source": "polygon"
                    })
                
                # Run fast strategy (with persistence gate for entries)
                trade_action = process_pair_barclose_fast(
                    self.portfolio,
                    sym1, sym2,
                    price1, price2,
                    z_now,
                    bar_ts,
                    beta_30_weekly=slow.beta_30_weekly if slow else 1.0,
                    beta_drift_pct=slow.beta_drift_pct if slow else 0.0,
                    z_entry=z_entry_thresh,
                    z_exit=z_exit_thresh,
                    max_drift_pct=float(p["max_drift_pct"]),
                    max_drift_delta=float(p.get("max_drift_delta", 0)),
                    alloc_pct=float(p["alloc_pct"]),
                    entry_allowed=passed_persistence,
                )
                
                if trade_action:
                    trade_actions.append(trade_action)
                    
            except Exception as e:
                log.error(f"Strategy fail {pair_name}: {e}")
                notifier.notify_once(f"STRAT_FAIL_{pair_name}", f"❌ Strategy fail {pair_name}: {e}", cooldown_minutes=60)
        
        # --- BATCH LOG 1M SIGNALS ---
        if signal_logs:
            try:
                db.save_1m_signal_logs(self.con, signal_logs)
            except Exception as log_err:
                log.debug(f"Failed to batch log 1m signals: {log_err}")
        
        # Send notifications (with deduplication)
        for action in trade_actions:
            try:
                self._send_trade_notification(action)
            except Exception as e:
                log.error(f"Failed to send trade notification: {e}")
        
        # Persist state
        try:
            symbol_frames = {s: self.cache_15m.data[s].to_frame() for s in self.symbols if s in self.cache_15m.data}
            pair_frames = build_pair_frames(symbol_frames, self.pairs, how="inner")
            metrics_df = build_live_metrics_table(pair_frames)
            
            # --- COMPUTE PnL START ---
            latest_prices = {}
            for sym in self.symbols:
                ts, p = self.cache_1m.get_last(sym)
                if p: latest_prices[sym] = p
            
            # Compute MTM positions (with PnL)
            mtm_positions = self.portfolio.mark_to_market_unrealized(latest_prices)
            
            # Compute total unrealized
            unrealized_pnl = 0.0
            if not mtm_positions.empty and "unrealized_pnl" in mtm_positions.columns:
                unrealized_pnl = mtm_positions["unrealized_pnl"].sum()
                
                # Add columns expected by DB if missing (mark_to_market returns specificcols)
                # We need to ensure it has all fields for save_open_positions or we merge?
                # Actually, mark_to_market_unrealized returns a flat DF.
                # However, it might miss static fields like 'qty1', 'qty2' if not carefully constructed.
                # Let's check portfolio.mark_to_market_readiness.
                # It does NOT return qty/beta etc. It returns a summary.
                # We should merge with open_positions_df or better yet, just add PnL to standard export.
                
                # REVISED APPROACH:
                base_df = self.portfolio.open_positions_df()
                if not base_df.empty:
                    # Map PnL back to base_df
                    # mtm_positions has 'pair' and 'unrealized_pnl'
                    pnl_map = mtm_positions.set_index("pair")
                    base_df["pnl_unrealized"] = base_df["pair"].map(pnl_map["unrealized_pnl"]).fillna(0.0)
                    base_df["last_price1"] = base_df["pair"].map(pnl_map["last_price1"])
                    base_df["last_price2"] = base_df["pair"].map(pnl_map["last_price2"])
                    mtm_positions = base_df # Use the enriched full DF
            
            # --- COMPUTE PnL END ---

            # --- COMPUTE PnL END ---

            # Determine new closed trades to persist (prevent duplicates)
            all_closed = self.portfolio.closed_trades_df()
            if not all_closed.empty and len(self.portfolio.closed) > self._persisted_closed_count:
                new_limit = len(self.portfolio.closed)
                # Slice the list of ClosedTrade objects directly to ensure we match indices if needed
                # But using the DF is cleaner if we index by count.
                # Assuming append-only:
                new_closed = all_closed.iloc[self._persisted_closed_count:]
                self._persisted_closed_count = new_limit
            else:
                new_closed = None

            persist_state(
                self.con, 
                metrics_df, 
                self.portfolio, 
                new_closed, 
                self.engine_id,
                positions_df=mtm_positions,
                unrealized_pnl=unrealized_pnl
            )

        except Exception as e:
            log.error(f"Persist failed: {e}")
        
        # Heartbeat update
        if self.heartbeat_thread:
            self.heartbeat_thread.meta["last_cycle"] = datetime.now(timezone.utc).isoformat()
            self.heartbeat_thread.meta["pairs_tracked"] = len(self.pairs)

    def _send_trade_notification(self, action: dict):
        """Send trade notification with deduplication."""
        event_id = action.get("event_id", "")
        
        if event_id in self._sent_event_ids:
            log.info(f"Skipping duplicate notification: {event_id}")
            return
        
        msg = self._format_trade_message(action)
        notifier.notify(msg)
        log.info(f"Trade notification sent: {event_id}")
        
        self._sent_event_ids.add(event_id)
        self._persist_sent_events()

    def _format_trade_message(self, action: dict) -> str:
        """Format trade action into Telegram message."""
        direction = action.get("direction", "")
        sym1 = action.get("sym1", "")
        sym2 = action.get("sym2", "")
        
        if direction == "LONG_SPREAD":
            dir_text = f"Long spread (Long {sym1} / Short {sym2})"
        elif direction == "SHORT_SPREAD":
            dir_text = f"Short spread (Short {sym1} / Long {sym2})"
        else:
            dir_text = direction
        
        if action.get("action") == "ENTRY":
            gross = action.get("gross_usd", 0)
            msg = (
                f"🔵 <b>ENTRY: {action['pair']}</b>\n"
                f"Direction: {dir_text}\n"
                f"Z-Score: {action['z']:.2f} (threshold: ±{action.get('z_entry_threshold', 0):.1f})\n"
                f"Beta Drift: {action.get('beta_drift_pct', 0):.1f}%\n"
                f"{sym1}: ${action['price1']:.2f} × {action['qty1']} shares\n"
                f"{sym2}: ${action['price2']:.2f} × {action['qty2']} shares\n"
                f"Gross Notional: ${gross:,.0f}"
            )
        elif action.get("action") == "EXIT":
            pnl = action.get("pnl", 0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            holding = action.get("holding_minutes", 0)
            holding_str = f"{holding // 60}h {holding % 60}m" if holding >= 60 else f"{holding}m"
            exit_reason = action.get("exit_reason", "unknown").replace("_", " ").title()
            
            msg = (
                f"{emoji} <b>EXIT: {action['pair']}</b>\n"
                f"P&L: <b>${pnl:+,.2f}</b>\n"
                f"Reason: {exit_reason}\n"
                f"Entry Z: {action.get('entry_z', 0):.2f} → Exit Z: {action['z']:.2f}\n"
                f"Held: {holding_str}\n"
                f"{sym1}: ${action['price1']:.2f}\n"
                f"{sym2}: ${action['price2']:.2f}"
            )
        else:
            msg = f"📋 Trade action: {action}"
        
        return msg

    def _persist_sent_events(self):
        """Save sent event IDs to DB."""
        try:
            if len(self._sent_event_ids) > 1000:
                self._sent_event_ids = set(list(self._sent_event_ids)[-500:])
            
            events_json = json.dumps(list(self._sent_event_ids))
            with self.con:
                self.con.execute("""
                    INSERT OR REPLACE INTO engine_heartbeat (engine_id, timestamp, status, error_message, meta_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    f"{self.engine_id}_events",
                    datetime.now(timezone.utc).isoformat(),
                    "events_cache",
                    None,
                    events_json
                ))
        except Exception as e:
            log.warning(f"Failed to persist sent events: {e}")

    def _load_sent_events(self) -> Set[str]:
        """Load previously sent event IDs from DB."""
        try:
            cursor = self.con.execute(
                "SELECT meta_json FROM engine_heartbeat WHERE engine_id = ?",
                (f"{self.engine_id}_events",)
            )
            row = cursor.fetchone()
            if row and row[0]:
                return set(json.loads(row[0]))
        except Exception as e:
            log.warning(f"Failed to load sent events: {e}")
        return set()

    def _restore_positions_from_db(self):
        """
        Restore open positions from DB on engine restart.
        Critical for maintaining state across restarts.
        """
        from portfolio import Position
        
        try:
            positions_df = db.get_open_positions(self.con)
            if positions_df.empty:
                log.info("No open positions to restore from DB.")
                return
            
            restored = 0
            for _, row in positions_df.iterrows():
                try:
                    pair = str(row.get("pair", ""))
                    if not pair:
                        continue
                    
                    # Parse entry_time
                    entry_time = row.get("entry_time")
                    if entry_time and not pd.isna(entry_time):
                        entry_time = pd.Timestamp(entry_time)
                    else:
                        entry_time = pd.Timestamp.now(tz="UTC")
                    
                    pos = Position(
                        pair=pair,
                        sym1=str(row.get("sym1", "")),
                        sym2=str(row.get("sym2", "")),
                        direction=str(row.get("direction", "")),
                        qty1=int(row.get("qty1", 0) or 0),
                        qty2=int(row.get("qty2", 0) or 0),
                        beta_entry=float(row.get("beta_entry", 0) or 0),
                        entry_time=entry_time,
                        entry_price1=float(row.get("entry_price1", 0) or 0),
                        entry_price2=float(row.get("entry_price2", 0) or 0),
                        entry_z=float(row.get("entry_z", 0) or 0),
                        beta_drift_limit=float(row.get("beta_drift_limit", 10.0) or 10.0),
                    )
                    
                    self.portfolio.positions[pair] = pos
                    restored += 1
                    log.info(f"Restored position: {pair} ({pos.direction})")
                    
                except Exception as e:
                    log.warning(f"Failed to restore position {row.get('pair')}: {e}")
            
            if restored > 0:
                log.info(f"Restored {restored} open position(s) from DB.")
                notifier.notify(f"📂 Restored {restored} open position(s) from previous session.")
                
        except Exception as e:
            log.error(f"Failed to restore positions from DB: {e}")

    def _restore_pnl_from_db(self):
        """
        Restore total realized P&L from closed_trades.
        """
        try:
            cursor = self.con.execute("SELECT SUM(pnl) FROM closed_trades")
            row = cursor.fetchone()
            if row and row[0] is not None:
                total_pnl = float(row[0])
                self.portfolio.realized_pnl = total_pnl
                # Also move the persisted count pointer so we don't re-save old trades if we were to load them
                # But we don't load closed trades into memory list 'self.portfolio.closed'.
                # So the list is empty. _persisted_closed_count is 0.
                log.info(f"Restored Realized P&L: ${total_pnl:,.2f}")
                notifier.notify(f"💰 Restored Realized P&L: ${total_pnl:,.2f}")
            else:
                log.info("No historical P&L found (start $0).")
                
        except Exception as e:
            log.warning(f"Failed to restore PnL from DB: {e}")

    def _force_close_all_positions(self, reason: str):
        """
        Force close all open positions (e.g. for EOD).
        """
        pairs = list(self.portfolio.positions.keys())
        if not pairs:
            return

        trade_actions = []
        for pair in pairs:
            try:
                pos = self.portfolio.positions[pair]
                
                # Get current prices
                ts1, p1 = self.cache_1m.get_last(pos.sym1)
                ts2, p2 = self.cache_1m.get_last(pos.sym2)
                
                if not p1 or not p2:
                    log.warning(f"Cannot force close {pair}: missing prices")
                    continue
                
                # Use current prices for exit
                # Calculate PnL locally to construct action
                if pos.direction == "SHORT_SPREAD":
                    pnl = (pos.entry_price1 - p1) * pos.qty1 + (p2 - pos.entry_price2) * pos.qty2
                else:
                    pnl = (p1 - pos.entry_price1) * pos.qty1 + (pos.entry_price2 - p2) * pos.qty2
                
                # Calculate Z (approx using last known or 0 if missing, strictly for logging)
                # We can try to re-compute or just use 0.0
                slow = self.slow_metrics.get(pair)
                z_now = 0.0
                if slow:
                    spread = p1 / p2
                    z_now = slow.compute_fast_zscore(pair, spread) or 0.0

                # Execute exit in portfolio
                entry_time = pos.entry_time
                bar_ts = datetime.now(timezone.utc)
                
                self.portfolio.exit(pair, exit_time=bar_ts, exit_price1=p1, exit_price2=p2, exit_z=z_now)
                
                # Construct action dict
                try:
                    holding_minutes = int((bar_ts - entry_time).total_seconds() / 60)
                except:
                    holding_minutes = 0
                
                event_id = f"{pair}:EXIT:{bar_ts.isoformat()}:{pos.direction}:FORCED"
                
                action = {
                    "action": "EXIT",
                    "event_id": event_id,
                    "pair": pair,
                    "sym1": pos.sym1,
                    "sym2": pos.sym2,
                    "price1": p1,
                    "price2": p2,
                    "z": z_now,
                    "direction": pos.direction,
                    "pnl": float(pnl),
                    "entry_z": pos.entry_z,
                    "timestamp": bar_ts.isoformat(),
                    "holding_minutes": holding_minutes,
                    "exit_reason": reason,
                    "qty1": pos.qty1,
                    "qty2": pos.qty2,
                }
                trade_actions.append(action)
                log.info(f"Force closed {pair} due to {reason}")
                
            except Exception as e:
                log.error(f"Failed to force close {pair}: {e}")

        # Send notifications and persist
        for action in trade_actions:
            self._send_trade_notification(action)
            
        # Trigger persistence immediately
        # We can call persist loop logic or just let next cycle handle it?
        # Better to persist now to be safe.
        # But simplistic: just depend on the next loop cycle or end of this cycle.
        # Actually, this is called inside the loop, so persistence will happen at end of loop.
        pass

    def _process_end_of_day(self):
        """
        Calculate daily stats and persist to DB.
        """
        if self._pnl_recorded_today:
            return

        try:
            log.info("Processing End-Of-Day stats...")
            
            # 1. Calculate Daily Realized P&L
            # We need to sum PnL of all trades closed TODAY.
            # self.portfolio.closed contains all closed trades since engine start?
            # No, 'closed_trades' table has history.
            # Best way: Query DB for trades closed today.
            
            # Determine "today" based on NY time
            import pytz
            ny_tz = pytz.timezone('US/Eastern')
            now_ny = datetime.now(ny_tz)
            today_str = now_ny.strftime("%Y-%m-%d")
            
            # Query DB
            # We need a db method for this. Or just raw sql.
            start_of_day = now_ny.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).isoformat()
            
            # closed_trades has exit_time in ISO.
            query = "SELECT pnl FROM closed_trades WHERE exit_time >= ?"
            cursor = self.con.execute(query, (start_of_day,))
            rows = cursor.fetchall()
            
            daily_realized_pnl = sum([r[0] for r in rows]) if rows else 0.0
            num_trades = len(rows)
            wins = sum(1 for r in rows if r[0] > 0)
            losses = sum(1 for r in rows if r[0] <= 0)
            
            # 2. Total Equity
            # realized_pnl + starting_equity ? 
            # Or current portfolio equity (which should be all cash if positions closed)
            # If we forced closed everything, equity is all realized.
            total_equity = self.portfolio.equity()
            
            # 3. Save to DB
            db.save_daily_performance(self.con, today_str, daily_realized_pnl, total_equity, num_trades, wins, losses)
            
            self._pnl_recorded_today = True
            log.info(f"EOD Stats Saved: {today_str} | PnL: ${daily_realized_pnl:.2f} | Equity: ${total_equity:.2f}")
            notifier.notify(
                f"📊 <b>Daily Report ({today_str})</b>\n"
                f"Realized P&L: <b>${daily_realized_pnl:+,.2f}</b>\n"
                f"Total Equity: <b>${total_equity:,.2f}</b>\n"
                f"Trades: {num_trades} (W:{wins} L:{losses})"
            )
            
        except Exception as e:
            log.error(f"Failed to process EOD stats: {e}")
            notifier.notify(f"⚠️ Failed to process EOD stats: {e}")



    def cleanup(self):
        log.info("Cleaning up...")
        if self.heartbeat_thread:
            self.heartbeat_thread.set_status("Stopped")
            self.heartbeat_thread.stop()
        if self.con:
            self.con.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-id", default="US_1M", help="Identifier for this engine instance")
    args = parser.parse_args()

    engine = TradingEngine1m(args.engine_id)
    engine.start()


if __name__ == "__main__":
    main()
