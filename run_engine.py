"""
run_engine.py

Unified entry point for the Pairs Trading Engine.
Supports two modes:
  1. --mode=realtime : NOT SUPPORTED (Requires IBKR).
  2. --mode=15m      : Polls Polygon.io every 15m for closed bars, executes strategy.

Features:
  - Heartbeat logging to DB (engine_heartbeat)
  - Robust error handling (updates status to 'Error' in DB)
  - Centralized DB access via db.py
"""

from __future__ import annotations

import argparse
import os
import time
import logging
import threading
import signal
import sys
import json
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List

import pandas as pd

# Local imports
from db import (
    connect_db, init_db, pulse_heartbeat, 
    upsert_pair_params, get_pair_params,
    log_order, get_heartbeat
)
import db  # Access to other db methods
from persist import persist_state

# Re-use existing logic where possible
# NOTE: IBKR Imports removed.
from data_polygon import fetch_many_symbols, fetch_latest_closed_15m_close

# NEW: Telegram Notifier
from telegram_notifier import notifier

from utils.pairs import parse_pairs
from price_cache import PriceCache
from live_metrics import build_live_metrics_table, build_pair_series_table
from portfolio import PaperPortfolio
from strategy.live_trader import process_pair_barclose

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("engine")

# ------------------------------------------------------------------------------
# CONFIG & CONSTANTS
# ------------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "data/live.db")
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "75")) # Warmup history
AGG_TIMEFRAME = "15m"
AGG_MINUTES = 15

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS (Migrated from Import_IBKR_multi if needed, or inline)
# ------------------------------------------------------------------------------
def unique_symbols_from_pairs(pairs: List[tuple]) -> List[str]:
    """Extract unique symbols from list of (s1, s2) tuples."""
    s = set()
    for s1, s2 in pairs:
        s.add(s1)
        s.add(s2)
    return list(s)

def build_pair_frames(symbol_frames: Dict[str, pd.DataFrame], 
                     pairs: List[tuple], 
                     how: str = "inner") -> Dict[tuple, pd.DataFrame]:
    """
    Combine individual symbol frames into pair frames.
    dict key: (sym1, sym2)
    dict value: DataFrame with cols [sym1, sym2]
    """
    results = {}
    for s1, s2 in pairs:
        if s1 in symbol_frames and s2 in symbol_frames:
            df1 = symbol_frames[s1]
            df2 = symbol_frames[s2]
            
            # Align
            # Ensure "close" col is renamed to symbol if not already
            if "close" in df1.columns:
                c1 = df1[["close"]].rename(columns={"close": s1})
            else:
                c1 = df1[[s1]] if s1 in df1.columns else df1.iloc[:, 0].to_frame(name=s1)
                
            if "close" in df2.columns:
                c2 = df2[["close"]].rename(columns={"close": s2})
            else:
                c2 = df2[[s2]] if s2 in df2.columns else df2.iloc[:, 0].to_frame(name=s2)
                
            # Merge
            joined = c1.join(c2, how=how).dropna()
            
            if not joined.empty:
                results[(s1, s2)] = joined
                
    return results

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
        # Create dedicated connection for this thread
        con = connect_db(self.db_path)
        try:
            while self.running:
                try:
                    pulse_heartbeat(
                        con, 
                        self.engine_id, 
                        self.status, 
                        error=self.error,
                        meta=self.meta
                    )
                except Exception as e:
                    log.error(f"Heartbeat write failed: {e}")
                time.sleep(self.interval_s)
        finally:
            con.close()

    def stop(self):
        self.running = False


# ------------------------------------------------------------------------------
# ENGINE CLASS
# ------------------------------------------------------------------------------
class TradingEngine:
    def __init__(self, mode: str, engine_id: str):
        self.mode = mode
        self.engine_id = engine_id
        
        # Database
        self.con = connect_db(DB_PATH)
        init_db(self.con)
        
        # NOTE: IBKR Connection Removed.
        
        # State
        self.portfolio = PaperPortfolio(starting_equity=100_000)
        self.cache = PriceCache()
        self.symbols: List[str] = []
        self.pairs: List[tuple] = []
        self.heartbeat_thread: Optional[HeartbeatThread] = None
        
        # Trade notification deduplication (load from DB for restart resilience)
        self._sent_event_ids: set = self._load_sent_events()

    def start(self):
        log.info(f"Starting Engine | Mode: {self.mode} | ID: {self.engine_id}")
        notifier.notify(f"🚀 Engine Starting | Mode: {self.mode} | ID: {self.engine_id}")
        
        # Start Heartbeat
        self.heartbeat_thread = HeartbeatThread(DB_PATH, self.engine_id)
        self.heartbeat_thread.start()
        
        try:
            self._run_setup()
            self.heartbeat_thread.set_status("Running")
            
            if self.mode == "realtime":
                self._run_realtime_loop()
            elif self.mode == "15m":
                self._run_15m_polling_loop()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
                
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        except Exception as e:
            log.exception("Fatal error in engine loop")
            notifier.notify(f"💀 Fatal Engine Error: {e}")
            self.heartbeat_thread.set_status("Error", error=str(e))
            # Wait a bit so the heartbeat persists the error
            time.sleep(5)
        finally:
            self.cleanup()

    def _load_params(self):
        """Load enabled pairs from DB."""
        df = get_pair_params(self.con)
        if df is None or df.empty:
            log.warning("No pair parameters found in DB.")
            return [], []
            
        # Filter enabled
        enabled = df[df["enabled"] == 1]
        raw_pairs = enabled["pair"].tolist()
        pairs_tuples = parse_pairs(raw_pairs)
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
        # 1. Load Pairs
        self.pairs, self.symbols = self._load_params()
        if not self.symbols:
            log.error("No enabled symbols found. Exiting setup.")
            return

        # 2. Warmup (POLYGON)
        self.heartbeat_thread.set_status("Warmup")
        log.info(f"Warming up {len(self.symbols)} symbols for {HISTORY_DAYS} days (Polygon)...")
        
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(days=HISTORY_DAYS)

        # Using Polygon fetcher
        frames = fetch_many_symbols(
            symbols=self.symbols,
            start_utc=start_utc.strftime("%Y-%m-%d"), # Polygon friendly format
            end_utc=end_utc.strftime("%Y-%m-%d"),
            verbose=True
        )
        
        for sym, df in frames.items():
            if not df.empty:
                self.cache.seed(sym, df)
                
        log.info("Warmup complete.")
        notifier.notify(f"✅ Warmup complete. Tracking {len(self.symbols)} symbols.")
        
        # 3. Persist Initial History (CRITICAL for Dashboard)
        log.info("Persisting initial history to DB...")
        try:
            symbol_frames_warm = {s: self.cache.data[s].to_frame() for s in self.symbols if s in self.cache.data}
            if symbol_frames_warm:
                pair_frames_warm = build_pair_frames(symbol_frames_warm, self.pairs, how="inner")
                history_df = build_pair_series_table(pair_frames_warm)
                if not history_df.empty:
                    # Save full history block
                    db.save_history(self.con, history_df, timeframe="15m")
                    log.info(f"Persisted {len(history_df)} historical bars to DB.")
                else:
                    log.warning("No historical bars generated during warmup.")
        except Exception as e:
            log.error(f"Failed to persist initial history: {e}")

        # 4. Save initial history/metrics to DB so UI has something immediately
        # Restore state BEFORE the first cycle so we don't overwrite DB with empty state
        self._restore_positions_from_db()
        self._run_strategy_cycle("Warmup-Finish")


    def _restore_positions_from_db(self):
        """
        Restore open positions from DB on engine restart.
        """
        from portfolio import Position
        
        try:
            positions_df = db.get_open_positions(self.con)
            if positions_df.empty:
                log.info("No open positions to restore from DB.")
                return
            
            # Filter for this engine? Or restore all?
            # Schema now has engine_id.
            if "engine_id" in positions_df.columns:
                positions_df = positions_df[positions_df["engine_id"] == self.engine_id]
            
            restored = 0
            for _, row in positions_df.iterrows():
                try:
                    pair = str(row.get("pair", ""))
                    if not pair: continue
                    
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


    def _run_realtime_loop(self):
        """
        NOT SUPPORTED.
        """
        raise NotImplementedError("Realtime requires IBKR; not supported in this build. Use --mode=15m for Polygon polling.")

    def _on_bar_update(self, sym: str, bars: list):
        pass


    def _run_15m_polling_loop(self):
        """
        Polls every X minutes for list of historical bars (Polygon).
        """
        from jp_market_time import sleep_until_next_15m_close
        from data_polygon import fetch_since_timestamp, fetch_latest_closed_15m_close
        import pytz
        
        # Market Status Tracking
        ny_tz = pytz.timezone('US/Eastern')
        was_market_open = False # Initialize conservative
        
        def is_market_open():
            now_ny = datetime.now(ny_tz)
            is_weekday = now_ny.weekday() < 5
            market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
            return is_weekday and (market_open <= now_ny <= market_close)

        # Initial Check
        was_market_open = is_market_open()
        
        while True:
            # 1. Check Market Status Transition
            current_open = is_market_open()
            if current_open and not was_market_open:
                notifier.notify("🔔 <b>Market Open</b> (US)")
            elif not current_open and was_market_open:
                notifier.notify("🏁 <b>Market Closed</b> (US)")
            was_market_open = current_open
            
            # 2. Update Data
            log.info("Updating latest closed bars (Polygon)...")
            updated_count = 0
            
            for sym in self.symbols:
                try:
                    # Gap Filling Logic
                    last_known_t, _ = self.cache.get_last(sym)
                    
                    if last_known_t:
                        # Smart fetch: Get everything since last known
                        new_bars = fetch_since_timestamp(sym, last_known_t)
                        if not new_bars.empty:
                            # Iterate all new bars and update cache
                            # This ensures we capture all history if we missed a cycle
                            for ts, row in new_bars.iterrows():
                                self.cache.update_close(sym, ts, float(row["close"]))
                            updated_count += 1
                    else:
                        # Fallback: Startup or no history, just get latest
                        t, close = fetch_latest_closed_15m_close(sym)
                        if t and close:
                            self.cache.update_close(sym, t, close)
                            updated_count += 1
                            
                except Exception as e:
                    log.warning(f"Failed update for {sym}: {e}")
                    notifier.notify_once(f"DATA_FAIL_{sym}", f"⚠️ Failed update for {sym}: {e}", cooldown_minutes=60)
            
            # Update Heartbeat Metadata for Dashboard
            if self.heartbeat_thread:
                self.heartbeat_thread.meta.update({
                    "last_data_update_ts": datetime.now(timezone.utc).isoformat(),
                    "updated_count": updated_count,
                    "total_symbols": len(self.symbols)
                })
            
            log.info(f"Updated {updated_count}/{len(self.symbols)} symbols.")
            
            if current_open:
                log.info("Market Open - Running strategy.")
                self._run_strategy_cycle(trigger_event="Poll-15m")
            else:
                 log.info("Market Closed - Skipping strategy.")
            
            log.info("Sleeping until next 15m close...")
            sleep_until_next_15m_close(buffer_s=30) # Increased buffer for REST API lag


    def _run_strategy_cycle(self, trigger_event: str):
        """
        Common logic: 
        1. Build PairFrames from Cache
        2. Evaluate Strategy
        3. Send Trade Notifications
        4. Persist everything
        """
        try:
            # 1. Build Frames
            symbol_frames_live = {s: self.cache.data[s].to_frame() for s in self.symbols if s in self.cache.data}
            pair_frames = build_pair_frames(symbol_frames_live, self.pairs, how="inner")
            
            if not pair_frames:
                return

            # 2. Params
            params_map = self._params_as_dict()

            # 3. Strategy + Notifications
            trade_actions = []
            
            for (sym1, sym2), df in pair_frames.items():
                pair_name = f"{sym1}-{sym2}"
                p = params_map.get(pair_name)
                
                # Check enabled
                if not p or int(p.get("enabled", 1)) != 1:
                    continue
                
                try:
                    # process_pair_barclose returns trade action if entry/exit occurred
                    trade_action = process_pair_barclose(
                        self.portfolio,
                        df,
                        sym1,
                        sym2,
                        z_entry=float(p["z_entry"]),
                        z_exit=float(p["z_exit"]),
                        max_drift_pct=float(p["max_drift_pct"]),
                        max_drift_delta=float(p.get("max_drift_delta", 0)),
                        alloc_pct=float(p["alloc_pct"]),
                    )
                    
                    if trade_action:
                        trade_actions.append(trade_action)
                    
                except Exception as e:
                    log.error(f"Strategy fail {pair_name}: {e}")
                    notifier.notify_once(f"STRAT_FAIL_{pair_name}", f"❌ Strategy fail {pair_name}: {e}", cooldown_minutes=60)

            # 4. Send Trade Notifications (with deduplication)
            for action in trade_actions:
                try:
                    self._send_trade_notification(action)
                except Exception as e:
                    log.error(f"Failed to send trade notification: {e}")

            # 5. Metrics & Persistence
            metrics_df = build_live_metrics_table(pair_frames)
            persist_state(self.con, metrics_df, self.portfolio, self.portfolio.closed_trades_df(), self.engine_id)
            
            # Heartbeat metadata update
            if self.heartbeat_thread:
                self.heartbeat_thread.meta["last_cycle"] = trigger_event
                self.heartbeat_thread.meta["pairs_tracked"] = len(pair_frames)
                
        except Exception as e:
            log.exception("Error in strategy cycle")
            if self.heartbeat_thread:
                self.heartbeat_thread.set_status("Warning", error=str(e))

    def _send_trade_notification(self, action: dict):
        """
        Format and send trade notification via Telegram.
        Uses event_id for deduplication to prevent duplicate messages on restarts.
        """
        event_id = action.get("event_id", "")
        
        # Check if already sent (deduplication)
        if event_id in self._sent_event_ids:
            log.info(f"Skipping duplicate notification: {event_id}")
            return
        
        # Format message
        msg = self._format_trade_message(action)
        
        # Send notification
        notifier.notify(msg)
        log.info(f"Trade notification sent: {event_id}")
        
        # Track sent event
        self._sent_event_ids.add(event_id)
        
        # Persist sent event IDs (for restart resilience)
        self._persist_sent_events()
    
    def _format_trade_message(self, action: dict) -> str:
        """Format trade action into a human-readable Telegram message."""
        
        direction = action.get("direction", "")
        sym1 = action.get("sym1", "")
        sym2 = action.get("sym2", "")
        
        # Clear direction labels
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
        """Save sent event IDs to DB for restart resilience."""
        try:
            # Keep only last 1000 events to prevent unbounded growth
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
    
    def _load_sent_events(self):
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


    def cleanup(self):
        log.info("Cleaning up...")
        if self.heartbeat_thread:
            self.heartbeat_thread.set_status("Stopped")
            self.heartbeat_thread.stop()
        
        # NOTE: IB disconnect removed.
        
        if self.con:
            self.con.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["realtime", "15m"], default="realtime", help="Running mode")
    parser.add_argument("--engine-id", default="US", help="Identifier for this engine instance")
    args = parser.parse_args()

    engine = TradingEngine(args.mode, args.engine_id)
    engine.start()

if __name__ == "__main__":
    main()
