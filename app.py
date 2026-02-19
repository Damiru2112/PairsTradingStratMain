from __future__ import annotations

import os
import time
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from streamlit_autorefresh import st_autorefresh

# Centralized DB Access
import db

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Pairs Trading Dashboard", layout="wide", page_icon="📈")
st_autorefresh(interval=15_000, key="auto_refresh")  # 15s refresh

# ----------------------------
# DB CONNECTION
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "data" / "live.db")

# Ensure DB is ready
con = db.connect_db(DB_PATH)
db.init_db(con)  # Safe idempotent call

# ----------------------------
# HEADER & HEARTBEAT
# ----------------------------
# ----------------------------
# STATUS DASHBOARD (Top)
# ----------------------------

# ----------------------------
# STATUS DASHBOARD (Top)
# ----------------------------

def get_market_status() -> dict:
    """Returns Market status details."""
    import pytz
    ny_tz = pytz.timezone('US/Eastern')
    now_ny = datetime.now(ny_tz)
    
    is_weekday = now_ny.weekday() < 5 # 0=Mon, 4=Fri
    
    # Simple market hours (9:30 - 16:00 ET)
    # Extended logic: maybe allow 9:00 - 16:15 for prep/cleanup
    market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_open = is_weekday and market_open <= now_ny <= market_close
    
    return {
        "is_open": is_open,
        "text": "OPEN" if is_open else "CLOSED",
        "color": "green" if is_open else "grey"
    }

def get_system_issues(con, heartbeats) -> list:
    """Checks Engine Heartbeats. Returns list of Issues."""
    issues = []
    
    if not heartbeats:
        issues.append({
            "key": "sys:no_heartbeat",
            "type": "system",
            "severity": "red",
            "pair": None,
            "title": "No Engines Found",
            "value": "Zero Heartbeats",
            "threshold": None,
            "status": "blocked"
        })
        return issues
    
    active_count = 0
    
    for hb in heartbeats:
        ts = hb.get("timestamp", "")
        status = hb.get("status", "Offline")
        engine_id = hb.get("engine_id", "Unknown")
        
        # Ignore event cache heartbeats (artifacts from deduction)
        if engine_id.endswith("_events") or status == "events_cache":
            continue
        
        # Check stale (20 mins)
        is_stale = False
        if ts:
            try:
                ts_dt = pd.to_datetime(ts)
                now_dt = datetime.now(timezone.utc)
                if (now_dt - ts_dt).total_seconds() > 1200:
                    is_stale = True
                    issues.append({
                        "key": f"sys:{engine_id}:stale",
                        "type": "system",
                        "severity": "red",
                        "pair": None,
                        "title": f"Engine {engine_id} Stale",
                        "value": f"Last seen {ts}",
                        "threshold": "20m ago",
                        "status": "blocked"
                    })
            except:
                is_stale = True
        else:
            is_stale = True
            
        if status == "Error":
             issues.append({
                "key": f"sys:{engine_id}:error",
                "type": "system",
                "severity": "red",
                "pair": None,
                "title": f"Engine {engine_id} Error",
                "value": hb.get("error_message", "Unknown Error"),
                "threshold": None,
                "status": "blocked"
            })

        if not is_stale and status != "Error":
            active_count += 1
            
    return issues

def get_data_issues(con, heartbeats, is_market_open: bool = True) -> list:
    """Checks data freshness. Returns list of Issues."""
    issues = []
    
    # Find most recent data update from ANY active engine
    age_sec = 99999
    
    for hb in heartbeats:
        engine_id = hb.get("engine_id", "")
        # Skip event cache entries
        if engine_id.endswith("_events") or hb.get("status") == "events_cache":
            continue
        
        meta_json = hb.get("meta_json")
        if meta_json:
            try:
                meta = json.loads(meta_json)
                last_ts_str = meta.get("last_data_update_ts")
                if last_ts_str:
                    last_ts = pd.to_datetime(last_ts_str)
                    now_dt = datetime.now(timezone.utc)
                    this_age = (now_dt - last_ts).total_seconds()
                    # Use most recent (smallest age)
                    if this_age < age_sec:
                        age_sec = this_age
            except:
                pass
    
    # Fallback to Snapshot if no heartbeat metadata found
    if age_sec == 99999:
        try:
            query = "SELECT MAX(last_updated) as last FROM live_metrics_snapshot"
            df = pd.read_sql_query(query, con)
            if not df.empty and df.iloc[0]["last"]:
                last_ts = pd.to_datetime(df.iloc[0]["last"], utc=True)
                now_dt = datetime.now(timezone.utc)
                age_sec = (now_dt - last_ts).total_seconds()
        except: pass

    # Format age for display
    def format_age(sec):
        if sec > 3600:
            return f"{int(sec//3600)}h ago"
        elif sec > 60:
            return f"{int(sec//60)}m ago"
        else:
            return f"{int(sec)}s ago"

    # Stale Threshold: Only apply if market is OPEN
    if is_market_open:
        if age_sec > 960:  # >16 minutes = blocked
            issues.append({
                "key": "data:feed:stale",
                "type": "data",
                "severity": "red",
                "pair": None,
                "title": "Data Stale",
                "value": format_age(age_sec),
                "threshold": "16m",
                "status": "blocked"
            })
        elif age_sec > 600:  # >10 minutes = watch
            issues.append({
                "key": "data:feed:aging",
                "type": "data",
                "severity": "amber",
                "pair": None,
                "title": "Data Aging",
                "value": format_age(age_sec),
                "threshold": "10m",
                "status": "watch"
            })
        
    return issues

def get_risk_issues(con, params) -> list:
    """Checks for drift breaches. Returns list of Issues."""
    issues = []
    try:
        metrics = db.get_latest_snapshot(con)
        if metrics.empty:
            # Maybe info?
            return issues
        
        if params is None or params.empty:
             issues.append({
                "key": "risk:config:missing",
                "type": "risk",
                "severity": "amber",
                "pair": None,
                "title": "No Risk Params",
                "value": "Missing",
                "threshold": None,
                "status": "watch"
            })
             return issues
             
        merged = pd.merge(metrics, params, on="pair", how="inner")
        if merged.empty:
            return issues

        # Check drift
        merged["drift_abs"] = merged["beta_drift_pct"].abs()
        
        # Breaches
        breaches = merged[merged["drift_abs"] > merged["max_drift_pct"]]
        for _, row in breaches.iterrows():
            pair = row["pair"]
            val = row["drift_abs"]
            lim = row["max_drift_pct"]
            issues.append({
                "key": f"risk:{pair}:beta_drift_breach",
                "type": "risk",
                "severity": "red",
                "pair": pair,
                "title": "Beta Drift Breach",
                "value": f"{val:.1f}%",
                "threshold": f"{lim:.1f}%",
                "status": "blocked",
                "z_score": row.get("z")
            })
            
        # Near Limit
        near_limit = merged[(merged["drift_abs"] > 0.8 * merged["max_drift_pct"]) & (merged["drift_abs"] <= merged["max_drift_pct"])]
        for _, row in near_limit.iterrows():
            pair = row["pair"]
            val = row["drift_abs"]
            lim = row["max_drift_pct"]
            issues.append({
                "key": f"risk:{pair}:beta_drift_near",
                "type": "risk",
                "severity": "amber",
                "pair": pair,
                "title": "Beta Drift Near Limit",
                "value": f"{val:.1f}%",
                "threshold": f"{lim:.1f}%",
                "status": "watch",
                "z_score": row.get("z")
            })
            
        return issues
            
    except Exception as e:
        issues.append({
            "key": "risk:check:fail",
            "type": "risk",
            "severity": "red",
            "pair": None,
            "title": "Risk Check Failed",
            "value": str(e),
            "threshold": None,
            "status": "blocked"
        })
        return issues

def calculate_verdict(mkt, issues, candidates_count) -> tuple[str, str, str]:
    """
    Returns (Verdict Title, Color, BackgroundColor)
    Precedence:
    1. ATTENTION REQUIRED (Red)
    2. TRADE CANDIDATES (Green/Blue)
    3. NO TRADE CONDITIONS (Grey)
    """
    
    # Check for any RED severity issues
    red_issues = [i for i in issues if i["severity"] == "red"]
    
    if red_issues:
        return "⚠️ ATTENTION REQUIRED", "white", "#d32f2f" # Red

    # 2. TRADE CANDIDATES
    # Conditions: Market Open, No Red Issues, Candidates > 0
    # Note: Explicit check for Data Stale is handled by Red Issues list now
    if mkt["is_open"]: 
        if candidates_count > 0:
            return f"{candidates_count} TRADE CANDIDATES", "white", "#2e7d32" # Green
    
    # 3. NO TRADE CONDITIONS
    # Default fallback
    return "NO TRADE CONDITIONS", "white", "#424242" # Grey


def detect_entry_signals(history_df: pd.DataFrame, z_entry: float) -> list:
    """
    Detect Z-score trade zone entry crossings from historical data.
    Returns list of dicts: {time, z, direction}
    - direction: 'long' (z < -z_entry) or 'short' (z > z_entry)
    Only marks FIRST entry per episode to avoid clutter.
    """
    signals = []
    in_trade_zone = False
    
    for _, row in history_df.iterrows():
        z = row.get("z", 0) or 0
        t = row.get("time")
        
        # Check zone transition
        if abs(z) >= z_entry and not in_trade_zone:
            # First entry into trade zone
            direction = "long" if z < 0 else "short"  # Negative Z = long spread
            signals.append({
                "time": t,
                "z": z,
                "direction": direction
            })
            in_trade_zone = True
            
        elif abs(z) < z_entry and in_trade_zone:
            # Exited trade zone, reset for next episode
            in_trade_zone = False
    
    return signals


def get_pair_blocking_status(pair: str, raw_issues: list) -> tuple:
    """
    Determine if a pair is currently blocked and why.
    Returns (is_blocked: bool, reason: str or None)
    Uses same logic as Trade Candidates.
    """
    # System/Data red issues block ALL pairs
    for issue in raw_issues:
        if issue.get("severity") == "red":
            if issue.get("type") == "system":
                return True, "System"
            if issue.get("type") == "data":
                return True, "Data"
    
    # Pair-specific risk red issues
    for issue in raw_issues:
        if issue.get("pair") == pair and issue.get("type") == "risk" and issue.get("severity") == "red":
            return True, "Risk"
    
    return False, None


def plot_z_histogram(history: pd.DataFrame, z_entry: float, n_bins: int = 30,
                     z_clip_range: tuple = None, current_z: float = None) -> go.Figure:
    """
    Create overlay histogram comparing 10-week vs 4-week z-score distributions.
    
    Args:
        history: DataFrame with 'time' and 'z' columns
        z_entry: Entry threshold for vertical reference lines
        n_bins: Number of histogram bins (20, 30, or 40)
        z_clip_range: Optional tuple (min, max) to clip z-values before binning
        current_z: Current z-score value for marker line
    
    Returns:
        Plotly Figure object (empty figure with annotation if insufficient data)
    """
    import numpy as np
    
    # Use data-based 'now' to avoid timezone issues
    if history.empty or "time" not in history.columns or "z" not in history.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig
    
    now = history["time"].max()
    cutoff_10w = now - timedelta(days=70)
    cutoff_4w = now - timedelta(days=28)
    
    # Filter windows
    hist_10w = history[history["time"] >= cutoff_10w].copy()
    hist_4w = history[history["time"] >= cutoff_4w].copy()
    
    # Extract z-values and drop NaN
    z10 = hist_10w["z"].dropna()
    z4 = hist_4w["z"].dropna()
    
    # Guard: insufficient data - return empty figure with message
    if len(z10) < 20:
        fig = go.Figure()
        fig.add_annotation(text="Not enough history for histogram", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="orange"))
        return fig
    
    # Apply clipping BEFORE computing bin edges
    if z_clip_range:
        z10 = z10.clip(z_clip_range[0], z_clip_range[1])
        z4 = z4.clip(z_clip_range[0], z_clip_range[1])
    
    # Compute shared bin edges from 10w data
    bin_min = z10.min()
    bin_max = z10.max()
    bin_size = (bin_max - bin_min) / n_bins if bin_max > bin_min else 0.1
    
    # Build figure
    fig = go.Figure()
    
    # Base histogram (10w) - solid/main
    fig.add_trace(go.Histogram(
        x=z10,
        name="10w (≈50d)",
        histnorm="probability density",
        xbins=dict(start=bin_min, end=bin_max, size=bin_size),
        marker_color="steelblue",
        opacity=0.85
    ))
    
    # Overlay histogram (4w) - translucent overlay
    if len(z4) >= 10:
        fig.add_trace(go.Histogram(
            x=z4,
            name="4w (≈20d)",
            histnorm="probability density",
            xbins=dict(start=bin_min, end=bin_max, size=bin_size),
            marker_color="coral",
            opacity=0.5
        ))
    
    # Reference lines
    fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1,
                  annotation_text="0", annotation_position="top")
    fig.add_vline(x=z_entry, line_dash="dash", line_color="green", line_width=1.5,
                  annotation_text=f"+Entry", annotation_position="top right")
    fig.add_vline(x=-z_entry, line_dash="dash", line_color="green", line_width=1.5,
                  annotation_text=f"-Entry", annotation_position="top left")
    
    # Current z marker
    if current_z is not None:
        fig.add_vline(x=current_z, line_dash="solid", line_color="red", line_width=2,
                      annotation_text=f"Current", annotation_position="bottom right")
    
    # Layout
    fig.update_layout(
        barmode="overlay",
        title=None,
        xaxis_title="Z-Score",
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=30, b=60),
        height=300
    )
    
    return fig


def get_trade_candidates(metrics, params, raw_issues, is_market_open=True, buffer=0.3):
    """
    Returns DataFrame sorted by priority with columns for display.
    """
    if params is None or params.empty or metrics is None or metrics.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    
    # Only enabled pairs
    if "enabled" not in params.columns:
        return pd.DataFrame(), pd.DataFrame(), []
        
    enabled_params = params[params["enabled"] == 1].copy()
    if enabled_params.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    
    # Merge metrics with params
    merged = pd.merge(metrics, enabled_params, on="pair", how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    
    # Ensure numeric z
    merged["z"] = pd.to_numeric(merged["z"], errors='coerce').fillna(0)
    merged["abs_z"] = merged["z"].abs()
    
    # Filter to near-entry zone: abs(z) >= z_entry - buffer
    merged["entry_threshold"] = merged["z_entry"] - buffer
    candidates = merged[merged["abs_z"] >= merged["entry_threshold"]].copy()
    
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    
    # Calculate "To Entry" (positive = away, negative = past entry)
    candidates["to_entry"] = candidates["z_entry"] - candidates["abs_z"]
    
    # Calculate time since signal (minutes)
    now_utc = datetime.now(timezone.utc)
    
    def calc_time_since(row):
        ts = row.get("last_updated") or row.get("time")
        if not ts:
            return 60.0  # Default to 60 mins if missing (treat as old)
        try:
            ts_dt = pd.to_datetime(ts, utc=True)
            return (now_utc - ts_dt).total_seconds() / 60.0
        except:
            return 60.0
    
    candidates["time_since_mins"] = candidates.apply(calc_time_since, axis=1)
    
    # Build lookup for issues by pair
    # System/data red issues block ALL pairs
    system_data_red = [i for i in raw_issues if i["type"] in ["system", "data"] and i["severity"] == "red"]
    has_global_block = len(system_data_red) > 0
    
    # Pair-level issues
    pair_issues = {}
    for issue in raw_issues:
        p = issue.get("pair")
        if p:
            if p not in pair_issues:
                pair_issues[p] = []
            pair_issues[p].append(issue)
    
    def determine_status(row):
        pair = row["pair"]
        to_entry = row["to_entry"]
        drift_pct = abs(row.get("beta_drift_pct", 0) or 0)
        max_drift = row.get("max_drift_pct", 20) or 20
        time_mins = row["time_since_mins"]
        
        # Check for blocks
        if has_global_block:
            return "🔴 Blocked (Data)"
        
        # Pair-specific red issues
        if pair in pair_issues:
            red_pair = [i for i in pair_issues[pair] if i["severity"] == "red"]
            if red_pair:
                return "🔴 Blocked (Risk)"
        
        # Check if at/past entry
        at_entry = to_entry <= 0
        
        # Amber conditions
        amber_drift = drift_pct > 0.8 * max_drift
        signal_old = is_market_open and time_mins > 30
        
        # Pair amber issues
        amber_issues = False
        if pair in pair_issues:
            amber_issues = any(i["severity"] == "amber" for i in pair_issues[pair])
        
        if at_entry and not amber_drift and not signal_old and not amber_issues:
            return "✅ Valid"
        else:
            return "🟡 Watch"
    
    candidates["status"] = candidates.apply(determine_status, axis=1)
    
    # Calculate priority score with guardrails
    def calc_priority(row):
        abs_z = row["abs_z"]
        drift_pct = abs(row.get("beta_drift_pct", 0) or 0)
        max_drift = row.get("max_drift_pct", 0) or 0
        time_mins = row["time_since_mins"]
        
        # Drift room: clamp to [0, 1]
        if max_drift > 0:
            drift_room = max(0, min(1, 1 - drift_pct / max_drift))
        else:
            drift_room = 0.5  # Fallback if missing
        
        # Weights
        w1 = 10  # z-score weight
        w2 = 5   # drift room weight
        w3 = 0.1 # time penalty
        
        return w1 * abs_z + w2 * drift_room - w3 * time_mins
    
    candidates["priority_score"] = candidates.apply(calc_priority, axis=1)
    
    # Sort by priority descending, limit to top 15
    candidates = candidates.sort_values("priority_score", ascending=False).head(15)
    
    # Format columns for display
    candidates["beta_drift_display"] = candidates["beta_drift_pct"].apply(
        lambda x: f"{abs(x):.1f}%" if pd.notnull(x) else "N/A"
    )
    candidates["to_entry_display"] = candidates["to_entry"].apply(
        lambda x: f"{x:+.2f}" if pd.notnull(x) else "N/A"
    )
    candidates["time_display"] = candidates["time_since_mins"].apply(
        lambda x: f"{int(x)}m" if pd.notnull(x) else "N/A"
    )
    
    return candidates, candidates, []


# ----------------------------
# LOGIC EXECUTION
# ----------------------------

# Fetch Objects (with retry for DB locks)
heartbeats = []
params = pd.DataFrame()
metrics = pd.DataFrame()

for i in range(3):
    try:
        heartbeats = db.get_heartbeat(con)
        params = db.get_pair_params(con)
        metrics = db.get_latest_snapshot(con)
        if not metrics.empty:
            break
        time.sleep(0.1)
    except Exception as e:
        if i == 2:
            st.error(f"DB Read Error: {e}")
        time.sleep(0.2)

# Run Checks (Issues List)
mkt = get_market_status()

raw_issues = []
raw_issues.extend(get_system_issues(con, heartbeats))
raw_issues.extend(get_data_issues(con, heartbeats, is_market_open=mkt["is_open"]))
raw_issues.extend(get_risk_issues(con, params))

# Candidate Scan
# ... code continues ...
candidates, filtered_candidates, blocked_reasons = get_trade_candidates(metrics, params, raw_issues, is_market_open=mkt["is_open"])

# --- UI FEEDBACK FOR EMPTY STATES ---
system_data_red = [i for i in raw_issues if i["type"] in ["system", "data"] and i["severity"] == "red"]
if system_data_red:
    blocker = system_data_red[0]
    st.error(f"⚠️  **System Halted**: {blocker['title']} ({blocker['value']})")
    
if metrics.empty:
    st.warning("⚠️  **No Market Data**: Live metrics table is empty. Engine might be warming up or restarting.")
# Simple definition for Verdict: Use potential actionable candidates
candidates_count = len(candidates) 


# Calculate Verdict
verdict_title, verdict_text_color, verdict_bg_color = calculate_verdict(mkt, raw_issues, candidates_count)

# Calc Badges
# Data: Check if any data issue exists
data_issues = [i for i in raw_issues if i["type"] == "data" and i["severity"] == "red"]
data_status = "STALE" if data_issues else "FRESH"
data_color = "red" if data_issues else "green"

# Risk: Count Red + Amber Risk items
risk_items = [i for i in raw_issues if i["type"] == "risk" and i["severity"] in ["red", "amber"]]
risk_status = "OK"
risk_color = "green"
if any(i["severity"] == "red" for i in risk_items):
    risk_status = "BREACH"
    risk_color = "red"
elif risk_items:
    risk_status = "WARNING"
    risk_color = "orange"
risk_label = f"{len(risk_items)} Risks" if risk_items else "Risk: OK"

# Alerts: Count ALL RED items (System + Data + Risk)
red_items = [i for i in raw_issues if i["severity"] == "red"]
alerts_count = len(red_items)
alerts_status = "None"
alerts_color = "green"
if alerts_count > 0:
    alerts_status = f"{alerts_count} Active"
    alerts_color = "red"

# System Badge details
sys_issues = [i for i in raw_issues if i["type"] == "system" and i["severity"] == "red"]
sys_color = "red" if sys_issues else "green" 
# (UI Badge removed for space, but logic kept for expander below)

# ----------------------------
# RENDER DECISION STRIP
# ----------------------------
st.markdown(f"""
    <style>
    .decision-strip {{
        background-color: {verdict_bg_color};
        color: {verdict_text_color};
        padding: 15px;
        text-align: center;
        border-radius: 5px;
        margin-bottom: 10px;
        font-family: sans-serif;
    }}
    .decision-title {{
        font-size: 28px;
        font-weight: bold;
        margin: 0;
    }}
    .badge-container {{
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-top: 10px;
    }}
    .status-badge {{
        background-color: rgba(0,0,0,0.2);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 500;
    }}
    </style>
    
    <div class="decision-strip">
        <div class="decision-title">{verdict_title}</div>
        <div class="badge-container">
            <span class="status-badge" style="border-left: 4px solid {mkt['color']}">Market: {mkt['text']}</span>
            <span class="status-badge" style="border-left: 4px solid {data_color}">Data: {data_status}</span>
            <span class="status-badge" style="border-left: 4px solid {risk_color}">{risk_label}</span>
            <span class="status-badge" style="border-left: 4px solid {alerts_color}">Alerts: {alerts_status}</span>
        </div>
    </div>
""", unsafe_allow_html=True)


# ----------------------------
# ALERTS & RISKS PANEL
# ----------------------------
# Auto-expand if Attention Required or if issues exist
panel_expanded = (verdict_title == "⚠️ ATTENTION REQUIRED") or (len(raw_issues) > 0)

if raw_issues:
    with st.expander(f"🚨 Alerts & Risks ({len(raw_issues)})", expanded=panel_expanded):
        
        # 1. Summary Bullets (Priority Sorted)
        # Priority: System (Red) > Data (Red) > Risk (Red) > Risk (Amber)
        def priority_sort(i):
            score = 0
            if i["severity"] == "red": score += 100
            if i["type"] == "system": score += 30
            elif i["type"] == "data": score += 20
            elif i["type"] == "risk": score += 10
            return -score # descending
            
        sorted_issues = sorted(raw_issues, key=priority_sort)
        
        # Market Closed Note
        if not mkt["is_open"] and any(i["type"] == "risk" for i in raw_issues):
            st.caption("ℹ️ *Risk evaluated on latest available snapshot. Breaches will block trading at next market open.*")
            
        # 1. Detailed Table (Styled)
        if sorted_issues:
            df_issues = pd.DataFrame(sorted_issues)
            # Cleanup for display
            disp_cols = ["type", "pair", "title", "value", "threshold", "z_score", "status", "timestamp"]
            # filter to available
            disp_cols = [c for c in disp_cols if c in df_issues.columns]
            
            # Styling definitions
            def highlight_severity(row):
                severity = row.get("severity", "info")
                if severity == "red":
                    return ['background-color: #5a2525'] * len(row) # Dark Red background
                elif severity == "amber":
                    return ['background-color: #5a4525'] * len(row) # Dark Amber background
                return [''] * len(row)

            # Apply style
            styled_df = df_issues.style.apply(highlight_severity, axis=1)

            st.dataframe(
                styled_df,
                column_order=disp_cols,
                column_config={
                    "pair": st.column_config.Column("Pair", width="medium"),
                    "z_score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                },
                hide_index=True,
                use_container_width=True
            )
            
        # 3. Jump To Interaction
        # Filter to pairs
        issue_pairs = [i["pair"] for i in sorted_issues if i["pair"]]
        # Unique preserve order
        seen = set()
        issue_pairs_u = [x for x in issue_pairs if not (x in seen or seen.add(x))]
        
        if issue_pairs_u:
            st.write("### Quick Actions")
            c_act, _ = st.columns([1, 2])
            with c_act:
                # Default index 0 is best priority
                jump_target = st.selectbox("Jump to Analysis", ["Select..."] + issue_pairs_u)
                if jump_target and jump_target != "Select...":
                    st.session_state["selected_pair_override"] = jump_target
                    st.success(f"Selected {jump_target} below.")

else:
    pass

# ----------------------------
# TRADE CANDIDATES SECTION
# ----------------------------
st.subheader("Trade Candidates (Sorted by Priority)")

# Reuse pre-calculated candidates
trade_candidates = filtered_candidates

if trade_candidates.empty:
    st.info("No pairs near entry thresholds right now.")
else:
    # Smart filter default: "Blocked only" when ATTENTION REQUIRED, else "All"
    default_filter = "Blocked only" if verdict_title == "⚠️ ATTENTION REQUIRED" else "All"
    filter_options = ["All", "Valid only", "Blocked only"]
    
    # Use session state to persist filter choice
    if "candidates_filter" not in st.session_state:
        st.session_state["candidates_filter"] = default_filter
    
    # Layout: filter + jump to pair
    filter_col, jump_col = st.columns([1, 1])
    
    with filter_col:
        selected_filter = st.radio(
            "Filter", 
            filter_options, 
            index=filter_options.index(st.session_state.get("candidates_filter", default_filter)),
            horizontal=True,
            key="candidates_filter_radio"
        )
        st.session_state["candidates_filter"] = selected_filter
    
    # Apply filter
    display_df = trade_candidates.copy()
    if selected_filter == "Valid only":
        display_df = display_df[display_df["status"].str.contains("Valid")]
    elif selected_filter == "Blocked only":
        display_df = display_df[display_df["status"].str.contains("Blocked")]
    
    # Prepare display columns
    display_cols = ["pair", "abs_z", "to_entry_display", "beta_drift_display", "time_display", "status"]
    col_names = {
        "pair": "Pair",
        "abs_z": "|Z|",
        "to_entry_display": "To Entry",
        "beta_drift_display": "Beta Drift %",
        "time_display": "Time Since Signal",
        "status": "Status"
    }
    
    # Filter to available columns
    display_cols = [c for c in display_cols if c in display_df.columns]
    
    if not display_df.empty:
        # Style function for status colors
        def style_status(row):
            status = row.get("status", "")
            if "Blocked" in status:
                return ['background-color: #5a2525'] * len(row)
            elif "Valid" in status:
                return ['background-color: #1b5e20'] * len(row)
            elif "Watch" in status:
                return ['background-color: #5a4525'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df[display_cols].rename(columns=col_names).style.apply(style_status, axis=1)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "|Z|": st.column_config.NumberColumn("|Z|", format="%.2f"),
            }
        )
        
        # Jump to Pair Analysis
        with jump_col:
            candidate_pairs = display_df["pair"].tolist()
            if candidate_pairs:
                jump_pair = st.selectbox(
                    "Jump to Pair Analysis",
                    ["Select..."] + candidate_pairs,
                    key="candidates_jump"
                )
                if jump_pair and jump_pair != "Select...":
                    st.session_state["selected_pair_override"] = jump_pair
                    st.success(f"Selected {jump_pair} — scroll down to Pair Analysis.")
    else:
        st.caption(f"No {selected_filter.lower().replace(' only', '')} candidates found.")

# ----------------------------
# EVENT RISK PANEL
# ----------------------------
try:
    from event_risk import compute_pair_event_state, init_event_tables, EASTERN
    from zoneinfo import ZoneInfo as _ZI
    
    # Ensure event tables exist
    init_event_tables(con)
    
    now_et = datetime.now(_ZI("America/New_York"))
    
    # Collect event states for all enabled pairs
    event_states = {}
    active_multiplier_pairs = []
    blocked_pairs = []
    all_upcoming = []
    
    enabled_pairs = params[params["enabled"] == 1]["pair"].tolist() if not params.empty else []
    
    for pair_name in enabled_pairs:
        try:
            state = compute_pair_event_state(pair_name, now_et, con)
            event_states[pair_name] = state
            
            if state["multiplier"] > 1.0:
                active_multiplier_pairs.append({
                    "pair": pair_name,
                    "multiplier": state["multiplier"],
                    "entry_blocked": state["entry_blocked"],
                    "events": ", ".join(e.get("window_label", e["type"]) for e in state["active_events"]),
                    "is_overridden": state.get("is_overridden", False),
                })
            if state["entry_blocked"]:
                blocked_pairs.append(pair_name)
            
            for ev in state.get("next_events_10d", []):
                all_upcoming.append({
                    "pair": pair_name,
                    "leg": ev.get("leg", ""),
                    "type": ev["type"].upper(),
                    "event_date": ev["event_date"],
                    "days_to": ev.get("trading_days_to", ""),
                    "label": ev.get("label", ""),
                })
        except Exception:
            pass
    
    # Only show panel if there's event data
    has_event_data = len(active_multiplier_pairs) > 0 or len(all_upcoming) > 0
    panel_label = "📅 Event Risk"
    if active_multiplier_pairs:
        panel_label += f" — {len(active_multiplier_pairs)} active"
    if blocked_pairs:
        panel_label += f" | {len(blocked_pairs)} blocked"
    
    with st.expander(panel_label, expanded=len(active_multiplier_pairs) > 0):
        if not has_event_data:
            st.caption("No event risk data. Event tables may not be populated yet (run engine to refresh).")
        else:
            # --- Active Multipliers ---
            if active_multiplier_pairs:
                st.write("### 🔴 Active Multipliers")
                act_df = pd.DataFrame(active_multiplier_pairs)
                
                # Ensure column exists
                if "is_overridden" not in act_df.columns:
                    act_df["is_overridden"] = False

                # Editable DataFrame
                edited_act = st.data_editor(
                    act_df,
                    column_config={
                        "pair": st.column_config.Column("Pair", disabled=True),
                        "multiplier": st.column_config.NumberColumn("Multiplier", format="%.2f", min_value=0.0, step=0.1, help="Set 0.0 to clear override."),
                        "entry_blocked": st.column_config.CheckboxColumn("Entry Blocked"),
                        "events": st.column_config.Column("Active Events", disabled=True),
                        "is_overridden": st.column_config.CheckboxColumn("Overridden?", disabled=True)
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="event_risk_editor",
                    disabled=["pair", "events", "is_overridden"] 
                )
                
                # Detect Changes
                try:
                    changes = []
                    # Align indices to compare
                    act_indexed = act_df.set_index("pair")
                    edited_indexed = edited_act.set_index("pair")
                    
                    for pair, row in edited_indexed.iterrows():
                        if pair not in act_indexed.index: continue
                        old_row = act_indexed.loc[pair]
                        
                        # Compare floats with tolerance
                        val_changed = abs(float(row["multiplier"]) - float(old_row["multiplier"])) > 0.001
                        blk_changed = bool(row["entry_blocked"]) != bool(old_row["entry_blocked"])
                        
                        if val_changed or blk_changed:
                            changes.append({
                                "pair": pair, 
                                "multiplier": float(row["multiplier"]), 
                                "entry_blocked": bool(row["entry_blocked"])
                            })
                    
                    if changes:
                        import event_risk
                        for change in changes:
                            val = change["multiplier"]
                            blk = change["entry_blocked"]
                            if val == 0.0:
                                event_risk.set_event_override(con, change["pair"], multiplier=None, entry_blocked=None)
                            else:
                                event_risk.set_event_override(con, change["pair"], multiplier=val, entry_blocked=blk)
                        
                        st.success(f"Saved {len(changes)} override(s). Refreshing...")
                        time.sleep(0.5)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Failed to save overrides: {e}")
            else:
                st.success("✅ No active event multipliers right now.")
            
            # --- Upcoming Events ---
            if all_upcoming:
                # Deduplicate: same event_date + leg + type
                seen_events = set()
                deduped = []
                for ev in all_upcoming:
                    key = (ev["event_date"], ev["leg"], ev["type"])
                    if key not in seen_events:
                        seen_events.add(key)
                        deduped.append(ev)
                deduped.sort(key=lambda x: (x.get("days_to", 99), x["event_date"]))
                
                st.write("### 📋 Upcoming Events (10 Trading Days)")
                up_df = pd.DataFrame(deduped[:30])  # Limit display
                
                st.dataframe(
                    up_df,
                    column_config={
                        "pair": "Pair",
                        "leg": "Ticker",
                        "type": "Event Type",
                        "event_date": "Event Date",
                        "days_to": st.column_config.NumberColumn("Trading Days To", format="%d"),
                        "label": "Label",
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No upcoming events in the next 10 trading days.")

except Exception as e:
    # Fail silently if event risk module not available
    pass

# Detailed Engine Status (Collapsed by default if healthy)
with st.expander("Engine Details", expanded=(sys_color!="green")):
    if not heartbeats:
        st.error("⚠️ No Engine Heartbeat Detected (Database empty?)")
    else:
        # Show status for each engine
        cols = st.columns(len(heartbeats))
        for i, hb in enumerate(heartbeats):
            engine_id = hb.get("engine_id", "Unknown")
            
            # Skip internal event caches
            if engine_id.endswith("_events") or hb.get("status") == "events_cache":
                continue

            status = hb.get("status", "Offline")
            ts = hb.get("timestamp", "")
            error = hb.get("error_message")
            
            # Color coding (logic repeated for detailed view)
            active_color = "off"
            
            # Re-check stale locally for display
            is_stale_local = False
            if ts:
                try:
                    ts_dt = pd.to_datetime(ts)
                    now_dt = datetime.now(timezone.utc)
                    if (now_dt - ts_dt).total_seconds() > 1200:
                        is_stale_local = True
                except: pass

            if is_stale_local:
                status = f"Offline (Last: {status})"
                active_color = "grey"
            elif status == "Running":
                active_color = "green"
            elif status == "Warning":
                active_color = "orange"
            elif status == "Error":
                active_color = "red"
            elif status == "Starting" or "Warmup" in status:
                active_color = "blue"
                
            with cols[i]:
                st.markdown(f"**{engine_id}**")
                st.caption(f":{active_color}[{status}]")
                st.caption(f"Last: {ts}")
                
                if error:
                    st.error(f"{error}")
                
                # Show meta details
                if hb.get("meta_json"):
                    try:
                        meta = json.loads(hb["meta_json"])
                        st.json(meta, expanded=False)
                    except:
                        pass

# ----------------------------
# DATA FETCHING
# ----------------------------
# 1. Snapshot Metrics (Z-scores)
metrics = db.get_latest_snapshot(con)

# 2. Open Positions
positions = db.get_open_positions(con)

# 3. PnL Summary
pnl_history = db.get_pnl_history(con, limit=1)
pnl_summary = pnl_history.iloc[0] if not pnl_history.empty else None

# 4. Parameters
params = db.get_pair_params(con)

# 5. Daily Performance
daily_df = db.get_daily_performance(con, limit=30)

# ----------------------------
# TOP SUMMARY (PNL & LIVE MARKET)
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Portfolio")
    if pnl_summary is not None:
        try:
            equity = float(pnl_summary.get("equity", 100000))
            realized = float(pnl_summary.get("realized_pnl", 0))
            unrealized = float(pnl_summary.get("unrealized_pnl", 0))
            
            # Calculate allocated capital from open positions
            allocated = 0.0
            if not positions.empty:
                # Sum of (qty1 * price1 + qty2 * price2) for each position
                for _, pos in positions.iterrows():
                    try:
                        q1 = float(pos.get("qty1", 0) or 0)
                        q2 = float(pos.get("qty2", 0) or 0)
                        p1 = float(pos.get("entry_price1", 0) or 0)
                        p2 = float(pos.get("entry_price2", 0) or 0)
                        allocated += (q1 * p1) + (q2 * p2)
                    except:
                        pass
            
            available = equity - allocated
            
            # Display metrics
            st.metric("Total Equity", f"${equity:,.2f}")
            st.metric("Available Equity", f"${available:,.2f}", 
                      delta=f"-${allocated:,.0f} allocated" if allocated > 0 else None)
            
            c1, c2 = st.columns(2)
            c1.metric("Realized P&L", f"${realized:+,.2f}")
            c2.metric("Unrealized P&L", f"${unrealized:+,.2f}")
        except Exception as e:
            st.error(f"Error parsing PnL: {e}")
    else:
        st.info("No PnL data available yet.")

with col2:
    st.subheader("Live Market (Snapshot)")
    if not metrics.empty:
        # Sort by absolute Z-score descending to show most extreme pairs
        m = metrics.copy()
        
        # Convert Time to NYC
        if "time" in m.columns:
            try:
                # Ensure UTC
                m["time"] = pd.to_datetime(m["time"], utc=True)
                # Convert to US/Eastern
                m["time"] = m["time"].dt.tz_convert('US/Eastern')
            except Exception as e:
                pass

        m["z"] = pd.to_numeric(m["z"], errors='coerce')
        m["abs_z"] = m["z"].abs()
        m = m.sort_values("abs_z", ascending=False).drop(columns=["abs_z"])
        st.dataframe(m, use_container_width=True, hide_index=True)
    else:
        st.info("Waiting for live metrics...")

st.divider()

# ----------------------------
# OPEN POSITIONS (Full Width)
# ----------------------------
st.subheader("Open Positions")
if not positions.empty:
    # Select key columns for display
    if "entry_time" in positions.columns:
        try:
            # 1. Parse to UTC datetime (Robust)
            # Use format='mixed' to handle potential inconsistencies in string inputs
            positions["entry_time_utc"] = pd.to_datetime(positions["entry_time"], format='mixed', utc=True)
            
            # 2. Calculate Holding Duration (UTC vs UTC)
            now_utc = datetime.now(timezone.utc)
            
            def calc_duration(entry_time_utc):
                if pd.isnull(entry_time_utc): 
                    return "N/A"
                try:
                    # Both are UTC, so subtraction is safe
                    diff = now_utc - entry_time_utc
                    total_seconds = int(diff.total_seconds())
                    
                    days = total_seconds // 86400
                    hours = (total_seconds % 86400) // 3600
                    minutes = (total_seconds % 3600) // 60
                    
                    if days > 0:
                        return f"{days}d {hours}h"
                    elif hours > 0:
                        return f"{hours}h {minutes}m"
                    else:
                        return f"{minutes}m"
                except Exception:
                    return "Error"

            positions["duration_str"] = positions["entry_time_utc"].apply(calc_duration)

            # 3. Convert to US/Eastern for Display
            positions["entry_time"] = positions["entry_time_utc"].dt.tz_convert('US/Eastern')
            
            # Create Display String
            positions["entry_time_str"] = positions["entry_time"].dt.strftime('%Y-%m-%d %H:%M %Z')

            # Merge with metrics to get Beta Drift
            if not metrics.empty:
                metrics_drift = metrics[["pair", "beta_drift_pct"]].copy()
                positions = pd.merge(positions, metrics_drift, on="pair", how="left")
            else:
                positions["beta_drift_pct"] = None

            # Calculate Equity
            def calc_equity(row):
                try:
                    q1 = float(row.get("qty1", 0) or 0)
                    q2 = float(row.get("qty2", 0) or 0)
                    p1 = float(row.get("last_price1", 0) or 0)
                    p2 = float(row.get("last_price2", 0) or 0)
                    return abs(q1 * p1) + abs(q2 * p2)
                except:
                    return 0.0

            positions["equity"] = positions.apply(calc_equity, axis=1)

            # Display Formatting
            positions["beta_drift_display"] = positions["beta_drift_pct"].apply(
                lambda x: f"{abs(x):.1f}%" if pd.notnull(x) else "N/A"
            )
            
            # Ensure limit is float
            if "beta_drift_limit" not in positions.columns:
                positions["beta_drift_limit"] = 10.0
            positions["beta_drift_limit"] = positions["beta_drift_limit"].fillna(10.0).astype(float)
            
        except Exception as e:
            # Fallback if something critically fails (e.g. no entry_time column)
            positions["entry_time_str"] = str(e)
            positions["duration_str"] = "Err"
    
    display_cols = ["pair", "duration_str", "beta_drift_display", "equity", "pnl_unrealized", "entry_time_str", "beta_drift_limit"]
    # Filter to columns that actually exist
    valid_cols = [c for c in display_cols if c in positions.columns]
    
    # Use data_editor to allow editing limits
    edited_positions = st.data_editor(
        positions[valid_cols], 
        use_container_width=True, 
        hide_index=True,
        key="open_positions_editor",
        column_config={
            "pair": st.column_config.Column("Pair", disabled=True),
            "duration_str": st.column_config.Column("Duration", disabled=True),
            "beta_drift_display": st.column_config.Column("Drift %", disabled=True),
            "equity": st.column_config.NumberColumn("Equity", format="$%.0f", disabled=True),
            "pnl_unrealized": st.column_config.NumberColumn("Unrealized P&L", format="$%.2f", disabled=True),
            "entry_time_str": st.column_config.TextColumn("Entry Time (NY)", disabled=True),
            "beta_drift_limit": st.column_config.NumberColumn("Drift Limit %", min_value=1.0, max_value=100.0, step=0.5, format="%.1f%%", help="Close position if drift exceeds this %")
        }
    )
    
    # Check for edits
    if not positions.empty:
        # Detect changes in beta_drift_limit
        # We compare the edited dataframe with the original for the limit column
        # Note: st.data_editor returns the full dataframe state
        
        # Identify rows where limit changed
        # We need a stable index or ID. 'pair' is unique.
        # Merge old and new on pair to compare
        merged_edits = pd.merge(positions[["pair", "beta_drift_limit"]], 
                               edited_positions[["pair", "beta_drift_limit"]], 
                               on="pair", suffixes=("_old", "_new"))
        
        updates = []
        for _, row in merged_edits.iterrows():
            if abs(row["beta_drift_limit_old"] - row["beta_drift_limit_new"]) > 0.01:
                updates.append({
                    "pair": row["pair"],
                    "beta_drift_limit": row["beta_drift_limit_new"]
                })
        
        if updates:
            try:
                db.update_position_limits(con, updates)
                st.toast(f"Updated limits for {len(updates)} position(s)!", icon="💾")
                time.sleep(1) # Give time for toast
                st.rerun()
            except Exception as e:
                st.error("Failed to save limits: " + str(e))

    st.divider()
    st.markdown("### Manual Actions")
    col1, col2 = st.columns([3, 1])
    with col1:
        pair_to_close = st.selectbox("Select Position to Close", options=positions["pair"].tolist(), key="manual_close_select")
    with col2:
        # Add some vertical spacing to align button
        st.write("") 
        st.write("")
        if st.button("Close Position", type="primary", use_container_width=True):
            if pair_to_close:
                try:
                    db.add_manual_command(con, "CLOSE_POSITION", {"pair": pair_to_close})
                    st.success(f"Close request sent for {pair_to_close}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to send close request: {e}")

else:
    st.caption("No open positions.")

st.divider()

# ----------------------------
# DAILY PERFORMANCE
# ----------------------------
st.subheader("Daily Performance")

if not daily_df.empty:
    # Prepare data
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date")
    
    # Layout: Chart + Table
    d_chk1, d_chk2 = st.columns([2, 1])
    
    with d_chk1:
        # PnL Bar Chart
        fig_pnl = go.Figure()
        
        # Color bars based on PnL
        colors = ["#2e7d32" if v >= 0 else "#d32f2f" for v in daily_df["realized_pnl"]]
        
        fig_pnl.add_trace(go.Bar(
            x=daily_df["date"],
            y=daily_df["realized_pnl"],
            marker_color=colors,
            name="Daily P&L",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>PnL: $%{y:,.2f}<extra></extra>"
        ))
        
        # Add Equity Line (Secondary Axis)
        fig_pnl.add_trace(go.Scatter(
            x=daily_df["date"],
            y=daily_df["total_equity"],
            mode="lines+markers",
            line=dict(color="white", width=2),
            marker=dict(size=6),
            name="Equity",
            yaxis="y2",
            hovertemplate="Equity: $%{y:,.2f}<extra></extra>"
        ))
        
        fig_pnl.update_layout(
            title="Daily P&L & Equity",
            xaxis_title="Date",
            yaxis_title="Realized P&L",
            yaxis2=dict(
                title="Total Equity",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(orientation="h", y=1.1),
            height=350,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_pnl, use_container_width=True)
        
    with d_chk2:
        st.write("### Recent Days")
        # Format for table
        disp_daily = daily_df.sort_values("date", ascending=False).copy()
        
        st.dataframe(
            disp_daily,
            column_order=["date", "realized_pnl", "num_trades", "total_equity"],
            column_config={
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "realized_pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
                "num_trades": st.column_config.NumberColumn("Trades"),
                "total_equity": st.column_config.NumberColumn("Equity", format="$%.0f"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Cumulative Stats
        total_pnl = daily_df["realized_pnl"].sum()
        total_trades = daily_df["num_trades"].sum()
        win_rate = 0.0
        if total_trades > 0:
            total_wins = daily_df["wins"].sum()
            win_rate = 100 * total_wins / total_trades
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Period P&L", f"${total_pnl:+,.2f}")
        col_m2.metric("Total Trades", f"{total_trades}") 
        col_m3.metric("Win Rate", f"{win_rate:.1f}%")

else:
    st.info("No daily performance data recorded yet. Data will appear after market close.")

st.divider()

# ----------------------------
# CHARTS & HISTORY
# ----------------------------
st.subheader("Pair Analysis")

# Pair Selector
all_pairs = []
if not params.empty:
    all_pairs = sorted(params["pair"].unique())
elif not metrics.empty:
    all_pairs = sorted(metrics["pair"].unique())

# Handle Jump To Override
pair_index = 0
if "selected_pair_override" in st.session_state:
    override = st.session_state["selected_pair_override"]
    if override in all_pairs:
        pair_index = all_pairs.index(override)
    # Clear after use so user can switch away normally
    del st.session_state["selected_pair_override"]

selected_pair = st.selectbox("Select Pair", all_pairs, index=pair_index) if all_pairs else None

if selected_pair:
    # --- Per-Pair Event Risk Detail ---
    try:
        from event_risk import compute_pair_event_state
        from zoneinfo import ZoneInfo as _ZI2
        
        _now_et = datetime.now(_ZI2("America/New_York"))
        pair_event = compute_pair_event_state(selected_pair, _now_et, con)
        
        if pair_event["multiplier"] > 1.0 or pair_event["entry_blocked"] or pair_event["next_events_10d"]:
            ev_col1, ev_col2, ev_col3 = st.columns(3)
            
            with ev_col1:
                mult_val = pair_event["multiplier"]
                mult_color = "🟢" if mult_val == 1.0 else ("🟡" if mult_val < 1.5 else "🔴")
                st.metric("Event Multiplier", f"{mult_val:.2f}x", delta=f"{mult_color}")
            
            with ev_col2:
                if pair_event["entry_blocked"]:
                    st.metric("Entry Status", "🔒 BLOCKED")
                else:
                    st.metric("Entry Status", "✅ Allowed")
            
            with ev_col3:
                events_str = ", ".join(
                    e.get("window_label", e["type"]) for e in pair_event["active_events"]
                ) if pair_event["active_events"] else "None"
                st.metric("Active Events", events_str[:40])
            
            # Next events for this pair
            if pair_event["next_events_10d"]:
                with st.expander(f"📅 Upcoming Events for {selected_pair}", expanded=False):
                    next_df = pd.DataFrame(pair_event["next_events_10d"])
                    st.dataframe(
                        next_df,
                        column_config={
                            "type": "Type",
                            "leg": "Ticker",
                            "event_date": "Date",
                            "trading_days_to": st.column_config.NumberColumn("Days To", format="%d"),
                            "label": "Label",
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
    except Exception:
        pass
    
    # Fetch 75 days of history (time-based, not bar count)
    # Use high limit and filter by timestamp
    history_raw = db.get_pair_history(con, selected_pair, timeframe="15m", limit=3000)
    
    if not history_raw.empty:
        # Ensure Types & Convert to UTC
        history_raw["time"] = pd.to_datetime(history_raw["time"], errors="coerce", utc=True)
        history_raw = history_raw.sort_values("time")
        
        # Filter to last 75 calendar days
        cutoff_75d = datetime.now(timezone.utc) - timedelta(days=75)
        cutoff_30d = datetime.now(timezone.utc) - timedelta(days=30)
        
        history = history_raw[history_raw["time"] >= cutoff_75d].copy()
        
        if history.empty:
            st.warning(f"No history found for {selected_pair} in last 75 days.")
        else:
            # Convert to Eastern for display labels
            history["time_ny"] = history["time"].dt.tz_convert('US/Eastern')
            
            # Detect Day Boundaries (for vertical lines)
            history["day_str"] = history["time_ny"].dt.strftime('%Y-%m-%d')
            day_changes = history[history["day_str"] != history["day_str"].shift(1)].iloc[1:]
            day_change_times = day_changes["time"].tolist()

            # Layout: Chart + Stats
            chk1, chk2 = st.columns([2, 1])
            
            with chk1:
                # --- Z-Score Chart (Enhanced) ---
                st.markdown(f"### {selected_pair} Z-Score History")
                
                # Get pair params
                z_entry = 2.0  # Default
                z_exit = 0.5   # Default
                if not params.empty:
                    param_row = params[params["pair"] == selected_pair]
                    if not param_row.empty:
                        z_entry = param_row.iloc[0]["z_entry"]
                        z_exit = param_row.iloc[0]["z_exit"]
                
                # Split into older (>30d) and recent (<=30d) segments
                history_old = history[history["time"] < cutoff_30d]
                history_recent = history[history["time"] >= cutoff_30d]
                
                # Compute dynamic y-range with padding
                z_vals = history["z"].dropna()
                if not z_vals.empty:
                    y_min = min(z_vals.min(), -z_entry) - 0.5
                    y_max = max(z_vals.max(), z_entry) + 0.5
                else:
                    y_min, y_max = -4, 4
                
                # Build figure with graph_objects for full control
                fig = go.Figure()
                
                # 1. Context Shading Zones (drawn first, behind everything)
                # Grey No-Trade Zone: between -z_entry and +z_entry
                fig.add_hrect(
                    y0=-z_entry, y1=z_entry,
                    fillcolor="rgba(128, 128, 128, 0.15)",
                    layer="below", line_width=0,
                    annotation_text="No-Trade Zone", annotation_position="top left",
                    annotation=dict(font_size=10, font_color="gray")
                )
                # Green Trade Zone: above +z_entry (dynamic range)
                fig.add_hrect(
                    y0=z_entry, y1=y_max,
                    fillcolor="rgba(76, 175, 80, 0.12)",
                    layer="below", line_width=0
                )
                # Green Trade Zone: below -z_entry (dynamic range)
                fig.add_hrect(
                    y0=y_min, y1=-z_entry,
                    fillcolor="rgba(76, 175, 80, 0.12)",
                    layer="below", line_width=0
                )
                
                # 2. Older history line (thin, faded) - if exists
                if not history_old.empty:
                    fig.add_trace(go.Scatter(
                        x=history_old["time"],
                        y=history_old["z"],
                        mode="lines",
                        line=dict(color="steelblue", width=1),
                        opacity=0.4,
                        name="Prior 45d",
                        hovertemplate="Z: %{y:.2f}<br>%{x}<extra></extra>"
                    ))
                
                # 3. Recent 30-day line (thick, bright)
                if not history_recent.empty:
                    fig.add_trace(go.Scatter(
                        x=history_recent["time"],
                        y=history_recent["z"],
                        mode="lines",
                        line=dict(color="dodgerblue", width=1.8),
                        name="Recent 30d",
                        hovertemplate="Z: %{y:.2f}<br>%{x}<extra></extra>"
                    ))
                
                # 4. Threshold Lines
                fig.add_hline(y=z_entry, line_dash="dash", line_color="green", 
                              annotation_text=f"+Entry ({z_entry})", annotation_position="right")
                fig.add_hline(y=-z_entry, line_dash="dash", line_color="green",
                              annotation_text=f"-Entry ({z_entry})", annotation_position="right")
                fig.add_hline(y=z_exit, line_dash="dot", line_color="orangered", 
                              annotation_text=f"+Exit ({z_exit})", annotation_position="right")
                fig.add_hline(y=-z_exit, line_dash="dot", line_color="orangered",
                              annotation_text=f"-Exit ({z_exit})", annotation_position="right")
                
                # 5. Entry Signal Markers (detect crossings)
                signals = detect_entry_signals(history, z_entry)
                is_blocked, block_reason = get_pair_blocking_status(selected_pair, raw_issues)
                
                # Plot entry markers - blue circles for valid, red X for blocked (current state)
                if signals:
                    sig_times = [s["time"] for s in signals]
                    sig_zs = [s["z"] for s in signals]
                    sig_dirs = [s["direction"] for s in signals]
                    
                    if is_blocked:
                        # Red X markers for blocked state
                        hover_texts = [f"Entry Signal (Blocked)<br>Reason: {block_reason}<br>Direction: {d}<br>Z: {z:.2f}" 
                                       for d, z in zip(sig_dirs, sig_zs)]
                        fig.add_trace(go.Scatter(
                            x=sig_times,
                            y=sig_zs,
                            mode="markers",
                            marker=dict(size=12, color="red", symbol="x",
                                        line=dict(width=2, color="darkred")),
                            showlegend=False,
                            hovertemplate="%{text}<extra></extra>",
                            text=hover_texts
                        ))
                    else:
                        # Blue circle markers for valid state
                        hover_texts = [f"Entry Signal<br>Direction: {d}<br>Z: {z:.2f}" 
                                       for d, z in zip(sig_dirs, sig_zs)]
                        fig.add_trace(go.Scatter(
                            x=sig_times,
                            y=sig_zs,
                            mode="markers",
                            marker=dict(size=10, color="blue", symbol="circle",
                                        line=dict(width=1, color="white")),
                            showlegend=False,
                            hovertemplate="%{text}<extra></extra>",
                            text=hover_texts
                        ))
                
                # 6. Day separators (subtle vertical lines)
                for dt in day_change_times:
                    fig.add_vline(x=dt, line_width=1, line_dash="dash", 
                                  line_color="grey", opacity=0.3)
                
                # 7. Current State Annotation (top-right)
                if not history.empty:
                    current_z = history.iloc[-1]["z"]
                    abs_z = abs(current_z) if pd.notnull(current_z) else 0
                    zone = "Trade" if abs_z >= z_entry else "No-Trade"
                    
                    if is_blocked:
                        status = f"Blocked ({block_reason})"
                        status_color = "#ff5252"
                    else:
                        status = "Valid"
                        status_color = "#69f0ae"
                    
                    annotation_text = (
                        f"<b>Current |Z|:</b> {abs_z:.2f}<br>"
                        f"<b>Zone:</b> {zone}<br>"
                        f"<b>Status:</b> <span style='color:{status_color}'>{status}</span>"
                    )
                    
                    fig.add_annotation(
                        x=1, y=1, xref="paper", yref="paper",
                        xanchor="right", yanchor="top",
                        text=annotation_text,
                        showarrow=False,
                        bgcolor="rgba(30,30,30,0.85)",
                        font=dict(color="white", size=11),
                        bordercolor="rgba(255,255,255,0.3)",
                        borderwidth=1, borderpad=8,
                        align="left"
                    )
                
                # Layout
                fig.update_layout(
                    title="Z-Score History (75 Days)",
                    xaxis_title="",
                    yaxis_title="Z-Score",
                    yaxis=dict(range=[y_min, y_max]),
                    xaxis=dict(tickangle=-45),
                    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                    margin=dict(r=100),  # Room for annotations
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True, 
                                config={'edits': {'annotationPosition': True}})

                # --- FAST vs SLOW Z-SCORE PANEL ---
                with st.expander("📊 Fast vs Slow Z-Score (1m vs 15m)", expanded=False):
                    st.caption("Compare 1-minute z-scores (used for entries) vs 15-minute z-scores (visualization).")
                    
                    # Configurable window
                    window_options = {30: "30 min", 60: "1 hour", 120: "2 hours", 240: "4 hours"}
                    window_mins = st.selectbox(
                        "Time Window", 
                        list(window_options.keys()), 
                        format_func=lambda x: window_options[x],
                        index=2,  # Default 120 min
                        key="z_compare_window"
                    )
                    
                    # Get 1m series data for this pair
                    try:
                        # Fetch all recent data (don't filter by time in SQL - timezone mismatch issues)
                        series_1m = db.get_1m_series(con, selected_pair, limit=1000)
                        
                        if series_1m.empty:
                            st.info("No 1-minute z-score data available yet. Data will appear after the engine runs.")
                        else:
                            # Process 1m data - parse to UTC for consistent comparison
                            series_1m["time"] = pd.to_datetime(series_1m["time"], errors="coerce", utc=True)
                            series_1m = series_1m.dropna(subset=["time"]).sort_values("time")
                            
                            # Filter by window in pandas (after timezone normalization)
                            cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_mins)
                            series_1m = series_1m[series_1m["time"] >= cutoff]
                            
                            if series_1m.empty:
                                st.info(f"No 1-minute data in the last {window_mins} minutes. Try a larger window.")
                            else:
                                # Get matching 15m window from history
                                window_start = series_1m["time"].min()
                                window_end = series_1m["time"].max()
                            
                                history_window = history[(history["time"] >= window_start) & (history["time"] <= window_end)] if window_start and not history.empty else pd.DataFrame()
                            
                                # Build comparison chart
                                fig_compare = go.Figure()
                            
                                # 1m z-score line (primary)
                                fig_compare.add_trace(go.Scatter(
                                    x=series_1m["time"],
                                    y=series_1m["z_1m"],
                                    mode="lines+markers",
                                    line=dict(color="dodgerblue", width=1.5),
                                    marker=dict(size=4),
                                    name="Z-Score (1m)",
                                    hovertemplate="1m Z: %{y:.2f}<br>%{x}<extra></extra>"
                                ))
                            
                                # 15m z-score line (for comparison)
                                if not history_window.empty:
                                    fig_compare.add_trace(go.Scatter(
                                        x=history_window["time"],
                                        y=history_window["z"],
                                        mode="lines+markers",
                                        line=dict(color="orange", width=2, dash="dot"),
                                        marker=dict(size=6, symbol="square"),
                                        name="Z-Score (15m)",
                                        hovertemplate="15m Z: %{y:.2f}<br>%{x}<extra></extra>"
                                    ))
                            
                                # Entry thresholds
                                fig_compare.add_hline(y=z_entry, line_dash="dash", line_color="green", opacity=0.6)
                                fig_compare.add_hline(y=-z_entry, line_dash="dash", line_color="green", opacity=0.6)
                            
                                # Mark persistence filter pass/fail
                                passed = series_1m[series_1m["passed_persistence"] == 1]
                                failed = series_1m[series_1m["passed_persistence"] == 0]
                            
                                if not passed.empty:
                                    fig_compare.add_trace(go.Scatter(
                                        x=passed["time"], y=passed["z_1m"],
                                        mode="markers", marker=dict(size=8, color="green", symbol="circle"),
                                        name="Persistence ✓", hovertemplate="PASSED<extra></extra>"
                                    ))
                            
                                fig_compare.update_layout(
                                    title="1-Minute vs 15-Minute Z-Score Comparison",
                                    xaxis_title="Time", yaxis_title="Z-Score",
                                    legend=dict(orientation="h", y=-0.15),
                                    height=350
                                )
                            
                                st.plotly_chart(fig_compare, use_container_width=True)
                            
                                # Summary metrics
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("1m Samples", len(series_1m))
                                with col_b:
                                    avg_z = series_1m["z_1m"].mean() if not series_1m.empty else 0
                                    st.metric("Avg 1m Z", f"{avg_z:.2f}")
                                with col_c:
                                    pct_passed = 100 * len(passed) / len(series_1m) if len(series_1m) > 0 else 0
                                    st.metric("Persistence %", f"{pct_passed:.0f}%")
                                
                    except Exception as e:
                        st.error(f"Error loading 1m data: {e}")

            # Beta Chart
                st.markdown(f"### {selected_pair} Beta Drift")
                fig_beta = px.line(history, x="time", y=["direct_beta", "beta_30_weekly"], 
                                 title="Direct vs 30D Beta", color_discrete_sequence=["blue", "orange"])
                
                fig_beta.update_xaxes(tickangle=-45)
                
                for dt in day_change_times:
                    fig_beta.add_vline(x=dt, line_width=1, line_dash="dash", line_color="grey", opacity=0.5)
                    
                st.plotly_chart(fig_beta, use_container_width=True)

            with chk2:
                st.markdown("### Z-Score Distribution (Recent Regime)")
                st.caption("10w base + 4w overlay")
                
                # Get current z value
                current_z_val = history.iloc[-1]["z"] if not history.empty else None
                
                # UI controls
                col_bins, col_clip = st.columns(2)
                with col_bins:
                    n_bins = st.select_slider("Bins", options=[20, 30, 40], value=30, key="hist_bins")
                with col_clip:
                    clip_z = st.checkbox("Clip to ±4", value=True, key="hist_clip")
                
                z_range = (-4, 4) if clip_z else None
                
                # Check sample size before plotting (warning in UI layer)
                cutoff_10w_check = history["time"].max() - timedelta(days=70) if not history.empty else None
                if cutoff_10w_check is not None:
                    z10_count = len(history[history["time"] >= cutoff_10w_check]["z"].dropna())
                    if z10_count < 100:
                        st.warning(f"⚠️ Not enough recent data for reliable histogram ({z10_count} samples, need 100+)")
                
                # Plot histogram
                fig_hist = plot_z_histogram(history, z_entry, n_bins, z_range, current_z_val)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Current z metric (optional but trader-useful)
                if current_z_val is not None:
                    st.metric("Current Z", f"{current_z_val:.2f}")
                
                st.markdown("### Recent Data")
                st.dataframe(history.tail(10).sort_values("time", ascending=False), use_container_width=True, hide_index=True)

    else:
        st.warning(f"No history found for {selected_pair}.")
        
    st.divider()
    st.subheader("Long Term Analysis (30 Days)")
    
    # Fetch Long History for this pair (reuse existing history if available, or fetch fresh)
    # Since Z-score chart already fetches 75 days, we can use that for consistency
    history_long = db.get_pair_history(con, selected_pair, timeframe="15m", limit=3000)
    
    if not history_long.empty:
        # Process Long History
        history_long["time"] = pd.to_datetime(history_long["time"], errors="coerce", utc=True)
        history_long = history_long.sort_values("time")
        
        # Filter to last 30 days
        cutoff_30d_long = datetime.now(timezone.utc) - timedelta(days=30)
        history_long = history_long[history_long["time"] >= cutoff_30d_long]
        
        # Convert to Eastern
        history_long["time_ny"] = history_long["time"].dt.tz_convert('US/Eastern')
        
        # Day boundaries for long chart
        history_long["day_str"] = history_long["time_ny"].dt.strftime('%Y-%m-%d')
        day_changes_long = history_long[history_long["day_str"] != history_long["day_str"].shift(1)].iloc[1:]
        day_change_times_long = day_changes_long["time"].tolist()

        # 30-Day Beta Chart
        st.markdown(f"### {selected_pair} Beta Drift (30 Days)")
        fig_beta_long = px.line(history_long, x="time", y=["direct_beta", "beta_30_weekly"], 
                          title="Direct vs 30D Beta (30 Day View)", color_discrete_sequence=["blue", "orange"])
        
        # Sparse ticks for long view
        fig_beta_long.update_xaxes(tickangle=-45, nticks=20)
        
        # Add Day Separators (only if reasonable number, say < 50)
        if len(day_change_times_long) < 60:
             for dt in day_change_times_long:
                fig_beta_long.add_vline(x=dt, line_width=1, line_dash="dash", line_color="grey", opacity=0.3)
            
        st.plotly_chart(fig_beta_long, use_container_width=True)
    else:
        st.info("Not enough history for long-term view.")


st.divider()

# ----------------------------
# PARAMETER EDITOR
# ----------------------------
with st.expander("⚙️ Configuration (Pair Parameters)"):
    if params.empty:
        st.warning("No parameters found. Please run seed script or engine init.")
    else:
        # Prepare for editor
        p_edit = params.copy()
        # Ensure enabled is bool for checkbox
        if "enabled" in p_edit.columns:
            p_edit["enabled"] = p_edit["enabled"].astype(int).astype(bool)
            
        edited_df = st.data_editor(
            p_edit.sort_values("pair"), 
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "enabled": st.column_config.CheckboxColumn("Enabled", help="Toggle trading for this pair"),
                "max_drift_pct": st.column_config.NumberColumn("Beta Drift (%)", help="Max allowed % deviation from entry Beta"),
                "max_drift_delta": st.column_config.NumberColumn("Delta Drift ($)", help="Max allowed $ deviation (unhedged exposure)")
            }
        )
        
        if st.button("Save Changes"):
            try:
                # Convert back to int for DB
                to_save = edited_df.copy()
                to_save["enabled"] = to_save["enabled"].astype(int)
                
                db.upsert_pair_params(con, to_save)
                con.commit()  # Explicit commit to ensure WAL is flushed
                st.success("Configuration saved! Reloading...")
                time.sleep(0.5)  # Brief visual feedback
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save settings: {e}")

    # ----------------------------
    # Bulk Update Section
    # ----------------------------
    st.divider()
    st.subheader("Bulk Parameter Update")
    st.info("Update a specific parameter for **ALL enabled pairs** at once.")
    
    with st.form("bulk_update_form"):
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            # Only allow bulk update for numeric risk params
            bulk_param = st.selectbox(
                "Parameter", 
                ["z_entry", "z_exit", "max_drift_pct", "max_drift_delta", "alloc_pct"]
            )
            
        with col_b2:
            bulk_val = st.number_input("New Value", value=0.0, step=0.1, format="%.2f")
            
        with col_b3:
            st.write("") # Spacer
            st.write("") # Spacer
            bulk_submit = st.form_submit_button("Apply to All Enabled Pairs")
            
        if bulk_submit:
            try:
                with con:
                    con.execute(f"UPDATE pair_params SET {bulk_param} = ? WHERE enabled = 1", (bulk_val,))
                    con.commit()
                st.success(f"Updated {bulk_param} to {bulk_val} for all enabled pairs!")
                time.sleep(1.0)
                st.rerun()
            except Exception as e:
                st.error(f"Bulk update failed: {e}")

# Close DB connection at end of script run (Streamlit runs script top-to-bottom)
# Note: In production Streamlit, caching connection resource is better, 
# but for low-traffic local dashboard this is fine.
con.close()
