"""
dashboard_jp.py

Japanese Pairs Trading Dashboard — renders inside the JP tab of app.py.
Mirrors the US dashboard layout but reads from jp_* DB tables,
uses JPY formatting, and references JST timezone.
"""
from __future__ import annotations

import time
import json
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import db
import auth

# Reuse Bloomberg theme from shared module
from theme import (
    BLOOMBERG_LAYOUT, BB_CYAN, BB_AMBER, BB_GREEN, BB_RED,
    BB_BLUE, BB_GREY, BB_LIGHT, BB_DIM, bb_layout,
)

# ---------------------------------------------------------------------------
# JP MARKET STATUS
# ---------------------------------------------------------------------------

def get_jp_market_status() -> dict:
    """Returns Tokyo Stock Exchange market status."""
    import pytz
    jst = pytz.timezone("Asia/Tokyo")
    now_jst = datetime.now(jst)

    is_weekday = now_jst.weekday() < 5

    # TSE hours: Morning 09:00–11:30, Afternoon 12:30–15:00 JST
    morning_open = now_jst.replace(hour=9, minute=0, second=0, microsecond=0)
    morning_close = now_jst.replace(hour=11, minute=30, second=0, microsecond=0)
    afternoon_open = now_jst.replace(hour=12, minute=30, second=0, microsecond=0)
    afternoon_close = now_jst.replace(hour=15, minute=0, second=0, microsecond=0)

    is_open = is_weekday and (
        (morning_open <= now_jst <= morning_close) or
        (afternoon_open <= now_jst <= afternoon_close)
    )

    # Distinguish mid-day break from fully closed
    is_lunch_break = is_weekday and (morning_close < now_jst < afternoon_open)

    if is_open:
        text, color = "OPEN", "green"
    elif is_lunch_break:
        text, color = "MID-DAY BREAK", "orange"
    else:
        text, color = "CLOSED", "grey"

    return {
        "is_open": is_open,
        "text": text,
        "color": color,
    }


# ---------------------------------------------------------------------------
# ISSUE DETECTION (mirrors US logic)
# ---------------------------------------------------------------------------

def get_jp_system_issues(con, heartbeats) -> list:
    issues = []
    jp_hbs = [h for h in heartbeats if "JP" in h.get("engine_id", "").upper()]

    if not jp_hbs:
        issues.append({
            "key": "jp_sys:no_heartbeat", "type": "system", "severity": "red",
            "pair": None, "title": "No JP Engine Found",
            "value": "Zero JP Heartbeats", "threshold": None, "status": "blocked",
        })
        return issues

    for hb in jp_hbs:
        ts = hb.get("timestamp", "")
        status = hb.get("status", "Offline")
        engine_id = hb.get("engine_id", "Unknown")

        if engine_id.endswith("_events") or status == "events_cache":
            continue

        if ts:
            try:
                ts_dt = pd.to_datetime(ts)
                now_dt = datetime.now(timezone.utc)
                if (now_dt - ts_dt).total_seconds() > 1200:
                    issues.append({
                        "key": f"jp_sys:{engine_id}:stale", "type": "system",
                        "severity": "red", "pair": None,
                        "title": f"Engine {engine_id} Stale",
                        "value": f"Last: {ts}", "threshold": "20 min",
                        "status": "blocked",
                    })
            except Exception:
                pass
    return issues


def get_jp_data_issues(con, heartbeats, is_market_open) -> list:
    issues = []
    if not is_market_open:
        return issues

    jp_hbs = [h for h in heartbeats if "JP" in h.get("engine_id", "").upper()
              and not h.get("engine_id", "").endswith("_events")]

    for hb in jp_hbs:
        ts = hb.get("timestamp", "")
        if ts:
            try:
                ts_dt = pd.to_datetime(ts)
                now_dt = datetime.now(timezone.utc)
                age_s = (now_dt - ts_dt).total_seconds()
                if age_s > 960:  # 16 minutes
                    issues.append({
                        "key": f"jp_data:{hb['engine_id']}:stale", "type": "data",
                        "severity": "red", "pair": None,
                        "title": f"JP Data Stale ({hb['engine_id']})",
                        "value": f"{int(age_s/60)}m old",
                        "threshold": "16 min", "status": "blocked",
                    })
            except Exception:
                pass
    return issues


def get_jp_risk_issues(con, params) -> list:
    issues = []
    if params.empty:
        return issues

    metrics = db.jp_get_latest_snapshot(con)
    if metrics.empty:
        return issues

    enabled = params[params["enabled"] == 1]
    merged = pd.merge(metrics, enabled, on="pair", how="inner")

    for _, row in merged.iterrows():
        drift = abs(row.get("beta_drift_pct", 0) or 0)
        max_drift = row.get("max_drift_pct", 20) or 20
        pair = row["pair"]

        if drift > max_drift:
            issues.append({
                "key": f"jp_risk:{pair}:drift", "type": "risk", "severity": "red",
                "pair": pair, "title": "Beta Drift Breach",
                "value": f"{drift:.1f}%", "threshold": f"{max_drift:.1f}%",
                "status": "blocked",
            })
        elif drift > 0.8 * max_drift:
            issues.append({
                "key": f"jp_risk:{pair}:drift_warn", "type": "risk", "severity": "amber",
                "pair": pair, "title": "Beta Drift Warning",
                "value": f"{drift:.1f}%", "threshold": f"{max_drift:.1f}%",
                "status": "warning",
            })
    return issues


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------

def calculate_jp_verdict(mkt, issues, candidates_count) -> tuple:
    red_issues = [i for i in issues if i["severity"] == "red"]
    if red_issues:
        return f"{len(red_issues)} ISSUES DETECTED", "white", "#d32f2f"
    if mkt["is_open"] and candidates_count > 0:
        return f"{candidates_count} TRADE CANDIDATES", "white", "#2e7d32"
    return "ALL SYSTEMS NOMINAL", "white", "#424242"


# ---------------------------------------------------------------------------
# SIGNAL / CHART HELPERS (same logic as US)
# ---------------------------------------------------------------------------

def detect_entry_signals(history_df, z_entry):
    signals = []
    in_trade_zone = False
    for _, row in history_df.iterrows():
        z = row.get("z", 0) or 0
        t = row.get("time")
        if abs(z) >= z_entry and not in_trade_zone:
            direction = "long" if z < 0 else "short"
            signals.append({"time": t, "z": z, "direction": direction})
            in_trade_zone = True
        elif abs(z) < z_entry and in_trade_zone:
            in_trade_zone = False
    return signals


def get_pair_blocking_status(pair, raw_issues):
    for issue in raw_issues:
        if issue.get("severity") == "red":
            if issue.get("type") == "system":
                return True, "System"
            if issue.get("type") == "data":
                return True, "Data"
    for issue in raw_issues:
        if issue.get("pair") == pair and issue.get("type") == "risk" and issue.get("severity") == "red":
            return True, "Risk"
    return False, None


def plot_z_histogram(history, z_entry, n_bins=30, z_clip_range=None, current_z=None):
    import numpy as np

    if history.empty or "time" not in history.columns or "z" not in history.columns:
        fig = go.Figure()
        fig.update_layout(**bb_layout())
        fig.add_annotation(text="NO DATA AVAILABLE", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=13, color=BB_GREY, family="IBM Plex Mono, monospace"))
        return fig

    now = history["time"].max()
    cutoff_10w = now - timedelta(days=70)
    cutoff_4w = now - timedelta(days=28)

    z10 = history[history["time"] >= cutoff_10w]["z"].dropna()
    z4 = history[history["time"] >= cutoff_4w]["z"].dropna()

    if len(z10) < 20:
        fig = go.Figure()
        fig.update_layout(**bb_layout())
        fig.add_annotation(text="INSUFFICIENT HISTORY FOR HISTOGRAM", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=12, color=BB_AMBER, family="IBM Plex Mono, monospace"))
        return fig

    if z_clip_range:
        z10 = z10.clip(z_clip_range[0], z_clip_range[1])
        z4 = z4.clip(z_clip_range[0], z_clip_range[1])

    bin_min = z10.min()
    bin_max = z10.max()
    bin_size = (bin_max - bin_min) / n_bins if bin_max > bin_min else 0.1

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=z10, name="10w", histnorm="probability density",
        xbins=dict(start=bin_min, end=bin_max, size=bin_size),
        marker_color=BB_CYAN, marker_line=dict(color="#0b0f14", width=0.5), opacity=0.85,
    ))
    if len(z4) >= 10:
        fig.add_trace(go.Histogram(
            x=z4, name="4w", histnorm="probability density",
            xbins=dict(start=bin_min, end=bin_max, size=bin_size),
            marker_color=BB_AMBER, marker_line=dict(color="#0b0f14", width=0.5), opacity=0.5,
        ))

    fig.add_vline(x=0, line_dash="solid", line_color=BB_GREY, line_width=1)
    fig.add_vline(x=z_entry, line_dash="dash", line_color=BB_GREEN, line_width=1.5,
                  annotation_text="+Entry", annotation_position="top right", annotation_font_color=BB_GREEN)
    fig.add_vline(x=-z_entry, line_dash="dash", line_color=BB_GREEN, line_width=1.5,
                  annotation_text="-Entry", annotation_position="top left", annotation_font_color=BB_GREEN)
    if current_z is not None:
        fig.add_vline(x=current_z, line_dash="solid", line_color=BB_RED, line_width=2,
                      annotation_text="Current", annotation_position="bottom right", annotation_font_color=BB_RED)

    fig.update_layout(**bb_layout(
        barmode="overlay", title=None, xaxis_title="Z-Score", yaxis_title="Density",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2332", borderwidth=1,
                    font=dict(size=10, color="#8899aa"), orientation="h", y=-0.15, x=0.5),
        margin=dict(t=20, b=50, l=50, r=20), height=300,
    ))
    return fig


# ---------------------------------------------------------------------------
# TRADE CANDIDATES (same logic as US)
# ---------------------------------------------------------------------------

def get_jp_trade_candidates(metrics, params, raw_issues, is_market_open=True, buffer=0.3):
    if params is None or params.empty or metrics is None or metrics.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    if "enabled" not in params.columns:
        return pd.DataFrame(), pd.DataFrame(), []

    enabled_params = params[params["enabled"] == 1].copy()
    if enabled_params.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    merged = pd.merge(metrics, enabled_params, on="pair", how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    merged["z"] = pd.to_numeric(merged["z"], errors='coerce').fillna(0)
    merged["abs_z"] = merged["z"].abs()
    merged["entry_threshold"] = merged["z_entry"] - buffer
    candidates = merged[merged["abs_z"] >= merged["entry_threshold"]].copy()

    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    candidates["to_entry"] = candidates["z_entry"] - candidates["abs_z"]

    now_utc = datetime.now(timezone.utc)
    def calc_time_since(row):
        ts = row.get("last_updated") or row.get("time")
        if not ts:
            return 60.0
        try:
            ts_dt = pd.to_datetime(ts, utc=True)
            return (now_utc - ts_dt).total_seconds() / 60.0
        except:
            return 60.0
    candidates["time_since_mins"] = candidates.apply(calc_time_since, axis=1)

    system_data_red = [i for i in raw_issues if i["type"] in ["system", "data"] and i["severity"] == "red"]
    has_global_block = len(system_data_red) > 0

    pair_issues = {}
    for issue in raw_issues:
        p = issue.get("pair")
        if p:
            pair_issues.setdefault(p, []).append(issue)

    def compute_risk_details(row):
        pair = row["pair"]
        drift_pct = abs(row.get("beta_drift_pct", 0) or 0)
        max_drift = row.get("max_drift_pct", 20) or 20
        abs_z = row["abs_z"]
        base_z_entry = row.get("z_entry", 3.5) or 3.5

        blocks = []

        drift_ok = drift_pct <= max_drift
        drift_display = f"{drift_pct:.1f}/{max_drift:.1f}%"
        drift_display = f"{'✓' if drift_ok else '⛔'} {drift_display}"
        if not drift_ok:
            blocks.append("Beta Drift")

        z_passes = abs_z >= base_z_entry
        z_vs_eff_display = f"{abs_z:.2f}/{base_z_entry:.2f}"
        z_vs_eff_display = f"{'✓' if z_passes else '⛔'} {z_vs_eff_display}"
        if not z_passes:
            blocks.append("Z < Entry")

        if has_global_block:
            blocks.append("Data/System")

        if has_global_block:
            status = "🔴 Blocked (Data)"
        elif pair in pair_issues and any(i["severity"] == "red" for i in pair_issues[pair]):
            status = "🔴 Blocked (Risk)"
        else:
            at_entry = row["to_entry"] <= 0
            amber_drift = drift_pct > 0.8 * max_drift
            signal_old = is_market_open and row["time_since_mins"] > 30
            if at_entry and not amber_drift and not signal_old:
                status = "✅ Valid"
            else:
                status = "🟡 Watch"

        block_reason = " | ".join(blocks) if blocks else "—"

        return pd.Series({
            "status": status, "drift_check": drift_display,
            "z_vs_eff": z_vs_eff_display, "block_reason": block_reason,
        })

    risk_details = candidates.apply(compute_risk_details, axis=1)
    for col in risk_details.columns:
        candidates[col] = risk_details[col]

    def calc_priority(row):
        abs_z = row["abs_z"]
        drift_pct = abs(row.get("beta_drift_pct", 0) or 0)
        max_drift = row.get("max_drift_pct", 0) or 0
        time_mins = row["time_since_mins"]
        drift_room = max(0, min(1, 1 - drift_pct / max_drift)) if max_drift > 0 else 0.5
        return 10 * abs_z + 5 * drift_room - 0.1 * time_mins

    candidates["priority_score"] = candidates.apply(calc_priority, axis=1)
    candidates = candidates.sort_values("priority_score", ascending=False).head(15)

    candidates["beta_drift_display"] = candidates["beta_drift_pct"].apply(
        lambda x: f"{abs(x):.1f}%" if pd.notnull(x) else "N/A")
    candidates["to_entry_display"] = candidates["to_entry"].apply(
        lambda x: f"{x:+.2f}" if pd.notnull(x) else "N/A")

    return candidates, candidates, []


# ===========================================================================
# MAIN RENDER FUNCTION
# ===========================================================================

def render(con):
    """Render the full JP Pairs dashboard inside the JP tab."""

    # ---- Init JP tables ----
    db.init_jp_db(con)

    # ---- Fetch data with retry ----
    heartbeats = []
    params = pd.DataFrame()
    metrics = pd.DataFrame()

    for i in range(3):
        try:
            heartbeats = db.get_heartbeat(con)
            params = db.jp_get_pair_params(con)
            metrics = db.jp_get_latest_snapshot(con)
            if not metrics.empty:
                break
            time.sleep(0.1)
        except Exception as e:
            if i == 2:
                st.error(f"JP DB Read Error: {e}")
            time.sleep(0.2)

    # ---- Issues ----
    mkt = get_jp_market_status()

    raw_issues = []
    raw_issues.extend(get_jp_system_issues(con, heartbeats))
    raw_issues.extend(get_jp_data_issues(con, heartbeats, is_market_open=mkt["is_open"]))
    raw_issues.extend(get_jp_risk_issues(con, params))

    # ---- Candidates ----
    candidates, filtered_candidates, _ = get_jp_trade_candidates(
        metrics, params, raw_issues, is_market_open=mkt["is_open"])

    system_data_red = [i for i in raw_issues if i["type"] in ["system", "data"] and i["severity"] == "red"]
    if system_data_red:
        blocker = system_data_red[0]
        st.error(f"**JP System Halted**: {blocker['title']} ({blocker['value']})")

    if metrics.empty:
        st.warning("**No JP Market Data**: Live metrics table is empty. JP engine might be warming up.")

    candidates_count = len(candidates)
    verdict_title, verdict_text_color, verdict_bg_color = calculate_jp_verdict(mkt, raw_issues, candidates_count)

    # Badges
    data_issues = [i for i in raw_issues if i["type"] == "data" and i["severity"] == "red"]
    data_status = "STALE" if data_issues else "FRESH"
    data_color = "red" if data_issues else "green"

    risk_items = [i for i in raw_issues if i["type"] == "risk" and i["severity"] in ["red", "amber"]]
    risk_status, risk_color = "OK", "green"
    if any(i["severity"] == "red" for i in risk_items):
        risk_status, risk_color = "BREACH", "red"
    elif risk_items:
        risk_status, risk_color = "WARNING", "orange"
    risk_label = f"{len(risk_items)} Risks" if risk_items else "Risk: OK"

    red_items = [i for i in raw_issues if i["severity"] == "red"]
    alerts_count = len(red_items)
    alerts_status = "None"
    alerts_color = "green"
    if alerts_count > 0:
        alerts_status = f"{alerts_count} Active"
        alerts_color = "red"

    # ---- Engine Health Strip ----
    st.markdown(f"""
        <div style="background-color:{verdict_bg_color}; color:{verdict_text_color};
                    padding:18px 24px; border-radius:8px; margin-bottom:12px; font-family:sans-serif;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <p style="font-size:14px; font-weight:600; text-transform:uppercase;
                          letter-spacing:1.5px; opacity:0.85; margin:0;">JP Engine Health</p>
                <p style="font-size:22px; font-weight:700; margin:0;">{verdict_title}</p>
            </div>
            <div style="display:flex; gap:12px; margin-top:12px;">
                <span style="background-color:rgba(255,255,255,0.12); padding:6px 14px;
                             border-radius:6px; font-size:13px; font-weight:500; display:flex;
                             align-items:center; gap:6px;">
                    <span style="width:8px; height:8px; border-radius:50%; display:inline-block;
                                 background-color:{mkt['color']};"></span> Market: {mkt['text']}
                </span>
                <span style="background-color:rgba(255,255,255,0.12); padding:6px 14px;
                             border-radius:6px; font-size:13px; font-weight:500; display:flex;
                             align-items:center; gap:6px;">
                    <span style="width:8px; height:8px; border-radius:50%; display:inline-block;
                                 background-color:{data_color};"></span> Data: {data_status}
                </span>
                <span style="background-color:rgba(255,255,255,0.12); padding:6px 14px;
                             border-radius:6px; font-size:13px; font-weight:500; display:flex;
                             align-items:center; gap:6px;">
                    <span style="width:8px; height:8px; border-radius:50%; display:inline-block;
                                 background-color:{risk_color};"></span> {risk_label}
                </span>
                <span style="background-color:rgba(255,255,255,0.12); padding:6px 14px;
                             border-radius:6px; font-size:13px; font-weight:500; display:flex;
                             align-items:center; gap:6px;">
                    <span style="width:8px; height:8px; border-radius:50%; display:inline-block;
                                 background-color:{alerts_color};"></span> Alerts: {alerts_status}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ---- Portfolio Summary + Live Market ----
    positions = db.jp_get_open_positions(con)
    pnl_history = db.jp_get_pnl_history(con, limit=1)
    pnl_summary = pnl_history.iloc[0] if not pnl_history.empty else None
    daily_df = db.jp_get_daily_performance(con, limit=30)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portfolio (JP)")
        if pnl_summary is not None:
            try:
                equity = float(pnl_summary.get("equity", 10_000_000))
                realized = float(pnl_summary.get("realized_pnl", 0))
                unrealized = float(pnl_summary.get("unrealized_pnl", 0))

                allocated = 0.0
                if not positions.empty:
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

                st.metric("Total Equity", f"¥{equity:,.0f}")
                st.metric("Available Equity", f"¥{available:,.0f}",
                          delta=f"-¥{allocated:,.0f} allocated" if allocated > 0 else None)

                portfolio_delta = 0.0
                if not positions.empty:
                    for _, pos in positions.iterrows():
                        try:
                            q1 = float(pos.get("qty1", 0) or 0)
                            q2 = float(pos.get("qty2", 0) or 0)
                            lp1 = float(pos.get("last_price1", 0) or 0)
                            lp2 = float(pos.get("last_price2", 0) or 0)
                            direction = pos.get("direction", "")
                            if direction == "LONG_SPREAD":
                                d = (lp1 * q1) - (lp2 * q2)
                            else:
                                d = (lp2 * q2) - (lp1 * q1)
                            portfolio_delta += d
                        except:
                            pass

                c1, c2, c3 = st.columns(3)
                c1.metric("Realized P&L", f"¥{realized:+,.0f}")
                c2.metric("Unrealized P&L", f"¥{unrealized:+,.0f}")
                c3.metric("Portfolio Delta", f"¥{portfolio_delta:+,.0f}")
            except Exception as e:
                st.error(f"Error parsing JP PnL: {e}")
        else:
            st.info("No JP PnL data available yet.")

    with col2:
        st.subheader("Live Market (JP Snapshot)")
        if not metrics.empty:
            m = metrics.copy()
            if "time" in m.columns:
                try:
                    m["time"] = pd.to_datetime(m["time"], utc=True)
                    m["time"] = m["time"].dt.tz_convert("Asia/Tokyo")
                except Exception:
                    pass
            m["z"] = pd.to_numeric(m["z"], errors="coerce")
            m["abs_z"] = m["z"].abs()
            m = m.sort_values("abs_z", ascending=False).drop(columns=["abs_z"])
            st.dataframe(m, use_container_width=True, hide_index=True)
        else:
            st.info("Waiting for JP live metrics...")

    st.divider()

    # ---- Daily Performance ----
    st.subheader("Daily Performance (JP)")
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.sort_values("date")

        d_chk1, d_chk2 = st.columns([2, 1])
        with d_chk1:
            fig_pnl = go.Figure()
            colors = [BB_GREEN if v >= 0 else BB_RED for v in daily_df["realized_pnl"]]

            equity_curve = db.jp_get_equity_curve(con, days=30)
            if not equity_curve.empty:
                equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"], utc=True)
                equity_curve = equity_curve.sort_values("timestamp")
                fig_pnl.add_trace(go.Scatter(
                    x=equity_curve["timestamp"], y=equity_curve["equity"],
                    mode="lines", line=dict(color=BB_AMBER, width=1.5),
                    name="Equity (Live)", yaxis="y2",
                    hovertemplate="Equity: ¥%{y:,.0f}<br>%{x}<extra></extra>",
                ))
            else:
                fig_pnl.add_trace(go.Scatter(
                    x=daily_df["date"], y=daily_df["total_equity"],
                    mode="lines+markers", line=dict(color=BB_AMBER, width=2),
                    marker=dict(size=4, color=BB_AMBER),
                    name="Equity", yaxis="y2",
                    hovertemplate="Equity: ¥%{y:,.0f}<extra></extra>",
                ))

            fig_pnl.add_trace(go.Bar(
                x=daily_df["date"], y=daily_df["realized_pnl"],
                marker_color=colors, marker_line=dict(color="#0b0f14", width=0.5),
                name="Daily P&L", opacity=0.9,
                hovertemplate="Date: %{x|%Y-%m-%d}<br>PnL: ¥%{y:,.0f}<extra></extra>",
            ))

            fig_pnl.update_layout(**bb_layout(
                title="DAILY P&L & EQUITY (JP)", xaxis_title="Date",
                yaxis=dict(title="Realized P&L (¥)", side="left", tickprefix="¥", tickformat=",.0f"),
                yaxis2=dict(title="Total Equity (¥)", overlaying="y", side="right", showgrid=False,
                            tickfont=dict(size=10, color=BB_AMBER),
                            title_font=dict(size=11, color=BB_AMBER),
                            tickprefix="¥", tickformat=",.0f"),
                legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2332", borderwidth=1,
                            font=dict(size=10, color="#8899aa"), orientation="h", y=1.1),
                height=350, bargap=0.3,
            ))
            st.plotly_chart(fig_pnl, use_container_width=True)

        with d_chk2:
            st.write("### Recent Days")
            disp_daily = daily_df.sort_values("date", ascending=False).copy()
            st.dataframe(
                disp_daily,
                column_order=["date", "realized_pnl", "num_trades", "total_equity"],
                column_config={
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "realized_pnl": st.column_config.NumberColumn("P&L", format="¥%.0f"),
                    "num_trades": st.column_config.NumberColumn("Trades"),
                    "total_equity": st.column_config.NumberColumn("Equity", format="¥%.0f"),
                },
                hide_index=True, use_container_width=True,
            )

        # Cumulative Stats
        total_pnl = daily_df["realized_pnl"].sum()
        total_trades = daily_df["num_trades"].sum()
        win_rate = 0.0
        if total_trades > 0:
            total_wins = daily_df["wins"].sum()
            win_rate = 100 * total_wins / total_trades

        avg_pnl_per_trade = 0.0
        avg_hold_str = "N/A"
        total_fees = 0.0
        try:
            closed_df = pd.read_sql_query("SELECT entry_time, exit_time, pnl, total_cost FROM jp_closed_trades", con)
            if not closed_df.empty:
                avg_pnl_per_trade = closed_df["pnl"].mean()
                closed_df["entry_time"] = pd.to_datetime(closed_df["entry_time"], format="mixed", utc=True)
                closed_df["exit_time"] = pd.to_datetime(closed_df["exit_time"], format="mixed", utc=True)
                valid = closed_df.dropna(subset=["entry_time", "exit_time"])
                if not valid.empty:
                    avg_hold_td = (valid["exit_time"] - valid["entry_time"]).mean()
                    total_hours = avg_hold_td.total_seconds() / 3600
                    avg_hold_str = f"{total_hours / 24:.1f}d" if total_hours >= 24 else f"{total_hours:.1f}h"
                total_fees = closed_df["total_cost"].fillna(0).sum()
        except Exception:
            pass

        sharpe_str = "N/A"
        try:
            import numpy as np
            eq_sorted = daily_df.sort_values("date")
            if "total_equity_mtm" in eq_sorted.columns:
                mtm_series = eq_sorted["total_equity_mtm"].dropna()
                if len(mtm_series) >= 3:
                    daily_returns = mtm_series.pct_change().dropna()
                    mean_ret = daily_returns.mean()
                    std_ret = daily_returns.std()
                    if std_ret > 0:
                        sharpe = mean_ret / std_ret * np.sqrt(252)
                        sharpe_str = f"{sharpe:.2f}"
        except Exception:
            pass

        col_a1, col_a2, col_a3 = st.columns(3)
        col_a1.metric("Total Trades", f"{total_trades}")
        col_a2.metric("Win Rate", f"{win_rate:.1f}%")
        col_a3.metric("Sharpe Ratio", sharpe_str)

        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Avg P&L / Trade", f"¥{avg_pnl_per_trade:+,.0f}")
        col_b2.metric("Avg Hold Time", avg_hold_str)
        col_b3.metric("Total Fees", f"¥{total_fees:,.0f}")
    else:
        st.info("No JP daily performance data recorded yet.")

    st.divider()

    # ---- Open Positions ----
    st.subheader("Open Positions (JP)")
    if not positions.empty:
        if "entry_time" in positions.columns:
            try:
                positions["entry_time_utc"] = pd.to_datetime(positions["entry_time"], format="mixed", utc=True)
                now_utc = datetime.now(timezone.utc)

                def calc_duration(entry_time_utc):
                    if pd.isnull(entry_time_utc):
                        return "N/A"
                    try:
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
                positions["entry_time"] = positions["entry_time_utc"].dt.tz_convert("Asia/Tokyo")
                positions["entry_time_str"] = positions["entry_time"].dt.strftime("%Y-%m-%d %H:%M JST")

                if not metrics.empty:
                    metrics_drift = metrics[["pair", "beta_drift_pct"]].copy()
                    positions = pd.merge(positions, metrics_drift, on="pair", how="left")
                else:
                    positions["beta_drift_pct"] = None

                def calc_equity(row):
                    try:
                        q1 = float(row.get("qty1", 0) or 0)
                        q2 = float(row.get("qty2", 0) or 0)
                        p1 = float(row.get("last_price1", 0) or 0)
                        p2 = float(row.get("last_price2", 0) or 0)
                        return abs(q1 * p1) + abs(q2 * p2)
                    except:
                        return 0.0

                def calc_delta(row):
                    try:
                        q1 = float(row.get("qty1", 0) or 0)
                        q2 = float(row.get("qty2", 0) or 0)
                        p1 = float(row.get("last_price1", 0) or 0)
                        p2 = float(row.get("last_price2", 0) or 0)
                        direction = row.get("direction", "")
                        if direction == "LONG_SPREAD":
                            return (p1 * q1) - (p2 * q2)
                        else:
                            return (p2 * q2) - (p1 * q1)
                    except:
                        return 0.0

                positions["equity"] = positions.apply(calc_equity, axis=1)
                positions["delta"] = positions.apply(calc_delta, axis=1)
                positions["beta_drift_display"] = positions["beta_drift_pct"].apply(
                    lambda x: f"{abs(x):.1f}%" if pd.notnull(x) else "N/A")
                if "beta_drift_limit" not in positions.columns:
                    positions["beta_drift_limit"] = 10.0
                positions["beta_drift_limit"] = positions["beta_drift_limit"].fillna(10.0).astype(float)

            except Exception as e:
                positions["entry_time_str"] = str(e)
                positions["duration_str"] = "Err"

        display_cols = ["pair", "duration_str", "beta_drift_display", "equity", "delta",
                        "pnl_unrealized", "entry_time_str", "beta_drift_limit"]
        valid_cols = [c for c in display_cols if c in positions.columns]

        edited_positions = st.data_editor(
            positions[valid_cols], use_container_width=True, hide_index=True,
            key="jp_open_positions_editor",
            column_config={
                "pair": st.column_config.Column("Pair", disabled=True),
                "duration_str": st.column_config.Column("Duration", disabled=True),
                "beta_drift_display": st.column_config.Column("Drift %", disabled=True),
                "equity": st.column_config.NumberColumn("Equity", format="¥%.0f", disabled=True),
                "delta": st.column_config.NumberColumn("Delta", format="¥%.0f", disabled=True),
                "pnl_unrealized": st.column_config.NumberColumn("Unrealized P&L", format="¥%.0f", disabled=True),
                "entry_time_str": st.column_config.TextColumn("Entry Time (JST)", disabled=True),
                "beta_drift_limit": st.column_config.NumberColumn("Drift Limit %", min_value=1.0, max_value=100.0,
                                                                   step=0.5, format="%.1f%%",
                                                                   help="Close position if drift exceeds this %"),
            },
        )

        if not positions.empty:
            merged_edits = pd.merge(positions[["pair", "beta_drift_limit"]],
                                    edited_positions[["pair", "beta_drift_limit"]],
                                    on="pair", suffixes=("_old", "_new"))
            updates = []
            for _, row in merged_edits.iterrows():
                if abs(row["beta_drift_limit_old"] - row["beta_drift_limit_new"]) > 0.01:
                    updates.append({"pair": row["pair"], "beta_drift_limit": row["beta_drift_limit_new"]})
            if updates:
                try:
                    db.jp_update_position_limits(con, updates)
                    st.toast(f"Updated JP limits for {len(updates)} position(s)!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error("Failed to save JP limits: " + str(e))
    else:
        st.caption("No open JP positions.")

    # ---- Portfolio Risk Controls ----
    st.markdown("#### JP Portfolio Risk Controls")
    if auth.is_admin():
        _rc = db.jp_get_risk_config(con)
        _rc_data = pd.DataFrame([
            {"Param": "Delta Warning %", "key": "delta_warning_pct",  "Value": _rc.get("delta_warning_pct", 0.015) * 100},
            {"Param": "Delta Soft %",    "key": "delta_soft_pct",     "Value": _rc.get("delta_soft_pct", 0.020) * 100},
            {"Param": "Delta Hard %",    "key": "delta_hard_pct",     "Value": _rc.get("delta_hard_pct", 0.030) * 100},
        ])
        _rc_orig = _rc_data["Value"].tolist()

        edited_rc = st.data_editor(
            _rc_data[["Param", "Value"]], use_container_width=True, hide_index=True,
            key="jp_risk_config_editor",
            column_config={
                "Param": st.column_config.TextColumn("Parameter", disabled=True),
                "Value": st.column_config.NumberColumn("Value (%)", min_value=0.1, max_value=100.0,
                                                       step=0.1, format="%.1f"),
            },
        )
        _rc_updates = {}
        for i, row in edited_rc.iterrows():
            new_val = row["Value"]
            if abs(new_val - _rc_orig[i]) > 0.01:
                _rc_updates[_rc_data.iloc[i]["key"]] = new_val / 100.0
        if _rc_updates:
            db.jp_save_risk_config(con, _rc_updates)
            st.toast(f"JP Risk config updated: {', '.join(_rc_updates.keys())}")
    else:
        st.caption("Login as admin to modify JP risk controls.")

    st.divider()

    # ---- Manual Actions ----
    st.markdown("### JP Manual Actions")
    if auth.is_admin():
        if not positions.empty:
            col1, col2 = st.columns([3, 1])
            with col1:
                pair_to_close = st.selectbox("Select JP Position to Close",
                                             options=positions["pair"].tolist(), key="jp_manual_close_select")
            with col2:
                st.write("")
                st.write("")
                if st.button("Close JP Position", type="primary", use_container_width=True, key="jp_close_btn"):
                    if pair_to_close:
                        try:
                            db.jp_add_manual_command(con, "CLOSE_POSITION", {"pair": pair_to_close})
                            st.success(f"Close request sent for {pair_to_close}")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to send close request: {e}")
        else:
            st.caption("No open JP positions to close.")
    else:
        st.caption("Login as admin to perform JP manual actions.")

    st.divider()

    # ---- Alerts & Risks ----
    panel_expanded = ("ISSUES DETECTED" in verdict_title) or (len(raw_issues) > 0)
    if raw_issues:
        with st.expander(f"Alerts & Risks ({len(raw_issues)})", expanded=panel_expanded):
            def priority_sort(i):
                score = 0
                if i["severity"] == "red": score += 100
                if i["type"] == "system": score += 30
                elif i["type"] == "data": score += 20
                elif i["type"] == "risk": score += 10
                return -score

            sorted_issues = sorted(raw_issues, key=priority_sort)
            if sorted_issues:
                df_issues = pd.DataFrame(sorted_issues)
                disp_cols = ["type", "pair", "title", "value", "threshold", "status"]
                disp_cols = [c for c in disp_cols if c in df_issues.columns]
                st.dataframe(df_issues[disp_cols], hide_index=True, use_container_width=True)

    # ---- Trade Candidates ----
    st.subheader("JP Trade Candidates (Sorted by Priority)")
    if filtered_candidates.empty:
        st.info("No JP pairs near entry thresholds right now.")
    else:
        display_cols = ["pair", "abs_z", "to_entry_display", "drift_check", "z_vs_eff", "block_reason", "status"]
        display_cols = [c for c in display_cols if c in filtered_candidates.columns]
        col_names = {
            "pair": "Pair", "abs_z": "|Z|", "to_entry_display": "To Entry",
            "drift_check": "Drift (Act/Lim)", "z_vs_eff": "|Z| vs Entry",
            "block_reason": "Block Reason", "status": "Status",
        }

        if not filtered_candidates.empty:
            renamed_df = filtered_candidates[display_cols].rename(columns=col_names)

            def style_risk_cells(row):
                styles = []
                for col_name, val in row.items():
                    val_str = str(val) if val is not None else ""
                    if "⛔" in val_str:
                        styles.append("background-color: #5a2525; color: #ff6b6b")
                    elif col_name == "Status":
                        if "Blocked" in val_str:
                            styles.append("background-color: #5a2525; color: #ff6b6b")
                        elif "Valid" in val_str:
                            styles.append("background-color: #1b5e20; color: #81c784")
                        elif "Watch" in val_str:
                            styles.append("background-color: #5a4525; color: #ffb74d")
                        else:
                            styles.append("")
                    elif col_name == "Block Reason" and val_str and val_str != "—":
                        styles.append("background-color: #5a2525; color: #ff6b6b; font-weight: bold")
                    else:
                        styles.append("")
                return styles

            styled_df = renamed_df.style.apply(style_risk_cells, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True,
                        column_config={"|Z|": st.column_config.NumberColumn("|Z|", format="%.2f")})

    st.divider()

    # ---- Configuration (Pair Parameters) ----
    st.subheader("JP Configuration (Pair Parameters)")
    if not auth.is_admin():
        if not params.empty:
            st.dataframe(params.sort_values("pair"), use_container_width=True, hide_index=True)
        else:
            st.warning("No JP parameters found.")
        st.caption("Login as admin to edit JP parameters.")
    else:
        if params.empty:
            st.warning("No JP parameters found. Pairs will be added once JP engine is configured.")
        else:
            p_edit = params.copy()
            if "enabled" in p_edit.columns:
                p_edit["enabled"] = p_edit["enabled"].astype(int).astype(bool)

            edited_df = st.data_editor(
                p_edit.sort_values("pair"), use_container_width=True, hide_index=True,
                num_rows="dynamic", key="jp_pair_params_editor",
                column_config={
                    "enabled": st.column_config.CheckboxColumn("Enabled"),
                    "max_drift_pct": st.column_config.NumberColumn("Beta Drift (%)"),
                    "max_drift_delta": st.column_config.NumberColumn("Delta Drift (¥)"),
                },
            )

            if st.button("Commit JP Config", key="jp_commit_params"):
                try:
                    to_save = edited_df.copy()
                    to_save["enabled"] = to_save["enabled"].astype(int)
                    db.jp_upsert_pair_params(con, to_save)
                    con.commit()
                    st.success("JP configuration saved! Reloading...")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save JP settings: {e}")

    st.divider()

    # ---- Pair Analysis ----
    st.subheader("JP Pair Analysis")
    all_pairs = []
    if not params.empty:
        all_pairs = sorted(params["pair"].unique())
    elif not metrics.empty:
        all_pairs = sorted(metrics["pair"].unique())

    pair_index = 0
    if "jp_selected_pair_override" in st.session_state:
        override = st.session_state["jp_selected_pair_override"]
        if override in all_pairs:
            pair_index = all_pairs.index(override)
        del st.session_state["jp_selected_pair_override"]

    selected_pair = st.selectbox("Select JP Pair", all_pairs, index=pair_index,
                                  key="jp_pair_selector") if all_pairs else None

    if selected_pair:
        history_raw = db.jp_get_pair_history(con, selected_pair, timeframe="15m", limit=3000)

        if not history_raw.empty:
            history_raw["time"] = pd.to_datetime(history_raw["time"], errors="coerce", utc=True)
            history_raw = history_raw.sort_values("time")

            cutoff_75d = datetime.now(timezone.utc) - timedelta(days=75)
            cutoff_30d = datetime.now(timezone.utc) - timedelta(days=30)
            history = history_raw[history_raw["time"] >= cutoff_75d].copy()

            if history.empty:
                st.warning(f"No JP history found for {selected_pair} in last 75 days.")
            else:
                history["time_jst"] = history["time"].dt.tz_convert("Asia/Tokyo")
                history["day_str"] = history["time_jst"].dt.strftime("%Y-%m-%d")
                day_changes = history[history["day_str"] != history["day_str"].shift(1)].iloc[1:]
                day_change_times = day_changes["time"].tolist()

                chk1, chk2 = st.columns([2, 1])

                with chk1:
                    st.markdown(f"### {selected_pair} Z-Score History")

                    z_entry = 2.0
                    z_exit = 0.5
                    if not params.empty:
                        param_row = params[params["pair"] == selected_pair]
                        if not param_row.empty:
                            z_entry = param_row.iloc[0]["z_entry"]
                            z_exit = param_row.iloc[0]["z_exit"]

                    history_old = history[history["time"] < cutoff_30d]
                    history_recent = history[history["time"] >= cutoff_30d]

                    z_vals = history["z"].dropna()
                    if not z_vals.empty:
                        y_min = min(z_vals.min(), -z_entry) - 0.5
                        y_max = max(z_vals.max(), z_entry) + 0.5
                    else:
                        y_min, y_max = -4, 4

                    fig = go.Figure()

                    fig.add_hrect(y0=-z_entry, y1=z_entry, fillcolor="rgba(90, 106, 122, 0.08)",
                                  layer="below", line_width=0,
                                  annotation_text="NO-TRADE ZONE", annotation_position="top left",
                                  annotation=dict(font_size=9, font_color=BB_GREY,
                                                  font=dict(family="IBM Plex Mono, monospace")))
                    fig.add_hrect(y0=z_entry, y1=y_max, fillcolor="rgba(0, 200, 83, 0.06)",
                                  layer="below", line_width=0)
                    fig.add_hrect(y0=y_min, y1=-z_entry, fillcolor="rgba(0, 200, 83, 0.06)",
                                  layer="below", line_width=0)

                    if not history_old.empty:
                        fig.add_trace(go.Scatter(
                            x=history_old["time"], y=history_old["z"],
                            mode="lines", line=dict(color=BB_DIM, width=1), opacity=0.5,
                            name="Prior 45d", hovertemplate="Z: %{y:.2f}<br>%{x}<extra></extra>",
                        ))
                    if not history_recent.empty:
                        fig.add_trace(go.Scatter(
                            x=history_recent["time"], y=history_recent["z"],
                            mode="lines", line=dict(color=BB_CYAN, width=1.8),
                            name="Recent 30d", hovertemplate="Z: %{y:.2f}<br>%{x}<extra></extra>",
                        ))

                    fig.add_hline(y=z_entry, line_dash="dash", line_color=BB_GREEN, line_width=1,
                                  annotation_text=f"+ENTRY ({z_entry})", annotation_position="right",
                                  annotation_font=dict(size=9, color=BB_GREEN, family="IBM Plex Mono, monospace"))
                    fig.add_hline(y=-z_entry, line_dash="dash", line_color=BB_GREEN, line_width=1,
                                  annotation_text=f"-ENTRY ({z_entry})", annotation_position="right",
                                  annotation_font=dict(size=9, color=BB_GREEN, family="IBM Plex Mono, monospace"))
                    fig.add_hline(y=z_exit, line_dash="dot", line_color=BB_AMBER, line_width=1,
                                  annotation_text=f"+EXIT ({z_exit})", annotation_position="right",
                                  annotation_font=dict(size=9, color=BB_AMBER, family="IBM Plex Mono, monospace"))
                    fig.add_hline(y=-z_exit, line_dash="dot", line_color=BB_AMBER, line_width=1,
                                  annotation_text=f"-EXIT ({z_exit})", annotation_position="right",
                                  annotation_font=dict(size=9, color=BB_AMBER, family="IBM Plex Mono, monospace"))

                    signals = detect_entry_signals(history, z_entry)
                    is_blocked, block_reason = get_pair_blocking_status(selected_pair, raw_issues)

                    if signals:
                        sig_times = [s["time"] for s in signals]
                        sig_zs = [s["z"] for s in signals]
                        sig_dirs = [s["direction"] for s in signals]
                        if is_blocked:
                            hover_texts = [f"Entry Signal (Blocked)<br>Reason: {block_reason}<br>Direction: {d}<br>Z: {z:.2f}"
                                           for d, z in zip(sig_dirs, sig_zs)]
                            fig.add_trace(go.Scatter(
                                x=sig_times, y=sig_zs, mode="markers",
                                marker=dict(size=10, color=BB_RED, symbol="x", line=dict(width=2, color="#cc0000")),
                                showlegend=False, hovertemplate="%{text}<extra></extra>", text=hover_texts,
                            ))
                        else:
                            hover_texts = [f"Entry Signal<br>Direction: {d}<br>Z: {z:.2f}"
                                           for d, z in zip(sig_dirs, sig_zs)]
                            fig.add_trace(go.Scatter(
                                x=sig_times, y=sig_zs, mode="markers",
                                marker=dict(size=8, color=BB_CYAN, symbol="circle", line=dict(width=1, color="#0b0f14")),
                                showlegend=False, hovertemplate="%{text}<extra></extra>", text=hover_texts,
                            ))

                    for dt in day_change_times:
                        fig.add_vline(x=dt, line_width=1, line_dash="dot", line_color="#1a2332", opacity=0.5)

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
                            x=1, y=1, xref="paper", yref="paper", xanchor="right", yanchor="top",
                            text=annotation_text, showarrow=False,
                            bgcolor="rgba(17, 24, 32, 0.92)",
                            font=dict(color=BB_LIGHT, size=10, family="IBM Plex Mono, monospace"),
                            bordercolor="#1a2332", borderwidth=1, borderpad=8, align="left",
                        )

                    fig.update_layout(**bb_layout(
                        title="Z-SCORE HISTORY (75 DAYS)", xaxis_title="", yaxis_title="Z-Score",
                        yaxis=dict(range=[y_min, y_max]), xaxis=dict(tickangle=-45),
                        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2332", borderwidth=1,
                                    font=dict(size=10, color="#8899aa"),
                                    orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                        margin=dict(t=35, b=50, l=50, r=100),
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    # Beta Chart
                    st.markdown(f"### {selected_pair} Beta Drift")
                    fig_beta = px.line(history, x="time", y=["direct_beta", "beta_30_weekly"],
                                       title="DIRECT vs 30D BETA", color_discrete_sequence=[BB_CYAN, BB_AMBER])
                    fig_beta.update_layout(**bb_layout())
                    fig_beta.update_xaxes(tickangle=-45)
                    for dt in day_change_times:
                        fig_beta.add_vline(x=dt, line_width=1, line_dash="dot", line_color="#1a2332", opacity=0.5)
                    st.plotly_chart(fig_beta, use_container_width=True)

                with chk2:
                    st.markdown("### Z-Score Distribution")
                    st.caption("10w base + 4w overlay")

                    current_z_val = history.iloc[-1]["z"] if not history.empty else None

                    col_bins, col_clip = st.columns(2)
                    with col_bins:
                        n_bins = st.select_slider("Bins", options=[20, 30, 40], value=30, key="jp_hist_bins")
                    with col_clip:
                        clip_z = st.checkbox("Clip to ±4", value=True, key="jp_hist_clip")

                    z_range = (-4, 4) if clip_z else None
                    fig_hist = plot_z_histogram(history, z_entry, n_bins, z_range, current_z_val)
                    st.plotly_chart(fig_hist, use_container_width=True)

                    if current_z_val is not None:
                        st.metric("Current Z", f"{current_z_val:.2f}")

                    st.markdown("### Recent Data")
                    st.dataframe(history.tail(10).sort_values("time", ascending=False),
                                 use_container_width=True, hide_index=True)
        else:
            st.warning(f"No JP history found for {selected_pair}.")

    # ---- Engine Details ----
    jp_hbs = [h for h in heartbeats if "JP" in h.get("engine_id", "").upper()]
    if jp_hbs:
        with st.expander("JP Engine Details", expanded=False):
            for hb in jp_hbs:
                engine_id = hb.get("engine_id", "Unknown")
                if engine_id.endswith("_events"):
                    continue
                status = hb.get("status", "Offline")
                ts = hb.get("timestamp", "")
                error = hb.get("error_message")

                st.markdown(f"**{engine_id}**")
                st.caption(f"Status: {status}")
                st.caption(f"Last: {ts}")
                if error:
                    st.error(f"{error}")
                if hb.get("meta_json"):
                    try:
                        meta = json.loads(hb["meta_json"])
                        st.json(meta, expanded=False)
                    except:
                        pass
