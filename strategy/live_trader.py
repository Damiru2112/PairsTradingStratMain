# strategy/live_trader.py
"""
Trading strategy for bar-close execution.

Provides:
- process_pair_barclose: Full strategy using 15m data (computes z/betas internally)
- process_pair_barclose_fast: Fast strategy using cached slow metrics + current prices
"""
from __future__ import annotations
import pandas as pd
from typing import Optional, Dict, Any

from analytics.betas import compute_direct_beta, compute_beta_30d_weekly
from analytics.zscore import compute_zscore_30d_weekly
from strategy.sizing import compute_pair_quantities
from portfolio import Position, PaperPortfolio


def _execute_exit(
    portfolio: PaperPortfolio,
    pair: str,
    sym1: str,
    sym2: str,
    p1: float,
    p2: float,
    z_now: float,
    bar_ts: pd.Timestamp,
    reason: str = "mean_reversion",
) -> Dict[str, Any]:
    """
    Shared exit execution logic.
    Returns trade action dict.
    """
    pos = portfolio.positions[pair]
    entry_time = pos.entry_time
    
    # Calculate holding duration
    try:
        holding_minutes = int((bar_ts - entry_time).total_seconds() / 60)
    except:
        holding_minutes = 0
    
    # Calculate PnL before exit
    if pos.direction == "SHORT_SPREAD":
        pnl = (pos.entry_price1 - p1) * pos.qty1 + (p2 - pos.entry_price2) * pos.qty2
    else:
        pnl = (p1 - pos.entry_price1) * pos.qty1 + (pos.entry_price2 - p2) * pos.qty2
    
    # Execute exit in portfolio
    portfolio.exit(pair, exit_time=bar_ts, exit_price1=p1, exit_price2=p2, exit_z=z_now)
    
    timestamp_iso = bar_ts.isoformat() if hasattr(bar_ts, 'isoformat') else str(bar_ts)
    event_id = f"{pair}:EXIT:{timestamp_iso}:{pos.direction}"
    
    return {
        "action": "EXIT",
        "event_id": event_id,
        "pair": pair,
        "sym1": sym1,
        "sym2": sym2,
        "price1": p1,
        "price2": p2,
        "z": z_now,
        "direction": pos.direction,
        "pnl": float(pnl),
        "entry_z": pos.entry_z,
        "timestamp": timestamp_iso,
        "holding_minutes": holding_minutes,
        "exit_reason": reason,
        "qty1": pos.qty1,
        "qty2": pos.qty2,
    }


def _execute_entry(
    portfolio: PaperPortfolio,
    pair: str,
    sym1: str,
    sym2: str,
    p1: float,
    p2: float,
    z_now: float,
    bar_ts: pd.Timestamp,
    beta_30: float,
    drift_pct: float,
    z_entry: float,
    alloc_pct: float,
) -> Dict[str, Any]:
    """
    Shared entry execution logic.
    Returns trade action dict.
    """
    direction = "SHORT_SPREAD" if z_now > 0 else "LONG_SPREAD"
    
    capital_per_trade = portfolio.equity() * float(alloc_pct)
    qty1, qty2 = compute_pair_quantities(
        p1, p2, float(beta_30),
        capital_per_trade=capital_per_trade
    )
    
    gross_usd = (qty1 * p1) + (qty2 * p2)
    
    # Execute entry in portfolio
    portfolio.enter(Position(
        pair=pair,
        sym1=sym1,
        sym2=sym2,
        direction=direction,
        qty1=qty1,
        qty2=qty2,
        beta_entry=float(beta_30),
        entry_time=bar_ts,
        entry_price1=p1,
        entry_price2=p2,
        entry_z=z_now
    ))
    
    timestamp_iso = bar_ts.isoformat() if hasattr(bar_ts, 'isoformat') else str(bar_ts)
    event_id = f"{pair}:ENTRY:{timestamp_iso}:{direction}"
    
    return {
        "action": "ENTRY",
        "event_id": event_id,
        "pair": pair,
        "sym1": sym1,
        "sym2": sym2,
        "price1": p1,
        "price2": p2,
        "z": z_now,
        "direction": direction,
        "beta_drift_pct": drift_pct,
        "qty1": qty1,
        "qty2": qty2,
        "timestamp": timestamp_iso,
        "z_entry_threshold": z_entry,
        "gross_usd": gross_usd,
    }


def process_pair_barclose(
    portfolio: PaperPortfolio,
    prices: pd.DataFrame,
    sym1: str,
    sym2: str,
    *,
    z_entry: float,
    z_exit: float,
    max_drift_pct: float,
    max_drift_delta: float,
    alloc_pct: float,
) -> Optional[Dict[str, Any]]:
    """
    Full bar-close trading logic using 15m data.
    Computes z-score and betas internally from price history.
    
    Returns:
        Trade action dict if entry/exit occurred, None otherwise.
    """
    pair = f"{sym1}-{sym2}"

    z = compute_zscore_30d_weekly(prices, sym1, sym2)
    if z.empty:
        return None

    direct_beta = compute_direct_beta(prices, sym1, sym2)
    beta_30 = compute_beta_30d_weekly(prices, sym1, sym2)

    t = z.index[-1]
    z_now = float(z.loc[t])

    # Latest closes
    p1 = float(prices[sym1].loc[t])
    p2 = float(prices[sym2].loc[t])

    # Pre-calc betas for use in exit/entry
    b30 = float(beta_30.reindex([t]).iloc[0])
    db = float(direct_beta.reindex([t]).iloc[0])
    
    drift_pct = 0.0
    if not (pd.isna(b30) or float(b30) == 0.0 or pd.isna(db)):
         drift_pct = abs(100.0 * (float(db) - b30) / b30)

    # --- EXIT ---
    if pair in portfolio.positions:
        pos = portfolio.positions[pair]
        drift_limit = getattr(pos, "beta_drift_limit", 10.0) # Fallback
        
        # 1. Beta Drift Breach
        if drift_pct > drift_limit:
             return _execute_exit(
                 portfolio, pair, sym1, sym2, p1, p2, z_now, t, 
                 reason=f"beta_drift_breach ({drift_pct:.1f}% > {drift_limit:.1f}%)"
             )

        # 2. Z-Score Exit
        if abs(z_now) <= z_exit:
            return _execute_exit(portfolio, pair, sym1, sym2, p1, p2, z_now, t, reason="mean_reversion")
        return None

    # --- ENTRY ---
    if abs(z_now) < z_entry:
        return None

    if pd.isna(b30) or float(b30) == 0.0 or pd.isna(db):
        return None

    if drift_pct >= max_drift_pct:
        return None

    # Check Delta Drift ($)
    # diff in shares of stock2 per 1 unit of stock1
    # value = diff * price2
    if max_drift_delta > 0:
        delta_drift = abs((float(db) - float(b30)) * p2)
        if delta_drift >= max_drift_delta:
            return None
    else:
        delta_drift = 0.0

    return _execute_entry(
        portfolio, pair, sym1, sym2, p1, p2, z_now, t,
        float(b30), drift_pct, z_entry, alloc_pct
    )


def process_pair_barclose_fast(
    portfolio: PaperPortfolio,
    sym1: str,
    sym2: str,
    price1: float,
    price2: float,
    z_now: float,
    bar_ts: pd.Timestamp,
    *,
    beta_30_weekly: float,
    beta_drift_pct: float,
    z_entry: float,
    z_exit: float,
    max_drift_pct: float,
    max_drift_delta: float, # NEW
    alloc_pct: float,
    entry_allowed: bool = True,
    # cached_direct_beta is needed for accurate delta calculation? 
    # Current fast mode relies on pre-calced drift pct from slow layer.
    # We can approximate or need to pass direct beta.
    # Actually, drifting pct is known, beta_30 is known.
    # drift_pct = (db - b30)/b30 
    # => db - b30 = drift_pct * b30 (signed? drift_pct is abs in signals.py? No, typically signed. Wait, signals.py uses abs)
    # The slow layer calculates drift_pct as ABS. This loses direction info.
    # Determining accurate dollar drift from just ABS drift % is impossible without sign.
    # However, if we assume worst case or just use the % to back-calc magnitude:
    # delta_drift = abs(db - b30) * p2 
    #             = (drift_pct/100 * b30) * p2
    # This works because drift_pct is abs(diff)/b30. So diff = drift_pct*b30. 
) -> Optional[Dict[str, Any]]:
    """
    Fast bar-close logic.
    """
    pair = f"{sym1}-{sym2}"

    # --- EXIT ---
    if pair in portfolio.positions:
        pos = portfolio.positions[pair]
        drift_limit = getattr(pos, "beta_drift_limit", 10.0) # Fallback
        
        # 1. Beta Drift Breach
        if beta_drift_pct > drift_limit:
             return _execute_exit(
                 portfolio, pair, sym1, sym2, price1, price2, z_now, bar_ts, 
                 reason=f"beta_drift_breach ({beta_drift_pct:.1f}% > {drift_limit:.1f}%)"
             )

        if abs(z_now) <= z_exit:
            return _execute_exit(portfolio, pair, sym1, sym2, price1, price2, z_now, bar_ts, reason="mean_reversion")
        return None

    # --- ENTRY ---
    if not entry_allowed:
        return None
    
    if abs(z_now) < z_entry:
        return None

    # Check Drift %
    if beta_drift_pct >= max_drift_pct:
        return None

    # Check Delta Drift ($)
    # We recover the magnitude of beta diff from the drift %
    # diff_magnitude = (beta_drift_pct / 100.0) * abs(beta_30_weekly)
    # delta_drift = diff_magnitude * price2
    if max_drift_delta > 0:
        diff_magnitude = (beta_drift_pct / 100.0) * abs(beta_30_weekly)
        delta_drift = diff_magnitude * price2
        if delta_drift >= max_drift_delta:
            return None

    if beta_30_weekly == 0.0 or pd.isna(beta_30_weekly):
        return None

    return _execute_entry(
        portfolio, pair, sym1, sym2, price1, price2, z_now, bar_ts,
        beta_30_weekly, beta_drift_pct, z_entry, alloc_pct
    )
