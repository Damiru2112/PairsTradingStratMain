# analytics/betas.py
from __future__ import annotations

import numpy as np
import pandas as pd

def compute_direct_beta(prices: pd.DataFrame, sym1: str, sym2: str) -> pd.Series:
    """Direct (instantaneous) beta = price1 / price2"""
    return (prices[sym1] / prices[sym2]).dropna()

def compute_beta_30d_weekly(
    prices: pd.DataFrame,
    sym1: str,
    sym2: str,
    *,
    lookback_days: int = 30,
    rebalance_days: int = 5,
    min_days_required: int = 35,
) -> pd.Series:
    """
    Rolling 30-day beta, recomputed ONLY at the end of each Friday,
    forward-filled for the following week.
    """
    ratio_15m = compute_direct_beta(prices, sym1, sym2)
    if ratio_15m.empty:
        return pd.Series(dtype=float)

    # 1. Estimate bars per day for lookback calculation
    bars_per_day = int(ratio_15m.groupby(ratio_15m.index.date).size().median())
    if bars_per_day <= 0:
        return pd.Series(dtype=float)

    approx_days = len(ratio_15m) / bars_per_day
    if approx_days < min_days_required:
        return pd.Series(dtype=float)

    lookback = lookback_days * bars_per_day

    # 2. Compute Rolling Mean (Candidate values at every bar)
    beta_hat = ratio_15m.rolling(lookback).mean()

    # 3. Create Friday-Only Mask
    # We want to select the value ONLY at the *last bar* of each Friday.
    
    # Identify Fridays (dayofweek==4)
    is_friday = ratio_15m.index.dayofweek == 4
    if not is_friday.any():
        # Fallback: if no Fridays exist in data (unlikely with 75 days), return empty or raw
        return pd.Series(dtype=float)

    # Find unique Friday dates
    friday_dates = np.unique(ratio_15m.index[is_friday].date)
    
    friday_mask = pd.Series(False, index=ratio_15m.index)
    
    # For each Friday, finding the last timestamp
    # Optimization: iterate only unique fridays (approx ~10-12 iterations for 75 days)
    for d in friday_dates:
        # Get subset of that day
        day_timestamps = ratio_15m.index[ratio_15m.index.date == d]
        if not day_timestamps.empty:
            last_ts = day_timestamps[-1]
            friday_mask.loc[last_ts] = True

    # 4. Sample & Forward Fill
    # values at non-Fridays become NaN -> ffill carries Friday value forward
    beta_30_weekly = beta_hat.where(friday_mask).ffill()
    
    # 5. Shift by 1 to avoid lookahead (next bar sees the frozen Friday value)
    return beta_30_weekly.shift(1)
