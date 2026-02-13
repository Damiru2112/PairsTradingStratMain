# analytics/zscore.py
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_zscore_30d_weekly(
    prices: pd.DataFrame,
    sym1: str,
    sym2: str,
    *,
    lookback_days: int = 30,
    rebalance_days: int = 5,
    min_days_required: int = 35,
) -> pd.Series:
    """
    Z-score of price ratio using 30-day rolling mean/std,
    recomputed weekly, forward-filled, shifted.
    """

    spread_15m = (prices[sym1] / prices[sym2]).dropna()
    if spread_15m.empty:
        return pd.Series(dtype=float)

    bars_per_day = int(spread_15m.groupby(spread_15m.index.date).size().median())
    if bars_per_day <= 0:
        return pd.Series(dtype=float)

    approx_days = len(spread_15m) / bars_per_day
    if approx_days < min_days_required:
        return pd.Series(dtype=float)

    lookback = lookback_days * bars_per_day
    rebalance = rebalance_days * bars_per_day

    spread_mean_hat = spread_15m.rolling(lookback).mean()
    spread_std_hat  = spread_15m.rolling(lookback).std()

    mask = np.zeros(len(spread_15m), dtype=bool)
    mask[::rebalance] = True

    mean_weekly = spread_mean_hat.where(mask).ffill().shift(1)
    std_weekly  = spread_std_hat.where(mask).ffill().shift(1)

    zscore = (spread_15m - mean_weekly) / std_weekly
    return zscore.dropna()
