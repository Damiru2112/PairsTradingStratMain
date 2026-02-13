# slow_metrics_cache.py
"""
Cache for 15-minute computed slow metrics, refreshed per 15m bar.

Architecture:
- Slow layer computes 30-day rolling mean/std from 15m bars
- Fast layer (1m) uses cached mean/std with current 1m spread to compute z-score
- This preserves original 30d rolling methodology while enabling 1m execution

Z-Score Consistency Note:
    Fast z-score uses 1m spread with 15m-distribution mean/std.
    Expect small scale mismatch vs pure 15m z-score; intentional to preserve
    original calibration while gaining 1m execution speed.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict
import pandas as pd
import numpy as np
import logging

log = logging.getLogger("slow_metrics_cache")


@dataclass
class SlowMetrics:
    """Cached slow analytics for a single pair."""
    pair: str
    sym1: str
    sym2: str
    spread_mean_30d: float      # 30-day rolling mean of spread
    spread_std_30d: float       # 30-day rolling std of spread
    beta_30_weekly: float       # 30-day weekly-rebalanced beta
    direct_beta: float          # Most recent direct beta
    beta_drift_pct: float       # abs(direct - weekly) / weekly * 100
    last_bar_ts: pd.Timestamp   # Timestamp of the 15m bar these were computed from
    last_updated: pd.Timestamp  # When cache was updated


class SlowMetricsCache:
    """
    Cache for 15m-computed slow metrics.
    Refreshed when a new 15m bar is available.
    """
    
    def __init__(self):
        self._cache: Dict[str, SlowMetrics] = {}
    
    def update(
        self,
        pair: str,
        sym1: str,
        sym2: str,
        prices_15m: pd.DataFrame,
        *,
        lookback_days: int = 30,
        rebalance_days: int = 5,
    ) -> Optional[SlowMetrics]:
        """
        Compute and cache slow metrics from 15m price data.
        
        Args:
            pair: Pair name (e.g. "WEC-AEE")
            sym1, sym2: Symbol names
            prices_15m: DataFrame with columns [sym1, sym2], index = timestamp
            lookback_days: Lookback for rolling mean/std (default 30)
            rebalance_days: Rebalance frequency for weekly metrics (default 5)
        
        Returns:
            SlowMetrics if computed successfully, None otherwise
        """
        if prices_15m is None or prices_15m.empty:
            return None
        
        if sym1 not in prices_15m.columns or sym2 not in prices_15m.columns:
            return None
        
        x = prices_15m[[sym1, sym2]].dropna()
        if len(x) < 10:
            return None
        
        t_last = x.index[-1]
        
        # 1. Compute spread = price1 / price2 (ratio)
        spread = (x[sym1] / x[sym2]).dropna()
        if spread.empty:
            return None
        
        # 2. Estimate bars per day
        bars_per_day = int(spread.groupby(spread.index.date).size().median())
        if bars_per_day <= 0:
            bars_per_day = 26  # fallback for 15m bars
        
        lookback = lookback_days * bars_per_day
        
        # 3. Compute rolling mean and std
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        
        # Get latest values
        if pd.isna(spread_mean.iloc[-1]) or pd.isna(spread_std.iloc[-1]):
            log.warning(f"{pair}: Insufficient history for 30d mean/std")
            return None
        
        mean_30d = float(spread_mean.iloc[-1])
        std_30d = float(spread_std.iloc[-1])
        
        # 4. Compute betas (reuse existing logic pattern)
        from analytics.betas import compute_direct_beta, compute_beta_30d_weekly
        
        direct_beta_series = compute_direct_beta(x, sym1, sym2)
        beta_30_series = compute_beta_30d_weekly(x, sym1, sym2)
        
        if direct_beta_series.empty or t_last not in direct_beta_series.index:
            return None
        
        direct_beta = float(direct_beta_series.loc[t_last])
        
        beta_30 = None
        if not beta_30_series.empty and t_last in beta_30_series.index:
            val = beta_30_series.loc[t_last]
            if pd.notna(val):
                beta_30 = float(val)
        
        if beta_30 is None or beta_30 == 0:
            beta_30 = direct_beta  # fallback
        
        # 5. Compute drift
        drift_pct = abs(100.0 * (direct_beta - beta_30) / beta_30) if beta_30 != 0 else 0.0
        
        # 6. Create and cache
        metrics = SlowMetrics(
            pair=pair,
            sym1=sym1,
            sym2=sym2,
            spread_mean_30d=mean_30d,
            spread_std_30d=std_30d,
            beta_30_weekly=beta_30,
            direct_beta=direct_beta,
            beta_drift_pct=drift_pct,
            last_bar_ts=t_last,
            last_updated=pd.Timestamp.now(tz="UTC"),
        )
        
        self._cache[pair] = metrics
        log.debug(f"SlowMetrics cached for {pair}: mean={mean_30d:.4f}, std={std_30d:.4f}")
        
        return metrics
    
    def get(self, pair: str) -> Optional[SlowMetrics]:
        """Get cached slow metrics for a pair."""
        return self._cache.get(pair)
    
    def compute_fast_zscore(self, pair: str, current_spread: float) -> Optional[float]:
        """
        Compute z-score using cached 30d mean/std and current spread.
        
        Formula: z = (current_spread - cached_mean_30d) / cached_std_30d
        
        Args:
            pair: Pair name
            current_spread: Current spread value (price1 / price2) from 1m data
        
        Returns:
            Z-score or None if cache miss
        """
        metrics = self._cache.get(pair)
        if metrics is None:
            return None
        
        if metrics.spread_std_30d == 0 or pd.isna(metrics.spread_std_30d):
            return None
        
        z = (current_spread - metrics.spread_mean_30d) / metrics.spread_std_30d
        return float(z)
    
    def get_all_pairs(self) -> list[str]:
        """Get list of all cached pairs."""
        return list(self._cache.keys())
    
    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return len(self._cache) == 0
    
    def get_last_bar_ts(self, pair: str) -> Optional[pd.Timestamp]:
        """Get the last 15m bar timestamp for a pair."""
        metrics = self._cache.get(pair)
        return metrics.last_bar_ts if metrics else None
    
    def to_dict(self) -> Dict[str, dict]:
        """Export cache as dict for persistence."""
        return {pair: asdict(m) for pair, m in self._cache.items()}
