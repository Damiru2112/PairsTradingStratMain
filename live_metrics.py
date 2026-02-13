# live_metrics.py
from __future__ import annotations
import pandas as pd

from analytics.betas import compute_direct_beta, compute_beta_30d_weekly
from analytics.zscore import compute_zscore_30d_weekly

BARS_PER_DAY_15M = 26  # ~6.5 trading hours / 15 min
DEFAULT_WINDOW = 10 * BARS_PER_DAY_15M  # ~10 trading days


def build_live_metrics_table(pair_frames: dict[tuple[str, str], pd.DataFrame],
                             window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """
    Returns one row per pair with columns:
    pair,time,z,direct_beta,beta_30_weekly,beta_drift_pct
    """
    rows = []

    for (a, b), df in pair_frames.items():
        if df is None or df.empty:
            continue
        if a not in df.columns or b not in df.columns:
            continue

        # Ensure we have enough data (though engine handles most filtering)
        x = df[[a, b]].dropna()
        if x.empty:
            continue

        t_last = x.index[-1]
        
        # 1. Compute Analytics using centralized logic
        # Note: These compute functions expect the full dataframe and return a Series matching the index
        direct_beta_series = compute_direct_beta(x, a, b)
        beta_30_series = compute_beta_30d_weekly(x, a, b)
        z_series = compute_zscore_30d_weekly(x, a, b)

        # 2. Get latest values
        if direct_beta_series.empty or t_last not in direct_beta_series.index:
            continue
            
        direct_last = float(direct_beta_series.loc[t_last])
        
        # Safe lookup for beta_30 (might be NaN if insufficient history)
        beta_last = None
        if not beta_30_series.empty and t_last in beta_30_series.index:
            val = beta_30_series.loc[t_last]
            if pd.notna(val):
                beta_last = float(val)
        
        # Safe lookup for z-score
        z_last = None
        if not z_series.empty and t_last in z_series.index:
            val = z_series.loc[t_last]
            if pd.notna(val):
                z_last = float(val)

        # 3. Compute Drift
        drift_pct = None
        if beta_last is not None and beta_last != 0:
            drift_pct = float(abs(100.0 * (direct_last - beta_last) / beta_last))

        rows.append({
            "pair": f"{a}-{b}",
            "time": pd.Timestamp(t_last).to_pydatetime(),
            "z": z_last,
            "direct_beta": direct_last,
            "beta_30_weekly": beta_last,
            "beta_drift_pct": drift_pct,
        })

    return pd.DataFrame(rows)

def build_pair_series_table(pair_frames: dict[tuple[str, str], pd.DataFrame],
                            window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """
    Returns many rows per pair across time:
    pair,time,z,direct_beta,beta_30_weekly,beta_drift_pct
    """
    out = []

    for (a, b), df in pair_frames.items():
        if df is None or df.empty or a not in df.columns or b not in df.columns:
            continue

        x = df[[a, b]].dropna()
        if x.empty:
            continue

        # 1. Compute Analytics
        direct_beta = compute_direct_beta(x, a, b)
        beta_30_weekly = compute_beta_30d_weekly(x, a, b)
        z_score = compute_zscore_30d_weekly(x, a, b)
        
        # 2. Compute Drift
        # Need to align indices carefully
        drift_pct = pd.Series(index=x.index, dtype=float)
        
        # We process where we have valid beta_30
        valid_mask = (beta_30_weekly != 0) & pd.notna(beta_30_weekly) & pd.notna(direct_beta)
        
        if valid_mask.any():
            b30 = beta_30_weekly[valid_mask]
            db = direct_beta[valid_mask]
            drift_pct[valid_mask] = (100.0 * (db - b30).abs() / b30).astype(float)

        df_out = pd.DataFrame({
            "pair": f"{a}-{b}",
            "time": x.index,
            "z": z_score.astype(float),
            "direct_beta": direct_beta.astype(float),
            "beta_30_weekly": beta_30_weekly.astype(float),
            "beta_drift_pct": drift_pct.astype(float),
        })
        
        # Filter to where we at least have some beta data to report, roughly matching previous logic
        # or maybe just drop completely empty rows
        df_out = df_out.dropna(subset=["direct_beta"])
        
        out.append(df_out)

    if not out:
        return pd.DataFrame(columns=["pair","time","z","direct_beta","beta_30_weekly","beta_drift_pct"])

    return pd.concat(out, ignore_index=True)
