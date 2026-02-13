# strategy/signals.py
from __future__ import annotations
import pandas as pd

def generate_trades_from_z(
    prices: pd.DataFrame,
    sym1: str,
    sym2: str,
    z: pd.Series,
    direct_beta: pd.Series,
    beta_30: pd.Series,
    *,
    z_entry: float = 3.0,
    z_exit: float = 1.0,
    max_drift_pct: float = 6.5,
) -> pd.DataFrame:
    """
    Simple pairs-trading state machine using your existing logic:

    - Enter when |z| >= z_entry
      * if z > 0: SHORT_SPREAD (short sym1, long sym2)
      * if z < 0: LONG_SPREAD  (long sym1, short sym2)

    - Exit when |z| <= z_exit

    Drift filter at entry:
      drift_pct = abs(100 * (direct_beta - beta_entry) / beta_entry)
      skip entry if drift_pct >= max_drift_pct

    Returns:
      DataFrame of trades with entry/exit info + beta drift stats.
    """

    # Align everything to z's index (the tradable timestamps)
    idx = z.index
    p1 = prices[sym1].reindex(idx).astype(float)
    p2 = prices[sym2].reindex(idx).astype(float)
    db = direct_beta.reindex(idx).astype(float)
    b30 = beta_30.reindex(idx).astype(float)

    trades = []
    in_trade = False
    entry = {}

    skipped = 0

    for t in idx:
        zi = float(z.loc[t])

        # ENTRY
        if (not in_trade) and (abs(zi) >= z_entry):
            beta_entry = b30.loc[t]
            if pd.isna(beta_entry) or beta_entry == 0:
                continue

            direct_b = db.loc[t]
            if pd.isna(direct_b):
                continue

            drift_pct = abs(100.0 * (direct_b - beta_entry) / beta_entry)
            if drift_pct >= max_drift_pct:
                skipped += 1
                continue

            direction = "SHORT_SPREAD" if zi > 0 else "LONG_SPREAD"

            entry = dict(
                direction=direction,
                entry_time=t,
                entry_z=zi,
                entry_price_stock1=float(p1.loc[t]),
                entry_price_stock2=float(p2.loc[t]),
                beta_entry=float(beta_entry),
                entry_drift_pct=float(drift_pct),
            )
            in_trade = True
            continue

        # EXIT
        if in_trade and (abs(zi) <= z_exit):
            beta_exit = b30.loc[t]
            direct_b = db.loc[t]

            exit_price1 = float(p1.loc[t])
            exit_price2 = float(p2.loc[t])

            # PnL per 1 share of stock1 hedged by beta shares of stock2 (your ratio-style neutrality)
            # SHORT_SPREAD: short stock1, long beta*stock2
            # LONG_SPREAD:  long stock1, short beta*stock2
            b = entry["beta_entry"]

            if entry["direction"] == "SHORT_SPREAD":
                pnl = (entry["entry_price_stock1"] - exit_price1) + b * (exit_price2 - entry["entry_price_stock2"])
            else:
                pnl = (exit_price1 - entry["entry_price_stock1"]) + b * (entry["entry_price_stock2"] - exit_price2)

            beta_change_pct = None
            if (not pd.isna(beta_exit)) and entry["beta_entry"] != 0:
                beta_change_pct = 100.0 * (float(beta_exit) - entry["beta_entry"]) / entry["beta_entry"]

            trades.append({
                **entry,
                "exit_time": t,
                "exit_z": zi,
                "exit_price_stock1": exit_price1,
                "exit_price_stock2": exit_price2,
                "beta_exit": float(beta_exit) if not pd.isna(beta_exit) else None,
                "beta_change_pct": beta_change_pct,
                "pnl": float(pnl),
            })

            in_trade = False
            entry = {}
            continue

    df = pd.DataFrame(trades)

    # attach meta as attributes (handy for printing)
    df.attrs["entries_skipped_drift"] = skipped
    return df
