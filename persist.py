from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd
import json

from db import (
    save_snapshot,
    save_history,
    save_open_positions,
    save_pnl_snapshot,
    # append_closed_trades, # TODO: re-implement or adapt
)


def persist_state(
    con, 
    metrics: pd.DataFrame, 
    portfolio, 
    new_closed: pd.DataFrame, 
    engine_id: str = "US",
    positions_df: pd.DataFrame = None,
    unrealized_pnl: float = 0.0
) -> None:
    """
    Persist all relevant state to the DB.
    """
    
    # 1. Snapshot (Live Metrics)
    if metrics is not None and not metrics.empty:
        save_snapshot(con, metrics)
        timestamp_col = "time" if "time" in metrics.columns else "last_updated"
        # Only save history if explicitly provided with time
        if "time" in metrics.columns:
            save_history(con, metrics, timeframe="15m") 

    # 2. Portfolio State (Positions)
    # Use explicitly provided DF (with PnL) if available, else fallback
    if positions_df is not None:
        save_open_positions(con, positions_df, engine_id=engine_id)
    else:
        open_pos = portfolio.open_positions_df()
        save_open_positions(con, open_pos, engine_id=engine_id)

    # 3. PnL Summary
    pnl = portfolio.pnl_df()
    if pnl is not None and not pnl.empty:
        last_row = pnl.iloc[-1]
        save_pnl_snapshot(
            con, 
            equity=float(last_row.get("equity", 0.0)),
            realized=float(last_row.get("realized_pnl", 0.0)),
            unrealized=float(unrealized_pnl), # Use the computed MTM PnL
            open_count=int(last_row.get("open_positions", 0)),
            closed_count=int(last_row.get("closed_trades", 0))
        )

    # 4. Closed Trades

    if new_closed is not None and not new_closed.empty:
        # We need to adapt to the new `closed_trades` table or `orders/fills`.
        # For now, let's just log them as closed trades roughly.
        # In the future, we should rely on orders/fills for history.
        with con:
            records = []
            for _, row in new_closed.iterrows():
                records.append((
                    str(row.get("pair")),
                    str(row.get("direction")),
                    str(row.get("entry_time")),
                    str(row.get("exit_time")),
                    float(row.get("pnl", 0.0)),
                    json.dumps(row.to_dict(), default=str)
                ))
            con.executemany("""
                INSERT INTO closed_trades (pair, direction, entry_time, exit_time, pnl, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, records)
