# portfolio.py
from __future__ import annotations
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class Position:
    pair: str
    sym1: str
    sym2: str
    direction: str           # LONG_SPREAD / SHORT_SPREAD
    qty1: int
    qty2: int
    beta_entry: float
    entry_time: pd.Timestamp
    entry_price1: float
    entry_price2: float
    entry_z: float
    beta_drift_limit: float = 10.0

@dataclass
class ClosedTrade:
    pair: str
    sym1: str
    sym2: str
    direction: str
    qty1: int
    qty2: int
    beta_entry: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price1: float
    entry_price2: float
    exit_price1: float
    exit_price2: float
    entry_z: float
    exit_z: float
    pnl: float

class PaperPortfolio:
    def __init__(self, starting_equity: float = 100_000.0):
        self.starting_equity = float(starting_equity)
        self.realized_pnl = 0.0
        self.positions: dict[str, Position] = {}  # key=pair
        self.closed: list[ClosedTrade] = []

    def equity(self) -> float:
        return self.starting_equity + self.realized_pnl

    def open_positions_df(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame(columns=[
                "pair","direction","qty1","qty2","beta_entry","entry_time","entry_price1","entry_price2","entry_z"
            ])
        return pd.DataFrame([asdict(p) for p in self.positions.values()])

    def pnl_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "starting_equity": self.starting_equity,
            "realized_pnl": self.realized_pnl,
            "equity": self.equity(),
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed),
        }])

    def mark_to_market_unrealized(self, latest_prices: dict[str, float]) -> pd.DataFrame:
        """
        Compute unrealized PnL for each open position using latest CLOSE prices.
        latest_prices: {symbol: close}
        """
        rows = []
        for pair, p in self.positions.items():
            p1_val = latest_prices.get(p.sym1)
            p2_val = latest_prices.get(p.sym2)

            if p1_val is None or p2_val is None:
                # Prevent crash if price missing (e.g. illiquid after hours)
                # Persist 0.0 PnL and None prices rather than crashing engine
                rows.append({
                    "pair": pair,
                    "direction": p.direction,
                    "entry_time": p.entry_time,
                    "entry_price1": p.entry_price1,
                    "entry_price2": p.entry_price2,
                    "last_price1": p1_val, # None
                    "last_price2": p2_val, # None
                    "unrealized_pnl": 0.0,
                })
                continue

            p1 = float(p1_val)
            p2 = float(p2_val)
            b = p.beta_entry

            # PnL per your earlier definition (per 1 share of sym1 with beta hedge in sym2)
            # Scale by qty1 and qty2? We keep it consistent with your hedge:
            # qty2 should already reflect beta sizing; still we compute by legs:
            if p.direction == "SHORT_SPREAD":
                pnl = (p.entry_price1 - p1) * p.qty1 + (p2 - p.entry_price2) * p.qty2
            else:  # LONG_SPREAD
                pnl = (p1 - p.entry_price1) * p.qty1 + (p.entry_price2 - p2) * p.qty2

            rows.append({
                "pair": pair,
                "direction": p.direction,
                "entry_time": p.entry_time,
                "entry_price1": p.entry_price1,
                "entry_price2": p.entry_price2,
                "last_price1": p1,
                "last_price2": p2,
                "unrealized_pnl": float(pnl),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "pair","direction","entry_time","entry_price1","entry_price2","last_price1","last_price2","unrealized_pnl"
        ])

    def enter(self, pos: Position) -> None:
        self.positions[pos.pair] = pos

    def exit(self, pair: str, exit_time: pd.Timestamp, exit_price1: float, exit_price2: float, exit_z: float) -> None:
        p = self.positions.pop(pair)

        if p.direction == "SHORT_SPREAD":
            pnl = (p.entry_price1 - exit_price1) * p.qty1 + (exit_price2 - p.entry_price2) * p.qty2
        else:
            pnl = (exit_price1 - p.entry_price1) * p.qty1 + (p.entry_price2 - exit_price2) * p.qty2

        self.realized_pnl += float(pnl)

        self.closed.append(ClosedTrade(
            pair=p.pair, sym1=p.sym1, sym2=p.sym2, direction=p.direction,
            qty1=p.qty1, qty2=p.qty2, beta_entry=p.beta_entry,
            entry_time=p.entry_time, exit_time=exit_time,
            entry_price1=p.entry_price1, entry_price2=p.entry_price2,
            exit_price1=float(exit_price1), exit_price2=float(exit_price2),
            entry_z=p.entry_z, exit_z=float(exit_z),
            pnl=float(pnl)
        ))

    def closed_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.closed]) if self.closed else pd.DataFrame()
