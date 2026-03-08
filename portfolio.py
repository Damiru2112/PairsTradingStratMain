# portfolio.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import pandas as pd

# ---------------------------------------------------------------------------
# Transaction cost constants (IBKR-style, US equities)
# ---------------------------------------------------------------------------
COMMISSION_PER_SHARE = 0.005        # IBKR fixed pricing
COMMISSION_MIN_PER_LEG = 1.00       # $1 minimum per leg
SEC_FEE_PER_DOLLAR_SOLD = 0.0       # $0 as of May 2025
TAF_PER_SHARE_SOLD = 0.000195       # FINRA TAF 2026 schedule
TAF_MAX_PER_TRADE = 9.79            # per-trade cap
DEFAULT_BORROW_RATE = 0.005         # 0.5% annual for liquid large-caps
SLIPPAGE_BPS = 1.0                  # 1 basis point per execution


def compute_commission(shares: int) -> float:
    """Broker commission for one leg."""
    return max(shares * COMMISSION_PER_SHARE, COMMISSION_MIN_PER_LEG)


def compute_reg_fees(shares_sold: int, notional_sold: float) -> float:
    """SEC + FINRA TAF on sell-side legs only."""
    taf = min(shares_sold * TAF_PER_SHARE_SOLD, TAF_MAX_PER_TRADE)
    sec = notional_sold * SEC_FEE_PER_DOLLAR_SOLD
    return taf + sec


def compute_borrow_cost(short_notional: float, days_held: float,
                        rate: float = DEFAULT_BORROW_RATE) -> float:
    """Short stock borrow interest."""
    return short_notional * rate * days_held / 365.0


def compute_slippage(notional: float) -> float:
    """Half-spread slippage estimate per execution."""
    return notional * SLIPPAGE_BPS / 10_000.0


def compute_round_trip_costs(
    qty1: int, qty2: int,
    entry_price1: float, entry_price2: float,
    exit_price1: float, exit_price2: float,
    direction: str, days_held: float,
    borrow_rate: float = DEFAULT_BORROW_RATE,
) -> dict:
    """
    Full round-trip cost breakdown for a closed pairs trade.
    Returns dict with: commission, reg_fees, borrow_cost, slippage, total_cost
    """
    # Commission: 4 legs (buy+sell each symbol)
    commission = (compute_commission(qty1) + compute_commission(qty2)) * 2

    # Regulatory fees: only on sell-side legs
    # LONG_SPREAD: entry sells qty2 (short sym2), exit sells qty1
    # SHORT_SPREAD: entry sells qty1 (short sym1), exit sells qty2
    if direction == "LONG_SPREAD":
        sell_shares = qty1 + qty2  # qty2 at entry + qty1 at exit
        sell_notional = qty2 * entry_price2 + qty1 * exit_price1
    else:  # SHORT_SPREAD
        sell_shares = qty1 + qty2  # qty1 at entry + qty2 at exit
        sell_notional = qty1 * entry_price1 + qty2 * exit_price2
    reg_fees = compute_reg_fees(sell_shares, sell_notional)

    # Borrow cost: on the short leg
    if direction == "LONG_SPREAD":
        short_notional = qty2 * entry_price2  # short sym2
    else:
        short_notional = qty1 * entry_price1  # short sym1
    borrow_cost = compute_borrow_cost(short_notional, days_held, borrow_rate)

    # Slippage: 4 legs
    slippage = (
        compute_slippage(qty1 * entry_price1) +
        compute_slippage(qty2 * entry_price2) +
        compute_slippage(qty1 * exit_price1) +
        compute_slippage(qty2 * exit_price2)
    )

    total_cost = commission + reg_fees + borrow_cost + slippage
    return {
        "commission": round(commission, 4),
        "reg_fees": round(reg_fees, 4),
        "borrow_cost": round(borrow_cost, 4),
        "slippage": round(slippage, 4),
        "total_cost": round(total_cost, 4),
    }


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
    pnl: float              # net P&L (after costs)
    commission: float = 0.0
    reg_fees: float = 0.0
    borrow_cost: float = 0.0
    slippage: float = 0.0
    total_cost: float = 0.0

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
                gross_pnl = (p.entry_price1 - p1) * p.qty1 + (p2 - p.entry_price2) * p.qty2
            else:  # LONG_SPREAD
                gross_pnl = (p1 - p.entry_price1) * p.qty1 + (p.entry_price2 - p2) * p.qty2

            # Estimate costs-to-close: exit commissions + exit slippage + exit reg fees + borrow accrued
            try:
                days_held = max((datetime.now(timezone.utc) - pd.Timestamp(p.entry_time).to_pydatetime().replace(
                    tzinfo=timezone.utc if p.entry_time.tzinfo is None else p.entry_time.tzinfo
                )).total_seconds() / 86400.0, 0.0)
            except Exception:
                days_held = 0.0

            # Exit commissions (2 legs)
            est_comm = compute_commission(p.qty1) + compute_commission(p.qty2)
            # Exit slippage (2 legs)
            est_slip = compute_slippage(p.qty1 * p1) + compute_slippage(p.qty2 * p2)
            # Exit reg fees (1 sell leg at exit)
            if p.direction == "LONG_SPREAD":
                est_reg = compute_reg_fees(p.qty1, p.qty1 * p1)  # sell sym1 at exit
            else:
                est_reg = compute_reg_fees(p.qty2, p.qty2 * p2)  # sell sym2 at exit
            # Borrow accrued so far
            if p.direction == "LONG_SPREAD":
                short_notional = p.qty2 * p.entry_price2
            else:
                short_notional = p.qty1 * p.entry_price1
            est_borrow = compute_borrow_cost(short_notional, days_held)

            est_cost = est_comm + est_slip + est_reg + est_borrow
            # Also add entry-side costs (already sunk but not yet accounted)
            entry_comm = compute_commission(p.qty1) + compute_commission(p.qty2)
            entry_slip = compute_slippage(p.qty1 * p.entry_price1) + compute_slippage(p.qty2 * p.entry_price2)
            if p.direction == "LONG_SPREAD":
                entry_reg = compute_reg_fees(p.qty2, p.qty2 * p.entry_price2)  # short sell sym2 at entry
            else:
                entry_reg = compute_reg_fees(p.qty1, p.qty1 * p.entry_price1)  # short sell sym1 at entry
            est_cost += entry_comm + entry_slip + entry_reg

            net_pnl = float(gross_pnl) - est_cost

            rows.append({
                "pair": pair,
                "direction": p.direction,
                "entry_time": p.entry_time,
                "entry_price1": p.entry_price1,
                "entry_price2": p.entry_price2,
                "last_price1": p1,
                "last_price2": p2,
                "unrealized_pnl": net_pnl,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "pair","direction","entry_time","entry_price1","entry_price2","last_price1","last_price2","unrealized_pnl"
        ])

    def enter(self, pos: Position) -> None:
        self.positions[pos.pair] = pos

    def exit(self, pair: str, exit_time: pd.Timestamp, exit_price1: float, exit_price2: float, exit_z: float) -> None:
        p = self.positions.pop(pair)

        if p.direction == "SHORT_SPREAD":
            gross_pnl = (p.entry_price1 - exit_price1) * p.qty1 + (exit_price2 - p.entry_price2) * p.qty2
        else:
            gross_pnl = (exit_price1 - p.entry_price1) * p.qty1 + (p.entry_price2 - exit_price2) * p.qty2

        # Compute holding duration in days
        try:
            days_held = max((exit_time - p.entry_time).total_seconds() / 86400.0, 0.0)
        except Exception:
            days_held = 0.0

        costs = compute_round_trip_costs(
            qty1=p.qty1, qty2=p.qty2,
            entry_price1=p.entry_price1, entry_price2=p.entry_price2,
            exit_price1=float(exit_price1), exit_price2=float(exit_price2),
            direction=p.direction, days_held=days_held,
        )

        net_pnl = float(gross_pnl) - costs["total_cost"]
        self.realized_pnl += net_pnl

        self.closed.append(ClosedTrade(
            pair=p.pair, sym1=p.sym1, sym2=p.sym2, direction=p.direction,
            qty1=p.qty1, qty2=p.qty2, beta_entry=p.beta_entry,
            entry_time=p.entry_time, exit_time=exit_time,
            entry_price1=p.entry_price1, entry_price2=p.entry_price2,
            exit_price1=float(exit_price1), exit_price2=float(exit_price2),
            entry_z=p.entry_z, exit_z=float(exit_z),
            pnl=net_pnl,
            commission=costs["commission"],
            reg_fees=costs["reg_fees"],
            borrow_cost=costs["borrow_cost"],
            slippage=costs["slippage"],
            total_cost=costs["total_cost"],
        ))

    def closed_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.closed]) if self.closed else pd.DataFrame()
