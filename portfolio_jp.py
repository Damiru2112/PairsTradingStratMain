# portfolio_jp.py
# Japanese equities cost model (JPX/TSE)
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import pandas as pd

# ---------------------------------------------------------------------------
# Transaction cost constants (JPX/TSE, typical online broker like SBI/Rakuten)
# ---------------------------------------------------------------------------
# Commission: flat per-trade tiers (simplified to basis points)
COMMISSION_BPS = 5.0                # ~5 bps per trade (covers most online brokers)
COMMISSION_MIN_PER_LEG = 100.0      # ¥100 minimum per leg
# No SEC/FINRA fees in Japan
# Borrow cost for short selling
DEFAULT_BORROW_RATE = 0.01          # 1.0% annual (JP stocks typically higher borrow cost)
# Slippage
SLIPPAGE_BPS = 2.0                  # 2 bps per execution (JP spreads can be wider)


def compute_commission(notional: float) -> float:
    """Broker commission for one leg (JPY)."""
    return max(notional * COMMISSION_BPS / 10_000.0, COMMISSION_MIN_PER_LEG)


def compute_borrow_cost(short_notional: float, days_held: float,
                        rate: float = DEFAULT_BORROW_RATE) -> float:
    """Short stock borrow interest (JPY)."""
    return short_notional * rate * days_held / 365.0


def compute_slippage(notional: float) -> float:
    """Half-spread slippage estimate per execution (JPY)."""
    return notional * SLIPPAGE_BPS / 10_000.0


def compute_round_trip_costs(
    qty1: int, qty2: int,
    entry_price1: float, entry_price2: float,
    exit_price1: float, exit_price2: float,
    direction: str, days_held: float,
    borrow_rate: float = DEFAULT_BORROW_RATE,
) -> dict:
    """
    Full round-trip cost breakdown for a closed JP pairs trade.
    Returns dict with: commission, borrow_cost, slippage, total_cost
    (No reg_fees for JP market)
    """
    # Commission: 4 legs (buy+sell each symbol)
    entry_notional_1 = qty1 * entry_price1
    entry_notional_2 = qty2 * entry_price2
    exit_notional_1 = qty1 * exit_price1
    exit_notional_2 = qty2 * exit_price2

    commission = (
        compute_commission(entry_notional_1) +
        compute_commission(entry_notional_2) +
        compute_commission(exit_notional_1) +
        compute_commission(exit_notional_2)
    )

    # Borrow cost: on the short leg
    if direction == "LONG_SPREAD":
        short_notional = entry_notional_2  # short sym2
    else:
        short_notional = entry_notional_1  # short sym1
    borrow_cost = compute_borrow_cost(short_notional, days_held, borrow_rate)

    # Slippage: 4 legs
    slippage = (
        compute_slippage(entry_notional_1) +
        compute_slippage(entry_notional_2) +
        compute_slippage(exit_notional_1) +
        compute_slippage(exit_notional_2)
    )

    total_cost = commission + borrow_cost + slippage
    return {
        "commission": round(commission, 4),
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
    pnl: float              # net P&L after costs (JPY)
    commission: float = 0.0
    borrow_cost: float = 0.0
    slippage: float = 0.0
    total_cost: float = 0.0


class PaperPortfolio:
    def __init__(self, starting_equity: float = 10_000_000.0):
        """JP portfolio defaults to ¥10M starting equity."""
        self.starting_equity = float(starting_equity)
        self.realized_pnl = 0.0
        self.positions: dict[str, Position] = {}
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
        rows = []
        for pair, p in self.positions.items():
            p1_val = latest_prices.get(p.sym1)
            p2_val = latest_prices.get(p.sym2)

            if p1_val is None or p2_val is None:
                rows.append({
                    "pair": pair, "direction": p.direction, "entry_time": p.entry_time,
                    "entry_price1": p.entry_price1, "entry_price2": p.entry_price2,
                    "last_price1": p1_val, "last_price2": p2_val, "unrealized_pnl": 0.0,
                })
                continue

            p1 = float(p1_val)
            p2 = float(p2_val)

            if p.direction == "SHORT_SPREAD":
                gross_pnl = (p.entry_price1 - p1) * p.qty1 + (p2 - p.entry_price2) * p.qty2
            else:
                gross_pnl = (p1 - p.entry_price1) * p.qty1 + (p.entry_price2 - p2) * p.qty2

            try:
                days_held = max((datetime.now(timezone.utc) - pd.Timestamp(p.entry_time).to_pydatetime().replace(
                    tzinfo=timezone.utc if p.entry_time.tzinfo is None else p.entry_time.tzinfo
                )).total_seconds() / 86400.0, 0.0)
            except Exception:
                days_held = 0.0

            # Estimate costs-to-close
            est_comm = compute_commission(p.qty1 * p1) + compute_commission(p.qty2 * p2)
            est_slip = compute_slippage(p.qty1 * p1) + compute_slippage(p.qty2 * p2)
            if p.direction == "LONG_SPREAD":
                short_notional = p.qty2 * p.entry_price2
            else:
                short_notional = p.qty1 * p.entry_price1
            est_borrow = compute_borrow_cost(short_notional, days_held)

            # Entry-side costs
            entry_comm = compute_commission(p.qty1 * p.entry_price1) + compute_commission(p.qty2 * p.entry_price2)
            entry_slip = compute_slippage(p.qty1 * p.entry_price1) + compute_slippage(p.qty2 * p.entry_price2)

            est_cost = est_comm + est_slip + est_borrow + entry_comm + entry_slip
            net_pnl = float(gross_pnl) - est_cost

            rows.append({
                "pair": pair, "direction": p.direction, "entry_time": p.entry_time,
                "entry_price1": p.entry_price1, "entry_price2": p.entry_price2,
                "last_price1": p1, "last_price2": p2, "unrealized_pnl": net_pnl,
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
            borrow_cost=costs["borrow_cost"],
            slippage=costs["slippage"],
            total_cost=costs["total_cost"],
        ))

    def closed_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.closed]) if self.closed else pd.DataFrame()
