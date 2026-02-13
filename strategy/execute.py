# strategy/execute.py
from __future__ import annotations
from dataclasses import dataclass
from ib_insync import IB

from broker_prices import get_mid_price
from broker_orders import place_pair_market_order
from strategy.sizing import compute_pair_quantities

@dataclass
class OpenPosition:
    pair: str
    sym1: str
    sym2: str
    direction: str           # LONG_SPREAD / SHORT_SPREAD
    qty1: int
    qty2: int
    entry_price1: float
    entry_price2: float
    beta_entry: float
    entry_z: float
    entry_time: object       # pd.Timestamp ok

def paper_enter_trade(
    ib: IB,
    sym1: str,
    sym2: str,
    *,
    direction: str,
    beta_entry: float,
    entry_z: float,
    entry_time,
    capital_per_trade: float,
    place_orders: bool = False,   # SAFETY: default False
) -> OpenPosition:
    """
    Computes quantities from live mid prices. Optionally places paper orders.
    Returns an OpenPosition record you can store in DB.
    """

    p1 = get_mid_price(ib, sym1)
    p2 = get_mid_price(ib, sym2)

    qty1, qty2 = compute_pair_quantities(
        p1, p2, beta_entry,
        capital_per_trade=capital_per_trade
    )

    if place_orders:
        place_pair_market_order(ib, sym1, sym2, qty1, qty2, direction)

    return OpenPosition(
        pair=f"{sym1}-{sym2}",
        sym1=sym1,
        sym2=sym2,
        direction=direction,
        qty1=qty1,
        qty2=qty2,
        entry_price1=p1,
        entry_price2=p2,
        beta_entry=beta_entry,
        entry_z=entry_z,
        entry_time=entry_time,
    )
