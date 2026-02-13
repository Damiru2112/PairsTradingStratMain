# strategy/sizing.py
from __future__ import annotations
import math

def compute_pair_quantities(
    price1: float,
    price2: float,
    beta: float,
    *,
    capital_per_trade: float,
    min_shares: int = 1,
) -> tuple[int, int]:
    """
    Dollar-neutral-ish sizing:
      - allocate half capital to leg1
      - allocate half capital * beta to leg2 (hedge scaled by beta)

    Returns (qty1, qty2) as positive integers (absolute quantities).
    Direction is handled elsewhere.
    """
    if price1 <= 0 or price2 <= 0 or beta <= 0:
        raise ValueError("Prices and beta must be positive.")

    # Beta-Normalized Sizing:
    # We want (qty1 * p1) + (qty2 * p2) approx capital_per_trade
    # And we want (qty2 * p2) = (qty1 * p1) * beta  (hedge relationship)
    #
    # Let L1 = leg1_notional, L2 = leg2_notional
    # L1 + L2 = capital_per_trade
    # L2 = L1 * beta
    # => L1 * (1 + beta) = capital_per_trade
    # => L1 = capital_per_trade / (1 + beta)
    
    leg1_notional = capital_per_trade / (1.0 + beta)
    leg2_notional = leg1_notional * beta

    qty1 = int(math.floor(leg1_notional / price1))
    qty2 = int(math.floor(leg2_notional / price2))

    qty1 = max(qty1, min_shares) if qty1 > 0 else 0
    qty2 = max(qty2, min_shares) if qty2 > 0 else 0

    if qty1 == 0 or qty2 == 0:
        raise RuntimeError(f"Computed zero quantity (qty1={qty1}, qty2={qty2}). Increase capital_per_trade.")

    return qty1, qty2
