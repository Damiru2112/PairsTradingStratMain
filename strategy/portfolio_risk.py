# strategy/portfolio_risk.py
"""
Portfolio-level risk overlay for trade admission.

Controls:
1. Delta neutrality — prevent directional drift
2. Sector crowding — prevent overconcentration

Only gates NEW entries. Exits are never blocked.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

from strategy.sizing import compute_pair_quantities

log = logging.getLogger("portfolio_risk")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

PORTFOLIO_RISK_CONFIG: Dict[str, Any] = {
    "enable_delta_control": True,
    "enable_sector_control": True,

    # Delta thresholds (as fraction of NAV)
    "delta_warning_pct": 0.015,   # 1.5%
    "delta_soft_pct": 0.020,      # 2.0%
    "delta_hard_pct": 0.030,      # 3.0%

    # Sector gross exposure thresholds (as fraction of NAV)
    "sector_warning_pct": 0.20,   # 20%
    "sector_soft_pct": 0.25,      # 25%
    "sector_hard_pct": 0.30,      # 30%

    # Size scaling
    "use_delta_size_scaling": True,
    "use_sector_size_scaling": True,

    # Z threshold tightening under sector crowding
    "sector_elevated_z_bump": 0.25,
    "sector_crowded_z_bump": 0.50,

    # Neutral delta tolerance — candidate delta below this is NEUTRAL
    "neutral_delta_tolerance_usd": 50.0,
}

# ──────────────────────────────────────────────
# SECTOR MAPPING
# ──────────────────────────────────────────────

PAIR_SECTOR: Dict[str, str] = {
    "A-RVTY": "Industrials",
    "AAT-GNL": "Real Estate (REIT)",
    "ABG-AN": "Consumer Discretionary",
    "ABT-BDX": "Healthcare",
    "ACM-BAH": "Industrials",
    "AGCO-ALSN": "Industrials",
    "AKR-CTO": "Real Estate (REIT)",
    "ALEX-FPI": "Real Estate (REIT)",
    "AME-HUBB": "Industrials",
    "AMH-ELS": "Real Estate (REIT)",
    "AMT-CCI": "Real Estate (REIT)",
    "APLE-CLDT": "Real Estate (REIT)",
    "ARI-BXMT": "Real Estate (REIT)",
    "AVB-EQR": "Real Estate (REIT)",
    "BBDC-BCSF": "Financials",
    "BBN-BOND": "Closed-End Funds",
    "BCH-BSAC": "Financials",
    "BOC-CBU": "Financials",
    "BST-NIE": "Closed-End Funds",
    "BSX-SYK": "Healthcare",
    "BV-EPAC": "Industrials",
    "CHMI-EFC": "Real Estate (REIT)",
    "CNI-CP": "Transportation",
    "CPT-IRT": "Real Estate (REIT)",
    "CTRE-LTC": "Real Estate (REIT)",
    "CUBE-NSA": "Real Estate (REIT)",
    "DSL-HYT": "Closed-End Funds",
    "ED-AWK": "Utilities",
    "ENS-THR": "Industrials",
    "EPD-ET": "Energy",
    "ETR-NI": "Utilities",
    "FAF-FNF": "Financials",
    "FCF-PFS": "Financials",
    "FFC-FPF": "Closed-End Funds",
    "FRA-JFR": "Closed-End Funds",
    "HMN-THG": "Financials",
    "HXL-TKR": "Industrials",
    "INVH-SUI": "Real Estate (REIT)",
    "IP-SW": "Materials",
    "ITW-PCAR": "Industrials",
    "JPM-BAC": "Financials",
    "KO-PEP": "Consumer Staples",
    "KW-PDM": "Real Estate (REIT)",
    "KYN-NML": "Closed-End Funds",
    "LYG-NOK": "Cross Sector",
    "MAA-UDR": "Real Estate (REIT)",
    "MAC-SKT": "Real Estate (REIT)",
    "MGM-MTCH": "Consumer Discretionary",
    "MTG-RDN": "Financials",
    "NAD-NEA": "Closed-End Funds",
    "PCN-PDI": "Closed-End Funds",
    "PDO-PFN": "Closed-End Funds",
    "QGEN-SNN": "Healthcare",
    "SII-TFPM": "Energy",
    "SO-D": "Utilities",
    "SPGI-ICE": "Financials",
    "TSN-HRL": "Consumer Staples",
    "TTC-VNT": "Industrials",
    "UBER-PANW": "Cross Sector",
    "VICI-SBAC": "Real Estate (REIT)",
    "WEC-AEE": "Utilities",
    "WMT-COST": "Consumer Staples",
    "XOM-CVX": "Energy",
}


# ──────────────────────────────────────────────
# PORTFOLIO STATE
# ──────────────────────────────────────────────

def compute_portfolio_state(
    portfolio,
    latest_prices: Dict[str, float],
    sector_map: Dict[str, str] = PAIR_SECTOR,
) -> Dict[str, Any]:
    """
    Compute current portfolio delta, gross exposure, and sector breakdowns.

    Returns a state dict used by evaluate_candidate_trade().
    """
    nav = portfolio.equity()
    # Include unrealized in NAV estimate
    unrealized = 0.0

    net_delta_usd = 0.0
    gross_exposure_usd = 0.0
    sector_gross: Dict[str, float] = {}
    sector_net: Dict[str, float] = {}

    for pair, pos in portfolio.positions.items():
        p1 = latest_prices.get(pos.sym1, pos.entry_price1)
        p2 = latest_prices.get(pos.sym2, pos.entry_price2)

        long_notional, short_notional = _position_notionals(pos, p1, p2)
        signed_delta = long_notional - short_notional
        gross = long_notional + short_notional

        net_delta_usd += signed_delta
        gross_exposure_usd += gross

        # Unrealized PnL approximation (gross, no costs)
        if pos.direction == "LONG_SPREAD":
            unrealized += (p1 - pos.entry_price1) * pos.qty1 + (pos.entry_price2 - p2) * pos.qty2
        else:
            unrealized += (pos.entry_price1 - p1) * pos.qty1 + (p2 - pos.entry_price2) * pos.qty2

        # Sector mapping
        sector = sector_map.get(pair)
        if sector is None:
            log.warning("[PORTFOLIO_RISK] Unmapped sector for pair=%s, assigning 'Unknown'", pair)
            sector = "Unknown"

        sector_gross[sector] = sector_gross.get(sector, 0.0) + gross
        sector_net[sector] = sector_net.get(sector, 0.0) + signed_delta

    total_nav = nav + unrealized
    if total_nav <= 0:
        total_nav = nav  # fallback

    return {
        "nav": total_nav,
        "net_delta_usd": net_delta_usd,
        "net_delta_pct": net_delta_usd / total_nav if total_nav else 0.0,
        "gross_exposure_usd": gross_exposure_usd,
        "gross_exposure_pct": gross_exposure_usd / total_nav if total_nav else 0.0,
        "sector_gross_usd": dict(sector_gross),
        "sector_gross_pct": {s: v / total_nav for s, v in sector_gross.items()} if total_nav else {},
        "sector_net_usd": dict(sector_net),
        "sector_net_pct": {s: v / total_nav for s, v in sector_net.items()} if total_nav else {},
    }


def _position_notionals(pos, p1: float, p2: float):
    """Return (long_notional, short_notional) for a position."""
    if pos.direction == "LONG_SPREAD":
        return (p1 * pos.qty1, p2 * pos.qty2)  # long sym1, short sym2
    else:  # SHORT_SPREAD
        return (p2 * pos.qty2, p1 * pos.qty1)  # long sym2, short sym1


# ──────────────────────────────────────────────
# PROJECTED STATE UPDATE
# ──────────────────────────────────────────────

def update_projected_state(
    state: Dict[str, Any],
    accepted: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update portfolio state after an accepted candidate entry.
    Returns a new state dict with updated delta / sector exposure.

    accepted must contain:
        candidate_delta_usd, candidate_gross_usd, sector
    """
    nav = state["nav"]
    new_net_delta = state["net_delta_usd"] + accepted["candidate_delta_usd"]
    new_gross = state["gross_exposure_usd"] + accepted["candidate_gross_usd"]

    sector = accepted["sector"]
    new_sector_gross = dict(state["sector_gross_usd"])
    new_sector_gross[sector] = new_sector_gross.get(sector, 0.0) + accepted["candidate_gross_usd"]

    new_sector_net = dict(state["sector_net_usd"])
    new_sector_net[sector] = new_sector_net.get(sector, 0.0) + accepted["candidate_delta_usd"]

    return {
        "nav": nav,
        "net_delta_usd": new_net_delta,
        "net_delta_pct": new_net_delta / nav if nav else 0.0,
        "gross_exposure_usd": new_gross,
        "gross_exposure_pct": new_gross / nav if nav else 0.0,
        "sector_gross_usd": new_sector_gross,
        "sector_gross_pct": {s: v / nav for s, v in new_sector_gross.items()} if nav else {},
        "sector_net_usd": new_sector_net,
        "sector_net_pct": {s: v / nav for s, v in new_sector_net.items()} if nav else {},
    }


# ──────────────────────────────────────────────
# SIZE MULTIPLIERS
# ──────────────────────────────────────────────

def delta_size_multiplier(net_delta_pct: float, delta_effect: str) -> float:
    """
    Scale trade size based on current portfolio delta drift.
    Delta-reducing trades get full size at all levels.
    """
    if delta_effect == "REDUCING":
        x = abs(net_delta_pct)
        if x < 0.020:
            return 1.00
        elif x < 0.025:
            return 0.50
        elif x < 0.030:
            return 0.25
        else:
            return 1.00  # reducing trades always allowed, even at hard breach

    # INCREASING or NEUTRAL-that-counts-as-increasing
    x = abs(net_delta_pct)
    if x < 0.015:
        return 1.00
    elif x < 0.020:
        return 0.75
    elif x < 0.025:
        return 0.00  # blocked
    elif x < 0.030:
        return 0.00  # blocked
    else:
        return 0.00  # blocked


def sector_size_multiplier(projected_sector_gross_pct: float) -> float:
    """Scale trade size based on projected sector gross exposure."""
    if projected_sector_gross_pct < 0.20:
        return 1.00
    elif projected_sector_gross_pct < 0.25:
        return 0.75
    elif projected_sector_gross_pct < 0.30:
        return 0.50
    else:
        return 0.00  # blocked


# ──────────────────────────────────────────────
# CANDIDATE EXPOSURE ESTIMATION
# ──────────────────────────────────────────────

def _estimate_candidate_exposure(
    candidate: Dict[str, Any],
    nav: float,
) -> Dict[str, float]:
    """
    Estimate candidate trade's delta and gross exposure using the exact
    sizing logic from strategy/sizing.py.

    candidate must contain:
        direction, price1, price2, beta, alloc_pct
    """
    p1 = candidate["price1"]
    p2 = candidate["price2"]
    beta = candidate["beta"]
    alloc_pct = candidate["alloc_pct"]
    direction = candidate["direction"]

    capital_per_trade = nav * alloc_pct

    try:
        qty1, qty2 = compute_pair_quantities(
            p1, p2, beta, capital_per_trade=capital_per_trade,
        )
    except (ValueError, RuntimeError):
        return {"long_notional": 0.0, "short_notional": 0.0,
                "candidate_delta_usd": 0.0, "candidate_gross_usd": 0.0}

    if direction == "LONG_SPREAD":
        long_notional = qty1 * p1
        short_notional = qty2 * p2
    else:  # SHORT_SPREAD
        long_notional = qty2 * p2
        short_notional = qty1 * p1

    return {
        "long_notional": long_notional,
        "short_notional": short_notional,
        "candidate_delta_usd": long_notional - short_notional,
        "candidate_gross_usd": long_notional + short_notional,
        "qty1": qty1,
        "qty2": qty2,
    }


# ──────────────────────────────────────────────
# MASTER TRADE ADMISSION
# ──────────────────────────────────────────────

def evaluate_candidate_trade(
    candidate: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    config: Dict[str, Any] = PORTFOLIO_RISK_CONFIG,
) -> Dict[str, Any]:
    """
    Evaluate whether a candidate entry should be admitted.

    candidate must contain:
        pair, direction, price1, price2, beta, alloc_pct

    Returns structured decision dict.
    """
    pair = candidate["pair"]
    nav = portfolio_state["nav"]

    # Sector lookup
    sector = PAIR_SECTOR.get(pair)
    if sector is None:
        log.warning("[PORTFOLIO_RISK] No sector mapping for pair=%s", pair)
        sector = "Unknown"

    # Estimate candidate exposure using exact sizing
    exposure = _estimate_candidate_exposure(candidate, nav)
    cand_delta = exposure["candidate_delta_usd"]
    cand_gross = exposure["candidate_gross_usd"]

    if cand_gross <= 0:
        return _reject(pair, sector, "signal_invalid", portfolio_state, exposure)

    # Current portfolio state
    cur_net_delta = portfolio_state["net_delta_usd"]
    cur_net_delta_pct = portfolio_state["net_delta_pct"]

    # Projected delta
    proj_net_delta = cur_net_delta + cand_delta
    proj_net_delta_pct = proj_net_delta / nav if nav else 0.0

    # Delta effect classification
    neutral_tol = config.get("neutral_delta_tolerance_usd", 50.0)
    cur_abs = abs(cur_net_delta)
    proj_abs = abs(proj_net_delta)

    if abs(cand_delta) < neutral_tol:
        delta_effect = "NEUTRAL"
    elif proj_abs < cur_abs:
        delta_effect = "REDUCING"
    else:
        delta_effect = "INCREASING"

    # Projected sector gross exposure
    cur_sector_gross = portfolio_state["sector_gross_usd"].get(sector, 0.0)
    proj_sector_gross = cur_sector_gross + cand_gross
    proj_sector_gross_pct = proj_sector_gross / nav if nav else 0.0

    # Sector state
    sector_hard = config.get("sector_hard_pct", 0.30)
    sector_soft = config.get("sector_soft_pct", 0.25)
    sector_warn = config.get("sector_warning_pct", 0.20)

    if proj_sector_gross_pct >= sector_hard:
        sector_state = "BLOCKED"
    elif proj_sector_gross_pct >= sector_soft:
        sector_state = "CROWDED"
    elif proj_sector_gross_pct >= sector_warn:
        sector_state = "ELEVATED"
    else:
        sector_state = "NORMAL"

    # Z-entry adjustment from sector crowding
    z_adj = 0.0
    if config.get("enable_sector_control", True):
        if sector_state == "ELEVATED":
            z_adj = config.get("sector_elevated_z_bump", 0.25)
        elif sector_state == "CROWDED":
            z_adj = config.get("sector_crowded_z_bump", 0.50)

    # Build base result
    result = {
        "allow": True,
        "final_size_multiplier": 1.0,
        "reject_reason": None,
        "delta_effect": delta_effect,
        "projected_net_delta_pct": proj_net_delta_pct,
        "projected_sector_gross_pct": proj_sector_gross_pct,
        "sector_state": sector_state,
        "sector": sector,
        "z_entry_adjustment": z_adj,
        "candidate_delta_usd": cand_delta,
        "candidate_gross_usd": cand_gross,
    }

    abs_delta_pct = abs(cur_net_delta_pct)
    delta_soft = config.get("delta_soft_pct", 0.020)
    delta_hard = config.get("delta_hard_pct", 0.030)

    # ── DELTA CONTROL ──
    if config.get("enable_delta_control", True):

        # Hard breach (≥3%)
        if abs_delta_pct >= delta_hard:
            if delta_effect == "INCREASING":
                return _reject(pair, sector, "delta_hard_block", portfolio_state, exposure,
                               result, delta_effect)
            if delta_effect == "NEUTRAL":
                return _reject(pair, sector, "delta_neutral_block", portfolio_state, exposure,
                               result, delta_effect)

        # Soft-to-hard (2.0–3.0%)
        elif abs_delta_pct >= delta_soft:
            if delta_effect == "INCREASING":
                return _reject(pair, sector, "delta_soft_block", portfolio_state, exposure,
                               result, delta_effect)
            # Neutral trades: allow only if below 2.5%
            if delta_effect == "NEUTRAL" and abs_delta_pct >= 0.025:
                return _reject(pair, sector, "delta_neutral_block", portfolio_state, exposure,
                               result, delta_effect)

    # ── SECTOR CONTROL ──
    if config.get("enable_sector_control", True):
        if sector_state == "BLOCKED":
            return _reject(pair, sector, "sector_hard_block", portfolio_state, exposure,
                           result, delta_effect)

    # ── SIZE SCALING ──
    size_mult = 1.0

    if config.get("use_delta_size_scaling", True) and config.get("enable_delta_control", True):
        size_mult *= delta_size_multiplier(cur_net_delta_pct, delta_effect)

    if config.get("use_sector_size_scaling", True) and config.get("enable_sector_control", True):
        size_mult *= sector_size_multiplier(proj_sector_gross_pct)

    if size_mult <= 0.0:
        return _reject(pair, sector, "size_scaled_to_zero", portfolio_state, exposure,
                       result, delta_effect)

    result["final_size_multiplier"] = size_mult

    _log_decision(pair, sector, delta_effect, cur_net_delta_pct, proj_net_delta_pct,
                  proj_sector_gross_pct, size_mult, z_adj, True, None)

    return result


def _reject(
    pair: str,
    sector: str,
    reason: str,
    state: Dict[str, Any],
    exposure: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
    delta_effect: str = "UNKNOWN",
) -> Dict[str, Any]:
    """Build a rejection result and log it."""
    proj_delta_pct = (state["net_delta_usd"] + exposure.get("candidate_delta_usd", 0.0)) / state["nav"] if state["nav"] else 0.0
    proj_sector_pct = 0.0

    r = result or {
        "allow": False,
        "final_size_multiplier": 0.0,
        "reject_reason": reason,
        "delta_effect": delta_effect,
        "projected_net_delta_pct": proj_delta_pct,
        "projected_sector_gross_pct": proj_sector_pct,
        "sector_state": "UNKNOWN",
        "sector": sector,
        "z_entry_adjustment": 0.0,
        "candidate_delta_usd": exposure.get("candidate_delta_usd", 0.0),
        "candidate_gross_usd": exposure.get("candidate_gross_usd", 0.0),
    }

    r["allow"] = False
    r["final_size_multiplier"] = 0.0
    r["reject_reason"] = reason

    _log_decision(
        pair, sector, delta_effect,
        state.get("net_delta_pct", 0.0),
        r.get("projected_net_delta_pct", 0.0),
        r.get("projected_sector_gross_pct", 0.0),
        0.0, r.get("z_entry_adjustment", 0.0),
        False, reason,
    )

    return r


def _log_decision(
    pair: str, sector: str, delta_effect: str,
    cur_delta_pct: float, proj_delta_pct: float,
    proj_sector_pct: float, size_mult: float,
    z_bump: float, allow: bool, reason: Optional[str],
):
    """Log every candidate evaluation for debugging."""
    status = "ALLOW" if allow else f"REJECT({reason})"
    log.info(
        "[PORTFOLIO_RISK] pair=%s sector=%s delta_effect=%s "
        "cur_delta_pct=%.4f proj_delta_pct=%.4f "
        "proj_sector_gross_pct=%.4f size_mult=%.2f z_bump=%.2f "
        "decision=%s",
        pair, sector, delta_effect,
        cur_delta_pct, proj_delta_pct,
        proj_sector_pct, size_mult, z_bump,
        status,
    )
