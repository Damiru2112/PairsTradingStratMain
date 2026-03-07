# tests/test_event_risk.py
"""
Acceptance tests for the Event Risk Layer.

Runs with: python -m pytest tests/test_event_risk.py -v
All tests use in-memory SQLite with synthetic event data
and mock current datetime for deterministic results.
"""
import sqlite3
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from event_risk import (
    compute_pair_event_state,
    resolve_event_trade_constraints,
    init_event_tables,
    trading_day_offset,
    current_trading_date,
    parse_pair_symbols,
    Z_EXIT_CAP,
    EARNINGS_MULTIPLIER,
    DIVIDEND_MULTIPLIER,
    MACRO_MULTIPLIER,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


@pytest.fixture
def con():
    """In-memory SQLite with event tables initialized."""
    c = sqlite3.connect(":memory:")
    init_event_tables(c)
    yield c
    c.close()


def _seed_earnings(con, ticker: str, report_date: str):
    con.execute(
        "INSERT OR REPLACE INTO events_earnings (ticker, report_date) VALUES (?, ?)",
        (ticker, report_date),
    )
    con.commit()


def _seed_dividend(con, ticker: str, ex_date: str, amount: float = 0.5):
    con.execute(
        "INSERT OR REPLACE INTO events_dividends (ticker, ex_date, amount) VALUES (?, ?, ?)",
        (ticker, ex_date, amount),
    )
    con.commit()


def _seed_macro(con, date_str: str, etype: str = "FOMC", label: str = "FOMC Decision"):
    con.execute(
        "INSERT OR REPLACE INTO events_macro (date, type, label) VALUES (?, ?, ?)",
        (date_str, etype, label),
    )
    con.commit()


# ============================================================
# Test 1: Earnings tomorrow → multiplier active, no block
# ============================================================
def test_earnings_tomorrow_multiplier_active(con):
    """If earnings are TOMORROW (day -1 in window), multiplier = 1.5, entry allowed."""
    # Use a known NYSE trading day: Thu 2026-02-12.
    # Earnings on Fri 2026-02-13 → "tomorrow" from Thu perspective
    _seed_earnings(con, "AAPL", "2026-02-13")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)  # Thursday 10 AM
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["multiplier"] == EARNINGS_MULTIPLIER
    assert state["base_entry_blocked"] is False
    assert len(state["active_events"]) >= 1
    assert any(e["type"] == "earnings" for e in state["active_events"])


# ============================================================
# Test 2: Earnings TODAY → entry blocked, exit allowed
# ============================================================
def test_earnings_today_entry_blocked(con):
    """Earnings day = today → entry_blocked = True, multiplier = 1.5."""
    _seed_earnings(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["multiplier"] == EARNINGS_MULTIPLIER
    assert state["base_entry_blocked"] is True


# ============================================================
# Test 3: Earnings 1 trading day ago → multiplier active
# ============================================================
def test_earnings_1day_ago_multiplier_active(con):
    """Event 1 trading day ago → still in [−1, +2] window → multiplier active."""
    # Earnings on Thu 2026-02-12, checking on Fri 2026-02-13
    _seed_earnings(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 13, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["multiplier"] == EARNINGS_MULTIPLIER
    assert state["base_entry_blocked"] is False  # Not day 0


# ============================================================
# Test 4: Earnings 3+ trading days ago → inactive
# ============================================================
def test_earnings_3days_ago_inactive(con):
    """Event 3+ trading days ago → outside [−1, +2] window → multiplier = 1.0."""
    # Earnings on Mon 2026-02-09 (a trading day).
    # Checking on Fri 2026-02-13 → offset = 4 trading days
    _seed_earnings(con, "AAPL", "2026-02-09")

    now_et = datetime(2026, 2, 13, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["multiplier"] == 1.0
    assert state["base_entry_blocked"] is False


# ============================================================
# Test 5: Dividend window [-1, +1]
# ============================================================
def test_dividend_window(con):
    """Dividend ex-date within [-1, +1] activates DIVIDEND_MULTIPLIER."""
    # Ex-date Fri 2026-02-13, check on Thu 2026-02-12 (day -1)
    _seed_dividend(con, "MSFT", "2026-02-13")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["multiplier"] == DIVIDEND_MULTIPLIER
    assert state["base_entry_blocked"] is False  # Day -1, not day 0


def test_dividend_today_not_blocked_by_default(con):
    """Dividend ex-date today → NOT blocked by default (config entry_blocked=0)."""
    _seed_dividend(con, "MSFT", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["multiplier"] == DIVIDEND_MULTIPLIER
    # Dividends default to entry_blocked=False in event_type_config
    assert state["base_entry_blocked"] is False


def test_dividend_today_blocked_when_config_enabled(con):
    """Dividend ex-date today → blocked if event_type_config entry_blocked=1."""
    _seed_dividend(con, "MSFT", "2026-02-12")
    # Override default config to block dividends
    from event_risk import set_event_type_config
    set_event_type_config(con, "dividend", DIVIDEND_MULTIPLIER, entry_blocked=True)

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["multiplier"] == DIVIDEND_MULTIPLIER
    assert state["base_entry_blocked"] is True


# ============================================================
# Test 6: Macro today blocks ALL pairs
# ============================================================
def test_macro_today_blocks_all(con):
    """Macro event today → blocks all pairs, multiplier = 1.2."""
    _seed_macro(con, "2026-02-12", "FOMC", "FOMC Decision")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)

    # Test with any arbitrary pair
    state1 = compute_pair_event_state("XOM-CVX", now_et, con)
    state2 = compute_pair_event_state("WMT-COST", now_et, con)

    for state in [state1, state2]:
        assert state["multiplier"] == MACRO_MULTIPLIER
        assert state["base_entry_blocked"] is True
        assert any(e["type"] == "macro" for e in state["active_events"])


# ============================================================
# Test 7: Overlapping events → max multiplier
# ============================================================
def test_combined_max_multiplier(con):
    """Overlapping earnings + macro → max(1.5, 1.2) = 1.5."""
    _seed_earnings(con, "AAPL", "2026-02-12")
    _seed_macro(con, "2026-02-12", "FOMC")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["multiplier"] == EARNINGS_MULTIPLIER  # max(1.5, 1.2) = 1.5
    assert state["base_entry_blocked"] is True
    assert len(state["active_events"]) >= 2


# ============================================================
# Test 8: No events → multiplier=1.0, entry allowed
# ============================================================
def test_no_events_multiplier_1(con):
    """No events nearby → multiplier=1.0, entry_allowed."""
    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["multiplier"] == 1.0
    assert state["base_entry_blocked"] is False
    assert state["active_events"] == []


# ============================================================
# Test 9: Exit allowed on event day (regression test)
# ============================================================
def test_exit_allowed_on_event_day(con):
    """
    Regression: even on event day (entry_blocked=True),
    the z_exit threshold should still be calculable and
    exits should proceed. We verify that the state returned
    does NOT prevent exit logic from running — entry_blocked
    only gates entries, not exits.
    """
    _seed_earnings(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["base_entry_blocked"] is True
    assert state["multiplier"] > 1.0

    # Simulate exit threshold calculation (as engine does)
    z_exit_base = 0.8
    z_exit_eff = min(z_exit_base * state["multiplier"], Z_EXIT_CAP)

    # Exit threshold should be widened but capped
    assert z_exit_eff == z_exit_base * EARNINGS_MULTIPLIER  # 0.8 * 1.5 = 1.2 < cap
    assert z_exit_eff <= Z_EXIT_CAP

    # With higher base, cap should kick in
    z_exit_base_high = 1.2
    z_exit_eff_high = min(z_exit_base_high * state["multiplier"], Z_EXIT_CAP)
    assert z_exit_eff_high == Z_EXIT_CAP  # 1.2 * 1.5 = 1.8 → capped at 1.5


# ============================================================
# Test 10: Timezone boundary (UTC midnight vs ET)
# ============================================================
def test_timezone_boundary(con):
    """
    At 00:30 UTC on 2026-02-13, it's still 2026-02-12 in US/Eastern (7:30 PM).
    An event on 2026-02-12 should be 'day 0', not 'already past'.
    """
    _seed_earnings(con, "AAPL", "2026-02-12")

    # 00:30 UTC on Feb 13 = 7:30 PM ET on Feb 12
    utc_time = datetime(2026, 2, 13, 0, 30, tzinfo=UTC)

    state = compute_pair_event_state("AAPL-MSFT", utc_time, con)

    # Should still be "event day" because ET date is Feb 12
    assert state["base_entry_blocked"] is True
    assert state["multiplier"] == EARNINGS_MULTIPLIER

    # Verify helper function directly
    et_date = current_trading_date(utc_time)
    assert et_date == date(2026, 2, 12)


# ============================================================
# Test 11: parse_pair_symbols
# ============================================================
def test_parse_pair_symbols():
    assert parse_pair_symbols("AAPL-MSFT") == ("AAPL", "MSFT")
    with pytest.raises(ValueError):
        parse_pair_symbols("INVALID")


# ============================================================
# Test 12: Layer 1 returns new fields
# ============================================================
def test_layer1_returns_base_blocking_reasons(con):
    """Layer 1 returns base_blocking_reasons for earnings day 0."""
    _seed_earnings(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    assert state["base_entry_blocked"] is True
    assert len(state["base_blocking_reasons"]) >= 1
    assert any("earnings" in r for r in state["base_blocking_reasons"])


def test_layer1_active_events_have_config_fields(con):
    """Active events include config_multiplier and config_entry_blocked."""
    _seed_earnings(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    for ev in state["active_events"]:
        assert "config_multiplier" in ev
        assert "config_entry_blocked" in ev


# ============================================================
# Test 13: Layer 2 — dividend short-leg blocking
# ============================================================
def test_dividend_short_leg_blocked(con):
    """Dividend on sym1, z>0 (short sym1) → dividend_short_leg_blocked."""
    _seed_dividend(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    # z > 0 → SHORT_SPREAD → sym1 (AAPL) is shorted
    constraints = resolve_event_trade_constraints("AAPL-MSFT", state, z_now=2.5)

    assert constraints["dividend_short_leg_blocked"] is True
    assert constraints["effective_entry_blocked"] is True
    assert constraints["intended_short_leg"] == "AAPL"
    assert any("short leg" in r for r in constraints["blocking_reasons"])


def test_dividend_long_leg_not_blocked(con):
    """Dividend on sym1, z<0 (short sym2) → no dividend block."""
    _seed_dividend(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    # z < 0 → LONG_SPREAD → sym2 (MSFT) is shorted, AAPL is long
    constraints = resolve_event_trade_constraints("AAPL-MSFT", state, z_now=-2.5)

    assert constraints["dividend_short_leg_blocked"] is False
    assert constraints["intended_short_leg"] == "MSFT"


def test_dividend_unknown_direction_warning(con):
    """Dividend active + z_now=None → warning, no block."""
    _seed_dividend(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    constraints = resolve_event_trade_constraints("AAPL-MSFT", state, z_now=None)

    assert constraints["dividend_short_leg_blocked"] is False
    assert constraints["intended_short_leg"] is None
    assert any("direction unknown" in w for w in constraints["warning_reasons"])


def test_dividend_z_zero_warning(con):
    """Dividend active + z_now=0 → warning, no block."""
    _seed_dividend(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    constraints = resolve_event_trade_constraints("AAPL-MSFT", state, z_now=0)

    assert constraints["dividend_short_leg_blocked"] is False
    assert constraints["intended_short_leg"] is None


# ============================================================
# Test 14: Layer 2 — hold warning for existing position
# ============================================================
def test_hold_warning_dividend_on_short_leg(con):
    """Existing SHORT_SPREAD position + dividend on sym1 → hold_warning."""
    _seed_dividend(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    constraints = resolve_event_trade_constraints(
        "AAPL-MSFT", state,
        z_now=2.5,
        current_position={"direction": "SHORT_SPREAD"},
    )

    assert constraints["hold_warning"] is True
    assert constraints["current_short_leg"] == "AAPL"
    assert any("open position" in w for w in constraints["warning_reasons"])


def test_no_hold_warning_dividend_on_long_leg(con):
    """Existing SHORT_SPREAD + dividend on sym2 (long leg) → no hold_warning."""
    _seed_dividend(con, "MSFT", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    constraints = resolve_event_trade_constraints(
        "AAPL-MSFT", state,
        z_now=2.5,
        current_position={"direction": "SHORT_SPREAD"},
    )

    assert constraints["hold_warning"] is False
    assert constraints["current_short_leg"] == "AAPL"


# ============================================================
# Test 15: Layer 2 — both legs have dividends
# ============================================================
def test_both_legs_dividend(con):
    """Both legs have dividends, z>0 → only sym1 (short) triggers block."""
    _seed_dividend(con, "AAPL", "2026-02-12")
    _seed_dividend(con, "MSFT", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    # z > 0 → short AAPL
    constraints = resolve_event_trade_constraints("AAPL-MSFT", state, z_now=2.5)

    assert constraints["dividend_short_leg_blocked"] is True
    assert constraints["intended_short_leg"] == "AAPL"


# ============================================================
# Test 16: Layer 2 seeds blocking_reasons from Layer 1
# ============================================================
def test_layer2_seeds_blocking_from_layer1(con):
    """Layer 2 carries forward earnings blocking reasons from Layer 1."""
    _seed_earnings(con, "AAPL", "2026-02-12")

    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)

    constraints = resolve_event_trade_constraints("AAPL-MSFT", state, z_now=2.5)

    assert constraints["effective_entry_blocked"] is True
    assert any("earnings" in r for r in constraints["blocking_reasons"])


# ============================================================
# Test 17: No events → clean constraints
# ============================================================
def test_no_events_clean_constraints(con):
    """No events → all clear."""
    now_et = datetime(2026, 2, 12, 10, 0, tzinfo=ET)
    state = compute_pair_event_state("AAPL-MSFT", now_et, con)
    constraints = resolve_event_trade_constraints("AAPL-MSFT", state, z_now=2.5)

    assert constraints["effective_multiplier"] == 1.0
    assert constraints["effective_entry_blocked"] is False
    assert constraints["dividend_short_leg_blocked"] is False
    assert constraints["hold_warning"] is False
    assert constraints["blocking_reasons"] == []
    assert constraints["warning_reasons"] == []
