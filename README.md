# Pairs Trading System
Production-Deployed Statistical Arbitrage Engine (US Equities)

Live Monitoring Dashboard: http://170.64.216.29:8501/

Check Screenshot folder for a quick view of the dashboard
---

## Overview

This project is a fully deployed statistical arbitrage system that trades mean reversion across 50+ US equity pairs.

The system runs continuously on a VPS, evaluates signals every minute during NYSE hours, simulates execution via a stateful portfolio engine, and exposes a real-time monitoring dashboard.

It is designed as a production trading system — not a notebook prototype.

---

## Strategy Design

### Spread Model

Spread is defined as a price ratio:

    spread = price_1 / price_2

Z-score:

    z = (spread - rolling_30d_mean) / rolling_30d_std

- 30 trading day lookback
- Weekly rebalance of slow statistics (Fridays only)
- Forward-filled intra-week
- Lookahead bias removed via shift(1)

---

### Entry & Exit

Entry:
- |z| ≥ 3.5 (configurable per pair)
- 2-of-3 consecutive 1-minute bars must exceed threshold (persistence filter)

Exit:
- |z| ≤ 1.0 (configurable)
- Exit threshold dynamically widened near events (capped at 1.5)

Re-entry:
- No cooldown

---

### Hedge Ratio

Two-layer hedge system:

1. Direct beta (instantaneous ratio, updated every bar)
2. 30-day rolling beta (weekly rebalance, forward-filled)

This reduces hedge noise while preserving statistical calibration.

---

### Position Sizing

Beta-adjusted dollar-neutral:

    capital_per_trade = equity * alloc_pct
    leg1 = capital_per_trade / (1 + beta)
    leg2 = leg1 * beta

Default:
- Starting equity: $100,000
- Allocation: 6.5% per trade

---

## Risk Management

Multi-layered protection:

- Beta drift entry filter (6.5%)
- Beta drift in-position stop (10%)
- Earnings multiplier (1.5x exit widening)
- Dividend multiplier (1.3x)
- Macro event multiplier (1.2x)
- Exit threshold cap (1.5)
- Market-hours gating
- One position per pair
- Persistence filter for false signal reduction
- Data staleness detection
- Engine heartbeat monitoring
- Telegram alert throttling

The system adjusts behavior around event risk rather than blindly disabling trading.

---

## Architecture

Dual-layer signal architecture:

Slow analytics layer (15m bars)
- Computes rolling mean, std, beta
- Rebalanced weekly
- Cached for reuse

Fast execution layer (1m bars)
- Evaluates signals every 60 seconds
- Applies risk filters
- Executes simulated trades
- Persists state to database

This preserves statistical stability while enabling responsive execution.

---

## Production Infrastructure

- Deployed on DigitalOcean VPS (Ubuntu)
- systemd-managed services
- SQLite (WAL mode) for concurrent engine/dashboard access
- Crash recovery via persistent state
- Event calendar integration (earnings/dividends/macro)
- Real-time Streamlit dashboard
- Modular separation: analytics → strategy → execution → persistence → monitoring

---

## Data

- Polygon.io (1m + 15m OHLCV)
- Earnings & dividends integration
- 75-day historical warmup
- Append-only CSV storage + SQLite persistence

---

## Scale & Performance

- 50+ concurrent pairs
- Medium-frequency (1-minute evaluation)
- Lightweight memory footprint
- Designed for robustness rather than HFT latency

---

## What Differentiates This System

- Weekly hedge rebalance to reduce intra-week beta noise
- Explicit lookahead bias prevention
- Event-aware dynamic exit thresholds
- Beta drift enforcement both at entry and during hold
- Persistence filter to reduce microstructure noise
- Real-time operational monitoring dashboard
- Full production deployment pipeline

This project bridges research logic and production trading infrastructure.

---

## Disclaimer

For educational and research purposes only.
