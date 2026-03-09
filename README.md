# Pairs Trading System
### Production-Deployed Statistical Arbitrage Engine (US Equities)

**Live Monitoring Dashboard:**  
http://uspairs.dr2112.com

📸 **Screenshots:**  
See the `/screenshots` folder for a quick view of the dashboard.

---

# Overview

This project is a **fully deployed statistical arbitrage system** that trades mean reversion across **50+ US equity pairs**.

The system:

- Runs continuously on a **VPS**
- Evaluates signals **every minute during NYSE hours**
- Simulates execution via a **stateful portfolio engine**
- Exposes a **real-time monitoring dashboard**

It is designed as a **production trading system**

---

# System Architecture

```
            Polygon.io
                 │
                 ▼
        Historical Data Loader
                 │
                 ▼
      ┌─────────────────────┐
      │   Analytics Layer   │
      │ (15m statistical)   │
      └─────────────────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │   Strategy Engine   │
      │  (1m signal logic)  │
      └─────────────────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │ Portfolio Simulator │
      │  Position Manager   │
      └─────────────────────┘
                 │
                 ▼
        SQLite Trading DB
          │            │
          ▼            ▼
   Streamlit Dashboard  Telegram Alerts
```

---

# Strategy Design

## Spread Model

Spread is defined as a **price ratio**:

```python
spread = price_1 / price_2
```

Z-score:

```python
z = (spread - rolling_30d_mean) / rolling_30d_std
```

Features:

- **30 trading day lookback**
- **Weekly rebalance of slow statistics (Fridays only)**
- Statistics **forward-filled intra-week**
- **Lookahead bias removed via `shift(1)`**

---

## Entry & Exit Rules

### Entry

```
|z| ≥ 3.5
```

Additional signal filter:

- **2-of-3 consecutive 1-minute bars** must exceed the threshold
- Reduces microstructure noise and false signals

### Exit

```
|z| ≤ 1.0
```

Dynamic widening near events:

- Earnings multiplier
- Dividend multiplier
- Macro event multiplier

Maximum exit threshold cap:

```
|z| ≤ 1.5
```

### Re-entry

- **No cooldown**
- New signals allowed immediately after exit

---

# Hedge Ratio

Two-layer hedge system:

### 1️⃣ Direct Beta
- Instantaneous hedge ratio
- Updated **every bar**

### 2️⃣ Rolling Beta
- **30-day rolling regression**
- Rebalanced **weekly**
- Forward-filled intra-week

Purpose:

- Reduce hedge noise
- Preserve statistical calibration

---

# Position Sizing

Beta-adjusted **dollar-neutral sizing**:

```python
capital_per_trade = equity * alloc_pct

leg1 = capital_per_trade / (1 + beta)
leg2 = leg1 * beta
```

Default parameters:

| Parameter | Value |
|---|---|
| Starting Equity | $100,000 |
| Allocation per trade | 6.5% |

---

# Transaction Cost Model

A **realistic four-component cost model** is applied to every trade.

## 1. Broker Commission (IBKR Fixed)

```
$0.005 per share
$1.00 minimum per leg
```

Applied to **all four executions** in a round-trip pairs trade.

---

## 2. Regulatory Fees (Sell-Side Only)

FINRA TAF:

```
$0.000195 per share sold
Cap: $9.79 per trade
```

SEC Section 31:

```
$0.00 per dollar sold
(Current rate as of May 2025)
```

Applied to **the two sell executions**.

---

## 3. Short Borrow Cost

Default:

```
0.5% annual
```

Accrued daily on short-leg notional.

Calculation:

```python
borrow_cost = short_notional * rate * days_held / 365
```

---

## 4. Slippage

Half-spread estimate:

```
1 basis point per execution
```

```python
slippage = notional * 1 / 10000
```

---

## Cost Impact

- All costs deducted **at trade close**
- Unrealized P&L reflects **estimated round-trip costs**

Trade records store:

- Commission
- Regulatory fees
- Borrow cost
- Slippage

for full **cost attribution**.

---

# Risk Management

The system implements **multi-layered risk protection**:

- Beta drift entry filter (**6.5%**)
- Beta drift in-position stop (**10%**)
- Earnings event multiplier (**1.5×**)
- Dividend multiplier (**1.3×**)
- Macro event multiplier (**1.2×**)
- Exit threshold cap (**1.5**)
- Market-hours gating
- One position per pair
- Persistence filter
- Data staleness detection
- Engine heartbeat monitoring
- Telegram alert throttling

Rather than disabling trading around events, the system **adapts risk dynamically**.

---

# Performance Tracking

## Equity Curve

Equity snapshots:

- **Every 15 minutes intraday**
- **End-of-day mark-to-market equity**

Includes:

- Realized P&L
- Unrealized P&L

---

## Sharpe Ratio

Computed from **daily MTM equity returns**:

```python
Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
```

---

## Dashboard Metrics

The monitoring dashboard displays:

- Total Trades
- Win Rate
- Sharpe Ratio
- Average P&L per Trade
- Average Hold Time
- Total Fees Paid

---

# Telegram Notifications

Real-time alerts include:

### Entry Alerts

- Pair
- Z-score
- Position sizing

### Exit Alerts

- Net P&L
- Fee breakdown
- Holding period

### End-of-Day Report

- Realized P&L
- Total fees paid
- Equity
- Win/loss count

---

# Architecture

The engine uses a **dual-layer signal architecture**.

## Slow Analytics Layer (15m Bars)

Responsible for:

- Rolling mean
- Rolling standard deviation
- Hedge ratio estimation

Characteristics:

- Weekly rebalance
- Cached computations

---

## Fast Execution Layer (1m Bars)

Responsible for:

- Signal evaluation
- Risk filter application
- Trade simulation
- Portfolio updates

Signals evaluated **every 60 seconds**.

---

# Production Infrastructure

Deployment stack:

| Component | Description |
|---|---|
| VPS | DigitalOcean (Ubuntu) |
| Services | systemd-managed processes |
| Database | SQLite (WAL mode) |
| Dashboard | Streamlit terminal UI |
| Alerts | Telegram bot |
| Deployment | GitHub → VPS auto-pull |

Features:

- Crash recovery via persistent state
- Concurrent engine/dashboard database access
- Event calendar integration
- Modular architecture

```
analytics → strategy → execution → persistence → monitoring
```

---

# Data Sources

Market Data:

- **Polygon.io**
- 1-minute OHLCV
- 15-minute OHLCV

Additional datasets:

- Earnings calendar
- Dividend events

Historical warmup:

```
75 trading days
```

Storage:

- Append-only CSV history
- SQLite trading database

---

# Scale & Performance

The engine currently supports:

- **50+ concurrent pairs**
- **1-minute signal evaluation**
- Medium-frequency stat arb

Design focus:

- **Robustness**
- **Operational reliability**
- **Research reproducibility**

(Not ultra-low-latency HFT)

---

# What Differentiates This System

Key design features:

- Weekly hedge rebalance to reduce beta noise
- Explicit **lookahead bias prevention**
- Event-aware **dynamic exit thresholds**
- Beta drift enforcement during entry and holding
- Persistence filter to reduce microstructure noise
- Full **four-component transaction cost model**
- Mark-to-market equity with proper Sharpe calculation
- Real-time monitoring dashboard
- Production deployment infrastructure

This project bridges **research logic and production trading systems.**

---

# Disclaimer

This project is provided **for educational and research purposes only**.

It does **not constitute investment advice** and should not be used for live trading without proper risk controls and regulatory compliance.
