# Leveraged Certificates Trading System

A systematic approach to trading leveraged certificates (leva fissa) on European indices and stocks. Built as a learning project following a structured 7-module curriculum.

## Overview

Leveraged certificates (e.g., 5x or 7x on FTSE MIB) amplify **daily** returns of an underlying index. Due to daily rebalancing, they suffer from **volatility decay** (compounding effect), making them unsuitable for buy-and-hold but potentially profitable for short-term directional trades in strong trend regimes.

This project provides:
- A Monte Carlo simulator to understand volatility decay mathematically
- A real-time regime scanner to identify favorable trading windows
- A complete backtesting engine with walk-forward validation and real costs
- A multi-asset/multi-leverage optimizer to find the best combinations
- A stock screener to evaluate single stocks as potential candidates

## Key Findings

| # | Asset | Leverage | Sharpe (OOS) | Profit Factor | Return | Max Drawdown | Trades |
|---|-------|----------|-------------|---------------|--------|-------------|--------|
| 1 | DAX | 3x | 2.21 | 2.33 | +3.65% | -1.07% | 11 |
| 2 | DAX | 7x | 2.12 | 2.25 | +3.53% | -1.17% | 11 |
| 3 | DAX | 5x | 2.10 | 2.24 | +3.50% | -1.17% | 11 |
| 4 | FTSE MIB | 3x | 1.93 | 1.95 | +3.43% | -1.29% | 19 |
| 5 | FTSE MIB | 5x | 1.78 | 1.87 | +3.15% | -1.29% | 19 |
| 6 | CAC 40 | 7x | 1.72 | 2.22 | +3.18% | -1.22% | 16 |
| 7 | FTSE MIB | 7x | 1.47 | 1.70 | +2.55% | -1.29% | 19 |
| 8 | CAC 40 | 5x | 1.45 | 2.12 | +2.98% | -1.26% | 16 |
| 9 | CAC 40 | 3x | 0.80 | 1.77 | +2.07% | -1.27% | 16 |
| 10 | EuroStoxx50 | 7x | -1.28 | 1.03 | +0.20% | -3.40% | 19 |

- **Best combinations**: DAX 3x (highest Sharpe) and FTSE MIB 3x (most trades, best statistical significance)
- **Lower leverage (3x) tends to outperform** higher leverage on risk-adjusted basis across most assets
- **CAC 40** shows moderate edge at higher leverage levels (5x-7x)
- **EuroStoxx 50 and S&P 500** do not show an edge with this strategy
- **Single stocks** generally underperform indices due to idiosyncratic risk and gap risk

## Files

### 1. `leva_fissa_montecarlo.py` — Monte Carlo Simulator

**Module 2 — Mathematics of Compounding**

Simulates 10,000 price paths and computes the distribution of leveraged certificate returns across different holding periods (1, 5, 10, 20, 60 days).

**What it does:**
- Generates random daily returns with configurable drift (μ) and volatility (σ)
- Computes certificate value using the exact daily rebalancing formula: `P(t) = P(t-1) × [1 + L × R(t)]`
- Compares actual certificate returns vs "naive" expected returns (L × index return)
- Produces a heatmap of P&L across different trend/volatility combinations
- Shows theoretical decay curves for leverage levels from 2x to 10x

**Key outputs:**
- `montecarlo_distributions.png` — P&L distribution histograms by holding period
- `montecarlo_paths.png` — 50 sample paths with median and 10-90th percentile bands
- `montecarlo_heatmap.png` — Trend vs volatility heatmap with breakeven line
- `montecarlo_decay_by_leverage.png` — Theoretical decay curves

**Usage:**
```bash
python leva_fissa_montecarlo.py
```

Edit the parameters at the top of the file to experiment:
```python
LEVERAGE = 7                # Certificate leverage
ANNUAL_RETURN = 0.08        # Expected annual return of underlying
ANNUAL_VOLATILITY = 0.18    # Annual volatility of underlying
```

---

### 2. `regime_scanner.py` — Market Regime Scanner

**Module 3 — Applied Technical Analysis**

Classifies the market into regimes (GREEN/AMBER/RED) using four indicators, determining whether conditions are favorable for leveraged certificate trading.

**Indicators used:**
- **ADX (14)**: Measures trend strength. ADX > 25 = trend present, < 20 = no trend
- **ATR% normalized**: Daily volatility relative to its 20-day moving average
- **ROC (5-day)**: Directional momentum confirmation
- **MACD histogram**: Timing — expanding in trend direction = favorable

**Signal logic (v2, asymmetric):**
- LONG: ADX ≥ 25 + ATR% below average + ROC > 0.5% + MACD histogram expanding positive
- SHORT: ADX ≥ 30 + ATR% below average + ROC < -0.8% + MACD histogram expanding negative (stricter thresholds for short trades)

**Key output:**
- `regime_scanner.png` — 5-panel chart: price with regime overlay, ADX, ATR%, ROC, MACD histogram

**Usage:**
```bash
python regime_scanner.py
```

The script downloads real FTSE MIB data via yfinance. If no internet is available, it falls back to synthetic data.

---

### 3. `backtest_strategy.py` — Backtest Engine v1 (Baseline)

**Module 6 — Backtesting and Validation**

Implements the complete trading strategy and tests it on historical data with walk-forward validation.

**Strategy rules:**
- Entry: on GREEN signal from regime scanner
- Stop loss: 1.5 × ATR from entry price
- Target: 1.5 × stop distance (R:R = 1:1.5)
- Max holding: 5 days
- Regime exit: close if signal changes to non-GREEN after 2+ days
- Position sizing: 1% risk per trade

**What it does:**
- Runs full in-sample backtest
- Runs walk-forward validation (150-day train / 50-day test / 50-day step)
- Computes: Sharpe, Sortino, profit factor, win rate, max drawdown, expectancy

**Usage:**
```bash
python backtest_strategy.py
```

Set `USE_SYNTHETIC = False` to use real data (requires internet).

---

### 4. `backtest_strategy_v2.py` — Backtest Engine v2 (Optimized)

**Module 6 — Optimized version with three improvements**

Runs both v1 (baseline) and v2 (optimized) on the same data and produces a comparison table.

**Optimizations vs v1:**

1. **Asymmetric Long/Short filter**: Short trades require ADX > 30 (vs 25 for Long) and ROC < -0.8% (vs 0.5% for Long), because bearish trends on equity indices are less persistent.

2. **Partial exit**: When the certificate reaches +7%, the system closes 50% of the position and moves the stop to breakeven on the remainder. This locks in profit while allowing further upside at zero risk.

3. **Real transaction costs**: Bid-ask spread (0.8% round-trip) and daily funding cost (3.5% annualized) are subtracted from the P&L.

**Key output:**
- `backtest_results.png` — 8-panel chart: equity curves (v1 vs v2), P&L per trade, distribution, exit reasons, Long vs Short boxplot, metrics comparison, walk-forward results, cumulative costs impact

**Usage:**
```bash
python backtest_strategy_v2.py
```

---

### 5. `backtest_strategy_v3.py` — Multi-Asset Multi-Leverage Optimizer

**Module 6 — Cross-asset validation**

Tests the v2 strategy across 5 indices (FTSE MIB, DAX, EuroStoxx 50, S&P 500, CAC 40) and 3 leverage levels (3x, 5x, 7x) = 15 combinations.

**What it does:**
- Downloads data for all 5 indices
- Runs full backtest + walk-forward for each of the 15 combinations
- Produces comparison tables (in-sample and OOS)
- Ranks all combinations by OOS Sharpe ratio
- Scales costs realistically by leverage level

**Key outputs:**
- Equity curves per asset (3 leverage levels overlaid)
- Heatmaps: Sharpe OOS and Profit Factor OOS by (asset, leverage)
- Scatter: Sharpe vs Max Drawdown for all combinations

**Usage:**
```bash
python backtest_strategy_v3.py
```

To add more indices, edit the `TICKERS` dictionary. To test on single stocks, replace tickers accordingly.

---

### 6. `stock_screener.py` — Stock Screener for Leveraged Trading

**Module 6 extension — Single stock candidate identification**

Screens 35 stocks (20 FTSE MIB + 15 DAX components) on 5 weighted criteria to identify the best candidates for leveraged certificate trading.

**Screening criteria (with weights):**
- **Low volatility (25%)**: ATR% — lower = less decay
- **Trend quality (30%)**: ADX average + % days trending + autocorrelation of returns
- **Liquidity (15%)**: Average daily turnover in EUR
- **Low gap risk (20%)**: Number of gaps > 2% in the past year
- **Momentum (10%)**: Daily Sharpe ratio

**Key output:**
- `screener_results.png` — 4-panel chart: score ranking, ATR% vs ADX scatter, gap risk vs score, score breakdown for top 10

**Usage:**
```bash
python stock_screener.py
```

To modify the stock universe, edit `FTSE_MIB_STOCKS` and `DAX_STOCKS` dictionaries.

## How the Files Connect

```
leva_fissa_montecarlo.py     →  Understand the product (theory)
         ↓
regime_scanner.py            →  Identify when to trade (signals)
         ↓
backtest_strategy.py         →  Test the strategy (baseline)
         ↓
backtest_strategy_v2.py      →  Optimize and add real costs
         ↓
backtest_strategy_v3.py      →  Validate across assets and leverage levels
         ↓
stock_screener.py            →  Explore single stock candidates
```

## Disclaimer

This project is for **educational purposes only**. Leveraged certificates are high-risk financial instruments that can result in significant losses. Past performance does not guarantee future results. The author is not a financial advisor. Always do your own research and consider your risk tolerance before trading.

## License

MIT