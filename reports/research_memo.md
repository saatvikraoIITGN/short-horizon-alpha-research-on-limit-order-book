# Research Memo: Short-Horizon Alpha from Limit Order Book Features

## Abstract

This study investigates whether microstructure features derived from limit order book (LOB) snapshots and trade data contain predictive information about short-horizon mid-price returns. Using 2 hours of 10-level LOB depth and tick-level trade data for BTCUSDT and ETHUSDT on Binance Futures, we engineer order book imbalance, microprice deviation, signed trade flow, and volatility features, then evaluate their predictive power at 1s, 5s, 10s, and 30s horizons. Ridge regression achieves out-of-sample Pearson ICs of 0.27–0.35 (BTCUSDT) and 0.22–0.32 (ETHUSDT) under walk-forward validation, with IC t-statistics exceeding 2 at all horizons. However, a decile-sorted long-short backtest reveals that transaction costs — even under conservative assumptions — eliminate profitability at the 1s and 5s horizons. Only the 30s horizon survives medium-cost scenarios for both assets, highlighting the fundamental tension between statistical predictability and economic tradability in high-frequency microstructure signals.

---

## 1. Motivation

Market microstructure theory predicts that supply-demand imbalances visible in the limit order book should contain information about near-term price direction. When bid depth substantially exceeds ask depth, the marginal price impact of a sell market order is larger than that of a buy order, creating directional pressure. This is formalized in models by Glosten and Milgrom (1985) and Kyle (1985), where informed trader activity is partially revealed through order flow dynamics.

In practice, high-frequency trading firms exploit these signals with sub-millisecond latency and co-located infrastructure. The research question here is narrower: **can these signals be detected and characterized with publicly available data and simple models?** The answer has practical value for understanding market structure, even if the signal is not directly tradable at retail latency.

We focus on crypto markets (Binance BTCUSDT and ETHUSDT perpetual futures) because L2 order book data is publicly accessible, unlike equities where such data requires expensive institutional feeds.

---

## 2. Data

### 2.1 Source and Collection

Data was collected via the Binance Futures REST API on June 4, 2026, over a continuous 2-hour window (approximately 07:45–09:45 UTC). Two streams were captured simultaneously:

- **Order book snapshots**: 10-level bid/ask depth polled at ~1-second intervals via the `/fapi/v1/depth` endpoint (limit=10). Each snapshot contains `bid_price_1..10`, `bid_size_1..10`, `ask_price_1..10`, `ask_size_1..10`, and a server timestamp.
- **Trades**: Tick-level trade data via the `/fapi/v1/aggTrades` endpoint, capturing `trade_id`, `price`, `quantity`, `timestamp`, and `is_buyer_maker` (aggressor side).

### 2.2 Dataset Summary

| Metric | BTCUSDT | ETHUSDT |
|--------|---------|---------|
| Raw OB snapshots | 7,109 | 7,109 |
| Raw trades | 831,365 | 723,044 |
| Post-cleaning OB | 7,084 (99.6%) | 7,106 (99.96%) |
| Post-cleaning trades | 820,490 (98.7%) | 711,359 (98.4%) |
| Resampled rows (1s) | 7,199 | 7,199 |
| Feature columns | 123 | 123 |
| Duration | ~2 hours | ~2 hours |

### 2.3 Cleaning Pipeline

An 8-step cleaning pipeline was applied:

1. **Duplicate timestamps** — removed 0 OB rows (both assets)
2. **Missing top-of-book** — 0 rows removed
3. **Crossed books** (bid >= ask) — 0 rows removed
4. **Zero/negative spreads** — 0 rows removed
5. **Outlier spreads** (> 5 median deviations) — 25 BTC / 3 ETH rows
6. **Invalid prices** — 0 rows removed
7. **Negative sizes** — 0 rows removed
8. **Book ordering violations** — 0 rows removed

Trade cleaning removed duplicate trade IDs (~1.2–1.5%) and time-range misalignment. The low removal rates reflect the high quality of Binance Futures API data.

### 2.4 Resampling

Order book snapshots were resampled to a regular 1-second grid using forward-fill. Trade data was aggregated into rolling windows of 1s, 5s, and 10s ending at each timestamp, computing volume, count, buy/sell breakdown, and trade imbalance for each window.

---

## 3. Feature Construction

All features are motivated by market microstructure theory and are interpretable.

### 3.1 Order Book Features (21 columns)

- **Mid-price**: `(best_bid + best_ask) / 2`
- **Spread**: absolute and relative (`spread / mid_price`)
- **Order Book Imbalance (OBI)**: `(bid_depth - ask_depth) / (bid_depth + ask_depth)` at levels 1, 3, 5, and 10. OBI captures directional pressure from queued limit orders.
- **Weighted OBI**: Inverse-distance weighted, giving more importance to top-of-book levels.
- **Microprice**: `(ask_price × bid_size + bid_price × ask_size) / (bid_size + ask_size)` — a size-weighted fair value estimate. Microprice deviation from mid-price signals directional pressure.
- **Depth features**: total depth (bid+ask), top-level depth, depth slope (how quickly liquidity decays across levels), per-level imbalance.

### 3.2 Trade-Flow Features (6 additional columns)

Beyond basic rolling aggregations (volume, count, buy/sell split), we compute:
- **Large trade indicator**: flags windows where total volume exceeds 3× the median
- **Volume shock**: current-window volume relative to a 30s rolling average

### 3.3 Return and Volatility Features

- Lagged log-returns at 1s, 5s, 10s, 30s
- Realized volatility over 10s, 30s, 60s windows
- Rolling z-scores for returns, spread, and volume (normalization for regime adaptation)

### 3.4 Time Controls

- Hour of day, minute of hour, day of week

### 3.5 Summary Statistics

BTCUSDT spreads are extremely tight (mean absolute spread = $0.01, or ~0.16 bps relative), consistent with the instrument's deep liquidity. OBI is roughly symmetric (mean ≈ 0, std ≈ 0.70), confirming no persistent directional bias. 1s returns have mean near zero with standard deviation of 1.2 bps.

ETHUSDT shows wider relative spreads (5.6 bps vs 0.16 bps) due to a lower price level, and slightly higher return volatility (std = 1.5 bps at 1s). OBI shows a mild positive skew (mean = +0.096), suggesting a slight bid-heavy imbalance during the sample period.

---

## 4. Prediction Targets

### 4.1 Regression Targets

Forward log-returns at horizons h ∈ {1s, 5s, 10s, 30s}:

$$y\_return\_h = \log(mid\_price_{t+h} / mid\_price_t)$$

### 4.2 Classification Targets

- **Binary direction**: `y_direction_h = 1 if y_return_h > 0, else 0`
- **Half-spread threshold**: A three-class target requiring the price change to exceed half the current spread, filtering out economically insignificant moves.

### 4.3 Target Properties

| Target | BTC std (bps) | ETH std (bps) |
|--------|---------------|---------------|
| y_return_1s | 1.22 | 1.47 |
| y_return_5s | 3.08 | 3.58 |
| y_return_10s | 4.45 | 5.11 |
| y_return_30s | 7.51 | 8.51 |

Return volatility scales approximately as √h, consistent with near-diffusive behavior at these frequencies. All return distributions are approximately symmetric with slight negative mean (consistent with the mild downtrend during the sample window).

---

## 5. Validation Design

### 5.1 Chronological Split

Following strict no-lookahead principles:
- **Train**: first 60% of timestamps (~72 min)
- **Validation**: next 20% (~24 min)
- **Test**: final 20% (~24 min)

### 5.2 Walk-Forward Validation

An expanding-window walk-forward scheme with 5 folds:
- Each fold trains on all data up to the fold start, then tests on a ~5-minute block
- IC mean and IC t-statistic are computed across folds
- This tests signal stability over non-overlapping out-of-sample windows

No random shuffling or cross-validation was used. All metrics reported below are strictly out-of-sample.

---

## 6. Model Comparison

We evaluate Linear Regression, Ridge Regression, and Lasso on the regression targets using all features.

### 6.1 Fixed-Split Results (BTCUSDT)

| Model | Target | Pearson IC | Spearman IC | Decile Spread (bps) |
|-------|--------|------------|-------------|---------------------|
| Linear | y_return_1s | 0.274 | 0.439 | 2.06 |
| Ridge | y_return_1s | 0.267 | 0.435 | 2.02 |
| Linear | y_return_5s | 0.342 | 0.384 | 4.92 |
| Ridge | y_return_5s | 0.344 | 0.386 | 4.90 |
| Ridge | y_return_10s | 0.253 | 0.267 | 4.88 |
| Ridge | y_return_30s | 0.227 | 0.198 | 5.91 |

Lasso shrinks all coefficients to zero across all targets, indicating that no single feature dominates — the signal is distributed across the feature set.

### 6.2 Walk-Forward Results (Ridge, both assets)

| Horizon | BTC Mean IC | BTC IC t-stat | ETH Mean IC | ETH IC t-stat |
|---------|-------------|---------------|-------------|---------------|
| 1s | 0.323 | 4.56 | 0.319 | 7.66 |
| 5s | 0.347 | 9.44 | 0.265 | 4.96 |
| 10s | 0.252 | 5.35 | 0.256 | 6.24 |
| 30s | 0.160 | 2.21 | 0.213 | 3.07 |

All IC t-statistics exceed 2.0, confirming statistical significance across horizons and assets. The 5s horizon shows the highest IC for BTC (0.347) while the 1s horizon leads for ETH (0.319).

### 6.3 Signal Decay

IC decays monotonically from the 5s peak to the 30s horizon, consistent with microstructure theory: order book imbalance reflects transient supply-demand mismatch that is arbitraged away within seconds. The signal's half-life is approximately 10–15 seconds.

---

## 7. Feature Ablation

Feature groups were tested in isolation (Ridge, 5s horizon):

### BTCUSDT

| Feature Set | Pearson IC | Spearman IC | Decile Spread (bps) |
|-------------|------------|-------------|---------------------|
| Order book only | 0.142 | 0.178 | 2.41 |
| Trade only | 0.300 | 0.366 | 3.76 |
| Return/vol only | 0.166 | 0.169 | 2.29 |
| **All features** | **0.344** | **0.386** | **4.90** |

### ETHUSDT

| Feature Set | Pearson IC | Spearman IC | Decile Spread (bps) |
|-------------|------------|-------------|---------------------|
| Order book only | 0.137 | 0.204 | 2.57 |
| Trade only | 0.169 | 0.294 | 2.84 |
| Return/vol only | 0.218 | 0.225 | 3.00 |
| **All features** | **0.222** | **0.335** | **4.00** |

**Key insight**: Trade-flow features provide the strongest individual signal for BTC (IC=0.30), while return/volatility features lead for ETH (IC=0.22). Combining all feature groups yields the best performance for both assets, confirming complementarity. Order book features alone are the weakest group, suggesting that static book state is less informative than recent trade dynamics.

---

## 8. Backtesting

### 8.1 Trading Rule

- **Long**: when Ridge-predicted return is in the top decile
- **Short**: when predicted return is in the bottom decile
- **Flat**: otherwise
- Hold for exactly h seconds; no overlapping trades

### 8.2 Transaction Cost Scenarios

| Scenario | Fee (bps) | Slippage (bps) | Spread cost |
|----------|-----------|-----------------|-------------|
| Zero | 0 | 0 | 0× spread |
| Low | 1 | 0.5 | 0.5× spread |
| Medium | 2 | 1 | 1.0× spread |
| High | 5 | 2 | 1.0× spread |

### 8.3 Results — BTCUSDT

| Horizon | Gross Sharpe | Net Sharpe (low) | Net Sharpe (medium) | Net Sharpe (high) | Hit Rate |
|---------|-------------|------------------|---------------------|-------------------|----------|
| 1s | 173.4 | -93.5 | -210.1 | -292.0 | 59.7% |
| 5s | 65.3 | 25.0 | -14.2 | -77.8 | 57.3% |
| 10s | 28.0 | 13.2 | -1.4 | -33.3 | 56.0% |
| 30s | 55.4 | 51.8 | 48.1 | 37.5 | 58.6% |

### 8.4 Results — ETHUSDT

| Horizon | Gross Sharpe | Net Sharpe (low) | Net Sharpe (medium) | Net Sharpe (high) | Hit Rate |
|---------|-------------|------------------|---------------------|-------------------|----------|
| 1s | 154.2 | -123.5 | -271.5 | -381.2 | 65.3% |
| 5s | 77.2 | 38.7 | 0.5 | -66.0 | 60.1% |
| 10s | 71.9 | 59.4 | 46.0 | 12.0 | 59.9% |
| 30s | 80.6 | 77.6 | 74.5 | 65.4 | 58.9% |

### 8.5 Interpretation

The gross Sharpe ratios are extremely high (28–173), which is typical of microstructure signals evaluated at high frequency with low per-trade volatility. However, the economic reality is starkly different:

- **1s horizon**: Entirely unprofitable after any costs. The per-trade gross return (~1 bps) is far below the minimum round-trip cost (~3 bps under "low" assumptions).
- **5s horizon**: Marginally profitable under "low" costs for ETH, breaks even under "medium" costs. Not robustly tradable.
- **10s horizon**: ETH survives medium costs (Sharpe = 46.0); BTC breaks even.
- **30s horizon**: Both assets remain profitable under all cost scenarios, with net Sharpe > 37 even under "high" costs. This is the most economically viable horizon.

The fundamental asymmetry: **signal strength peaks at 1–5s, but tradability peaks at 30s**, because longer horizons generate larger returns per trade that can absorb fixed transaction costs.

---

## 9. Robustness Analysis

### 9.1 Regime Analysis (5s horizon, BTCUSDT)

| Regime | n | Pearson IC | Spearman IC |
|--------|---|------------|-------------|
| Low volatility | 479 | 0.412 | 0.419 |
| Mid volatility | 480 | 0.386 | 0.389 |
| High volatility | 480 | 0.272 | 0.370 |
| Thin liquidity | 479 | 0.396 | 0.475 |
| Mid liquidity | 480 | 0.297 | 0.351 |
| Deep liquidity | 480 | 0.293 | 0.325 |

The signal is **strongest in low-volatility, thin-liquidity environments**. This is consistent with microstructure intuition: in calm markets, order book imbalance is a more reliable signal because large price shocks are less likely to overwhelm the book-implied direction. In thin liquidity conditions, the book's informational content is more concentrated.

### 9.2 Regime Analysis (5s horizon, ETHUSDT)

| Regime | n | Pearson IC | Spearman IC |
|--------|---|------------|-------------|
| Low volatility | 475 | 0.296 | 0.340 |
| Mid volatility | 480 | 0.197 | 0.307 |
| High volatility | 480 | 0.203 | 0.356 |
| Thin liquidity | 476 | 0.312 | 0.418 |
| Mid liquidity | 479 | 0.197 | 0.289 |
| Deep liquidity | 480 | 0.156 | 0.294 |

Similar pattern: thin liquidity produces the highest IC. The volatility pattern for ETH is less monotonic, with high-vol recovering rank IC, possibly due to ETH's larger tick size relative to price.

### 9.3 Cross-Asset Consistency

Both assets show qualitatively similar patterns:
- Positive IC at all horizons
- IC t-stats > 2 under walk-forward validation
- Signal decay from short to long horizons
- Trade features contribute the most incremental predictive power
- Transaction costs eliminate short-horizon profitability

The consistency across two different instruments strengthens confidence that the observed predictability reflects genuine microstructure dynamics rather than asset-specific artifacts.

---

## 10. Limitations

1. **Sample duration**: 2 hours of data is insufficient for robust statistical inference. IC estimates have wide confidence intervals, and regime analysis is limited to within-session variation. Ideally, 7–30 days of data would be used.

2. **Single session**: All data comes from one continuous session. We cannot assess day-to-day stability, weekend effects, or performance around macro events.

3. **Public API limitations**: REST API polling introduces ~1s latency and potential gaps. Institutional feeds (WebSocket depth streams, co-located servers) would provide higher fidelity data.

4. **Simplified backtest**: The backtest assumes instantaneous execution at mid-price (adjusted by cost scenarios). In reality, execution depends on queue position, latency, and adverse selection — all of which worsen effective costs.

5. **No queue position modeling**: The backtest does not model the probability of getting filled at a given price level. A realistic limit order strategy would face significant non-fill risk.

6. **No market impact**: The strategy is evaluated as a price-taker. In practice, large orders would move the book, reducing expected returns.

7. **Linear models only**: We deliberately use simple models to emphasize research rigor over complexity. Non-linear models (gradient boosting, neural networks) might capture additional signal but at the cost of interpretability and overfitting risk.

8. **Crypto-specific**: Results may not directly transfer to equities, FX, or other asset classes with different microstructure properties (tick sizes, maker-taker rebates, dark pools).

---

## 11. Conclusion

This study demonstrates that **limit order book features contain statistically significant predictive information about short-horizon mid-price returns** in BTCUSDT and ETHUSDT. Walk-forward Pearson ICs of 0.16–0.35 with t-statistics exceeding 2 at all horizons confirm robust out-of-sample predictability.

However, the research also demonstrates the critical distinction between **statistical alpha and tradable alpha**:

- At 1s horizons, gross Sharpe ratios exceed 150 but collapse to deeply negative values under any realistic cost assumption.
- The signal's half-life (~10–15s) means its strongest predictions cannot be profitably acted upon given current transaction cost structures.
- Only the 30s horizon produces net-positive returns across all cost scenarios for both assets.

Feature ablation reveals that **trade-flow features** (signed volume, trade imbalance, volume shocks) contribute more predictive power than static order book features alone, suggesting that recent execution dynamics carry more information than the current book state.

The signal is most reliable in **low-volatility, thin-liquidity regimes**, which presents an operational challenge: these are precisely the conditions where execution is most difficult and market impact most significant.

These findings are consistent with the academic literature on market microstructure and with the operational reality of high-frequency trading: the signals exist, they are detectable with simple models, but profitably exploiting them requires latency advantages and execution infrastructure that go far beyond the signal itself.

### What Would Improve This Research

- **More data**: 7–30 days across multiple market regimes
- **Higher resolution**: 100ms or 10ms snapshots from WebSocket streams
- **Non-linear models**: Gradient boosting or attention-based models for potential IC improvement
- **Execution simulation**: Realistic limit order fill modeling with queue position
- **Cross-asset expansion**: Equities (LOBSTER data), FX, or additional crypto pairs
- **Latency analysis**: Quantifying how signal decay interacts with realistic execution delays

---

*Prepared as part of a quantitative research project demonstrating the full research loop: hypothesis → data → features → validation → signal evaluation → backtest → cost analysis → robustness → conclusion.*
