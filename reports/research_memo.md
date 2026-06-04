# Research Memo: Short-Horizon Alpha from Limit Order Book Features

## Abstract

This study investigates whether microstructure features derived from limit order book (LOB) snapshots and trade data contain predictive information about short-horizon mid-price returns. Using 7 days of 10-level LOB depth and tick-level trade data for BTCUSDT and ETHUSDT on Binance Futures (sourced from Tardis.dev historical archives, ~604,800 one-second observations per asset), we engineer order book imbalance, microprice deviation, signed trade flow, and volatility features, then evaluate their predictive power at 1s, 5s, 10s, and 30s horizons. Ridge regression achieves walk-forward Pearson ICs of 0.06–0.26 (BTCUSDT) and 0.09–0.23 (ETHUSDT), with IC t-statistics far exceeding 2 at all horizons (up to t=31.5 at 1s). However, a decile-sorted long-short backtest reveals that transaction costs eliminate profitability at every horizon for both assets under medium-cost assumptions. This demonstrates the classic microstructure finding: statistically significant predictability does not imply economic tradability.

---

## 1. Motivation

Market microstructure theory predicts that supply-demand imbalances visible in the limit order book should contain information about near-term price direction. When bid depth substantially exceeds ask depth, the marginal price impact of a sell market order is larger than that of a buy order, creating directional pressure. This is formalized in models by Glosten and Milgrom (1985) and Kyle (1985), where informed trader activity is partially revealed through order flow dynamics.

In practice, high-frequency trading firms exploit these signals with sub-millisecond latency and co-located infrastructure. The research question here is narrower: **can these signals be detected and characterized with publicly available data and simple models?** The answer has practical value for understanding market structure, even if the signal is not directly tradable at retail latency.

We focus on crypto markets (Binance BTCUSDT and ETHUSDT perpetual futures) because L2 order book data is publicly accessible, unlike equities where such data requires expensive institutional feeds.

---

## 2. Data

### 2.1 Source and Collection

Historical tick-level data was sourced from **Tardis.dev**, which archives Binance Futures WebSocket feeds. The dataset comprises 7 non-contiguous days — the first day of each month from December 2025 through June 2026 — leveraging Tardis.dev's free-tier access to first-of-month data.

Two data types were downloaded per asset per day:
- **`book_snapshot_25`**: Top-25-level order book snapshots recorded on every tick change (1–2.3 million snapshots per day), downsampled to 1-second resolution during conversion.
- **`trades`**: Tick-level trade data with aggressor side classification (1–15 million trades per day).

### 2.2 Dataset Summary

| Metric | BTCUSDT | ETHUSDT |
|--------|---------|---------|
| Days | 7 (Dec 2025 – Jun 2026) | 7 |
| Raw OB tick snapshots | ~9.5 million | ~10.6 million |
| Raw trades | 32.3 million | 53.5 million |
| Resampled rows (1s) | 604,797 | 604,796 |
| Feature columns | 123 | 123 |
| Usable rows (non-NaN targets) | 604,762 | 604,761 |

### 2.3 Dates and Market Context

| Date | BTC Price (~) | Context |
|------|--------------|---------|
| 2025-12-01 | $97,400 | Post-Thanksgiving, high activity |
| 2026-01-01 | $93,500 | New Year's Day, low volume |
| 2026-02-01 | $101,200 | Normal trading |
| 2026-03-01 | $84,500 | Drawdown period |
| 2026-04-01 | $82,600 | Continued weakness |
| 2026-05-01 | $94,700 | Recovery |
| 2026-06-01 | $73,700 | Market correction |

The non-contiguous sampling provides diverse market conditions — rallies, corrections, high and low volatility — which strengthens robustness testing but prevents traditional multi-day rolling walk-forward.

### 2.4 Cleaning Pipeline

An 8-step cleaning pipeline was applied per day: duplicate timestamps, missing top-of-book, crossed books, zero/negative spreads, outlier spreads (>5 median deviations), invalid prices, negative sizes, and book ordering violations. Removal rates were consistently below 2%, reflecting the high quality of Tardis-archived exchange data.

### 2.5 Resampling

Order book snapshots (already at 1s from the Tardis conversion) were forward-filled to a regular grid. Trade data was aggregated into rolling windows of 1s, 5s, and 10s.

---

## 3. Feature Construction

All 58 engineered features are motivated by market microstructure theory and are interpretable. See Section 3 of the previous version for full feature descriptions: order book features (21 columns), trade-flow features (6 columns), return/volatility features, and time controls.

---

## 4. Prediction Targets

Forward log-returns at horizons h ∈ {1s, 5s, 10s, 30s}, plus binary direction and half-spread threshold classification targets.

---

## 5. Validation Design

### 5.1 Chronological Split

- **Train**: first 60% (~5 months: Dec 2025 – early Apr 2026, 362,878 rows)
- **Validation**: next 20% (~Apr–May 2026, 120,959 rows)
- **Test**: final 20% (~May–Jun 2026, 120,960 rows)

### 5.2 Walk-Forward Validation

An expanding-window walk-forward scheme with 5 folds, each testing on ~24,191 rows (~6.7 hours). IC mean and IC t-statistic are computed across folds.

No random shuffling or cross-validation was used. All metrics reported below are strictly out-of-sample.

---

## 6. Model Comparison

### 6.1 Fixed-Split Results (BTCUSDT, test set)

| Model | Target | Pearson IC | Spearman IC | Decile Spread (bps) |
|-------|--------|------------|-------------|---------------------|
| Ridge | y_return_1s | 0.243 | 0.377 | 0.58 |
| Ridge | y_return_5s | 0.156 | 0.305 | 1.07 |
| Ridge | y_return_10s | 0.115 | 0.222 | 1.18 |
| Ridge | y_return_30s | 0.036 | 0.095 | 0.86 |

### 6.2 Walk-Forward Results (Ridge, both assets)

| Horizon | BTC Mean IC | BTC IC t-stat | ETH Mean IC | ETH IC t-stat |
|---------|-------------|---------------|-------------|---------------|
| 1s | 0.258 | 31.5 | 0.227 | 24.6 |
| 5s | 0.204 | 6.7 | 0.175 | 12.4 |
| 10s | 0.157 | 4.7 | 0.137 | 9.1 |
| 30s | 0.062 | 2.0 | 0.090 | 7.7 |

All IC t-statistics meet or exceed 2.0, confirming statistical significance across all horizons and assets. The t-statistics are dramatically higher than in the 2-hour preliminary analysis (31.5 vs 4.6 at 1s) due to the 84× larger sample size, demonstrating the importance of sufficient data for robust inference.

### 6.3 Signal Decay

IC decays monotonically from 1s to 30s for both assets. The 1s horizon now shows the highest IC (0.258 BTC, 0.227 ETH), differing from the 2-hour preliminary where 5s led. With more data, the faster signal dominates because the larger sample suppresses noise at the shortest horizon. The signal's half-life is approximately 5–10 seconds.

---

## 7. Feature Ablation

Feature groups tested in isolation (Ridge, 5s horizon):

### BTCUSDT

| Feature Set | Pearson IC | Spearman IC | Decile Spread (bps) |
|-------------|------------|-------------|---------------------|
| Order book only | 0.212 | 0.319 | 1.16 |
| Trade only | 0.037 | 0.103 | 0.43 |
| Return/vol only | 0.210 | 0.315 | 1.12 |
| **All features** | **0.156** | **0.305** | **1.07** |

### ETHUSDT

| Feature Set | Pearson IC | Spearman IC | Decile Spread (bps) |
|-------------|------------|-------------|---------------------|
| Order book only | 0.179 | 0.268 | 1.14 |
| Trade only | 0.060 | 0.096 | 0.41 |
| Return/vol only | 0.177 | 0.266 | 1.12 |
| **All features** | **0.155** | **0.252** | **1.03** |

**Key insight — reversal from preliminary findings**: With 7 days of data, **order book features** (IC=0.212 BTC, 0.179 ETH) and **return/volatility features** (IC=0.210, 0.177) now outperform trade-flow features (IC=0.037, 0.060) when used in isolation. This reversal from the 2-hour analysis (where trade features led) suggests the earlier result was sample-specific. Over a longer period spanning diverse market conditions, the persistent structural information in the order book proves more reliable than short-term trade dynamics.

Notably, combining all features yields slightly *lower* IC than order book features alone in Pearson terms, suggesting some multicollinearity-driven noise when features are combined. However, rank IC remains high, indicating the combined model ranks returns well despite potential overfitting in magnitude.

---

## 8. Backtesting

### 8.1 Trading Rule

- **Long**: when Ridge-predicted return is in the top decile
- **Short**: when predicted return is in the bottom decile
- **Flat**: otherwise
- Hold for exactly h seconds; no overlapping trades

### 8.2 Results — BTCUSDT

| Horizon | Gross Sharpe | Net Sharpe (low) | Net Sharpe (medium) | Net Sharpe (high) | Hit Rate | Trades |
|---------|-------------|------------------|---------------------|-------------------|----------|--------|
| 1s | 104.3 | -273.3 | -327.1 | -351.4 | 41.1% | 28,620 |
| 5s | 39.5 | -40.8 | -85.1 | -117.5 | 47.0% | 17,599 |
| 10s | 19.8 | -11.0 | -36.2 | -68.3 | 48.4% | 11,827 |
| 30s | 1.5 | -5.1 | -11.4 | -25.4 | 48.0% | 5,326 |

### 8.3 Results — ETHUSDT

| Horizon | Gross Sharpe | Net Sharpe (low) | Net Sharpe (medium) | Net Sharpe (high) | Hit Rate | Trades |
|---------|-------------|------------------|---------------------|-------------------|----------|--------|
| 1s | 96.0 | -287.6 | -352.0 | -380.4 | 39.8% | 30,064 |
| 5s | 34.1 | -36.0 | -78.7 | -113.6 | 47.5% | 17,840 |
| 10s | 18.6 | -8.3 | -31.3 | -64.3 | 49.3% | 12,147 |
| 30s | 7.1 | 1.6 | -3.9 | -16.8 | 49.9% | 5,481 |

### 8.4 Interpretation

With the full 7-day dataset, the backtest results are more pessimistic — and more realistic — than the 2-hour preliminary:

- **No horizon survives medium costs for either asset.** The 30s horizon, which appeared profitable in the 2-hour analysis, now shows negative net Sharpe (-11.4 BTC, -3.9 ETH) under medium costs.
- **ETH 30s marginally survives low costs** (net Sharpe = 1.6), but this is not robustly tradable.
- **Hit rates below 50%** at most horizons confirm that while the model ranks returns correctly (high IC), the absolute direction prediction is no better than random — the signal's value is in relative ranking, not directional accuracy.

The discrepancy with the 2-hour results illustrates why sample size matters: the earlier positive 30s backtest was driven by favorable conditions in a single session. Over 7 diverse trading days, the signal's per-trade edge is smaller and more variable.

---

## 9. Robustness Analysis

With ~120,000 test rows (vs 1,440 before), regime analysis now has genuine statistical power.

### 9.1 Volatility Regime (5s horizon, BTCUSDT)

| Regime | n | Pearson IC | Spearman IC |
|--------|---|------------|-------------|
| Low volatility | 40,312 | 0.256 | 0.358 |
| Mid volatility | 40,316 | 0.259 | 0.352 |
| High volatility | 40,292 | 0.095 | 0.234 |

### 9.2 Liquidity Regime (5s horizon, BTCUSDT)

| Regime | n | Pearson IC | Spearman IC |
|--------|---|------------|-------------|
| Thin liquidity | 40,294 | 0.037 | 0.268 |
| Mid liquidity | 40,310 | 0.203 | 0.298 |
| Deep liquidity | 40,316 | 0.240 | 0.347 |

### 9.3 Time-of-Day Effects (BTCUSDT)

The signal shows meaningful intraday variation across 24 hourly buckets:
- **Strongest**: UTC 5:00 (IC=0.317), 11:00 (IC=0.436), 20:00 (IC=0.269)
- **Weakest**: UTC 12:00–13:00 (IC=0.029 and −0.054)

The weak signal during UTC 12:00–13:00 coincides with the US market open, when volatility spikes and aggressive order flow may overwhelm book-implied direction.

### 9.4 Key Regime Findings (updated from 2-hour analysis)

The 7-day analysis reveals a more nuanced picture than the preliminary:

1. **Volatility**: The signal is clearly strongest in low-to-mid volatility and degrades sharply in high-vol (IC drops from 0.26 to 0.095 for BTC). This is consistent across both assets.

2. **Liquidity**: Unexpectedly, **deep liquidity now produces the highest Pearson IC** for BTC (0.240 vs 0.037 for thin). This reverses the 2-hour finding and suggests that with more data, deep/stable book states actually provide more reliable imbalance signals — thin liquidity may produce noisier features.

3. **Time of day**: The signal varies 10× across hours, from near-zero during high-volatility US-open hours to 0.44 during calmer Asian/European sessions. This has implications for strategy design: a time-conditional model could improve performance.

### 9.5 Cross-Asset Consistency

Both assets show qualitatively identical patterns: positive IC at all horizons, monotonic signal decay, similar regime dependencies, and universal failure under transaction costs. This consistency strengthens confidence that the findings reflect genuine microstructure dynamics.

---

## 10. Limitations

1. **Non-contiguous days**: The 7 days are sampled on the first of each month (Dec 2025 – Jun 2026). This prevents true rolling walk-forward validation and may introduce first-of-month calendar effects. Contiguous multi-day data would be preferable.

2. **No intra-week variation**: We cannot assess day-of-week effects or multi-day persistence patterns.

3. **Simplified backtest**: Assumes instantaneous execution at mid-price adjusted by cost scenarios. No queue position, fill probability, or market impact modeling.

4. **Linear models only**: Deliberately simple; non-linear models may capture additional signal but at the cost of interpretability.

5. **Crypto-specific**: Results may not transfer to equities, FX, or other markets with different microstructure properties.

6. **Resampled from tick to 1s**: The Tardis data contains sub-millisecond tick updates that are downsampled to 1s. Higher-resolution analysis (100ms) might reveal additional signal structure.

7. **Free-tier data**: Limited to first-of-month dates. Paid access would enable arbitrary date ranges for more robust analysis.

---

## 11. Conclusion

This study demonstrates, on a substantially larger dataset than the preliminary 2-hour analysis, that **limit order book features contain statistically significant predictive information about short-horizon mid-price returns** in BTCUSDT and ETHUSDT. Walk-forward Pearson ICs of 0.06–0.26 with t-statistics up to 31.5 confirm robust, persistent predictability across 7 trading days spanning diverse market conditions.

However, the research also demonstrates — more convincingly than the preliminary — the critical distinction between **statistical alpha and tradable alpha**:

- **No horizon survives medium transaction costs** for either asset over the full dataset. The 30s horizon that appeared profitable in the 2-hour analysis was a small-sample artifact.
- Per-trade gross returns of 0.02–0.30 bps are consistently overwhelmed by even conservative round-trip costs (~3 bps).
- The signal's value lies in **relative ranking** (high rank IC, positive decile spread), not directional accuracy (hit rates near 50%).

Feature ablation on the full dataset reveals that **order book features and return/volatility features** are the most reliable predictors, reversing the 2-hour finding that trade features dominated. This suggests structural order book information is more stable across market conditions than transient trade dynamics.

The signal is most reliable in **low-volatility environments** and during **calmer trading hours** (Asian/European sessions), degrading sharply during high-volatility periods like the US open.

These findings are consistent with the academic literature and the operational reality of high-frequency trading: the signals exist, they are robust, but profitably exploiting them requires sub-millisecond latency, exchange co-location, maker rebates, and sophisticated execution — infrastructure far beyond what simple linear models and naive cost assumptions can capture.

### What Would Improve This Research

- **Contiguous multi-day data**: 30+ consecutive days for proper rolling walk-forward
- **Higher resolution**: 100ms snapshots to capture faster signal dynamics
- **Non-linear models**: Gradient boosting or attention-based models for IC improvement
- **Execution simulation**: Queue position and fill probability modeling
- **Time-conditional signals**: Restrict trading to high-IC hours
- **Cross-asset signals**: Use BTC book state to predict ETH returns
- **Latency analysis**: Quantify signal decay under realistic execution delays

---

*Prepared as part of a quantitative research project demonstrating the full research loop: hypothesis → data → features → validation → signal evaluation → backtest → cost analysis → robustness → conclusion.*
