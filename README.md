# Short-Horizon Alpha Research on Limit Order Book Data

A rigorous quantitative research project investigating whether microstructure features derived from limit order book (LOB) snapshots and trade data predict short-horizon mid-price returns. The project demonstrates the full quant research loop — from hypothesis through data engineering, feature construction, chronological validation, transaction-cost-aware backtesting, and robustness analysis — using BTCUSDT and ETHUSDT on Binance Futures.

## Research Question

> Do limit order book features predict short-horizon mid-price returns? At which horizons does the signal work best, does predictive power decay quickly, and does the signal survive realistic transaction costs?

This matters for quantitative research because microstructure signals are among the most theoretically grounded alpha sources, yet the gap between statistical predictability and economic tradability is often underappreciated. This project explicitly quantifies that gap.

## Data

- **Assets**: BTCUSDT and ETHUSDT perpetual futures (Binance)
- **Order book**: 10-level bid/ask depth snapshots at 1s frequency (downsampled from tick-level)
- **Trades**: Tick-level with aggressor side classification (32M BTC, 54M ETH trades)
- **Source**: Tardis.dev historical archives (Binance Futures WebSocket recordings) + live REST API collector
- **Duration**: 7 full trading days (first of each month, Dec 2025 – Jun 2026), ~604,800 rows per asset
- **Sampling**: Resampled to 1-second regular grid; trades aggregated into 1s/5s/10s rolling windows

## Methodology

1. **Data Pipeline**: Collect → Clean (8-step pipeline) → Resample → Align order book and trade streams
2. **Feature Engineering**: 58 features across 4 groups (order book, trade-flow, return/volatility, time controls)
3. **Target Construction**: Forward log-returns and classification targets at 1s, 5s, 10s, 30s horizons
4. **Modeling**: Linear, Ridge, Lasso regression with chronological train/validation/test split
5. **Walk-Forward Validation**: 5-fold expanding-window scheme with fold-level IC statistics
6. **Backtesting**: Decile-sorted long/short strategy under 4 cost scenarios (zero, low, medium, high)
7. **Robustness**: Regime analysis (volatility, liquidity, spread), feature ablation, cross-asset comparison

## Key Features

| Group | Features | Motivation |
|-------|----------|------------|
| Order Book | OBI (levels 1,3,5,10), weighted OBI, microprice deviation, spread, depth, depth slope | Supply-demand imbalance signals directional pressure |
| Trade Flow | Signed volume, trade imbalance, large trade indicator, volume shock | Recent execution dynamics reveal informed trading |
| Return/Vol | Lagged returns (1–30s), realized volatility (10–60s), rolling z-scores | Momentum and mean-reversion at micro timescales |
| Time | Hour, minute, day of week | Control for intraday seasonality |

## Validation Framework

- **Chronological split**: 60% train / 20% validation / 20% test — no random shuffling
- **Walk-forward**: Expanding-window with 5 non-overlapping test folds
- **Metrics**: Pearson IC, Spearman rank IC, IC t-statistic, decile spread, MSE
- **No lookahead bias**: All features use only past data; targets are forward-looking

## Backtesting Setup

- **Strategy**: Long top decile, short bottom decile of predicted returns; hold for h seconds
- **Cost scenarios**: Zero (0 bps), Low (1 bps fee + 0.5 bps slippage), Medium (2+1 bps), High (5+2 bps)
- **Metrics**: Gross/net Sharpe, PnL, hit rate, max drawdown, turnover, cost sensitivity

## Main Findings

1. **Statistically significant predictability**: Walk-forward Pearson IC of 0.06–0.26 across horizons and assets (IC t-stats up to 31.5, all > 2.0)
2. **Signal decay confirmed**: IC peaks at 1s and decays monotonically to 30s, consistent with microstructure theory
3. **Order book features most stable**: Over 7 days, order book imbalance (IC=0.21) outperforms trade-flow features (IC=0.04), reversing the 2-hour preliminary finding
4. **Transaction costs eliminate edge at all horizons**: No horizon survives medium costs for either asset over the full dataset
5. **Statistical alpha ≠ tradable alpha**: Gross Sharpe 1.5–104 collapses to negative net Sharpe under any realistic cost scenario
6. **Signal strongest in low-volatility regimes**: IC drops from 0.26 to 0.10 in high-volatility conditions
7. **Strong intraday variation**: IC varies 10× across hours, weakest during US-open volatility spike
8. **Cross-asset consistency**: Both BTC and ETH show qualitatively identical patterns across 7 months

## Repository Structure

```
lob-alpha-research/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/              # Raw order book and trade parquet files
│   ├── interim/          # Cleaned, resampled, aligned data
│   └── processed/        # Feature-engineered datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_backtest_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── download_binance_data.py   # WebSocket, REST, historical downloaders
│   │   ├── load_data.py               # Parquet loading utilities
│   │   ├── clean_data.py              # 8-step cleaning pipeline
│   │   └── resample_data.py           # Time-grid resampling and alignment
│   ├── features/
│   │   ├── orderbook_features.py      # OBI, microprice, spread, depth
│   │   ├── trade_features.py          # Volume shock, large trade detection
│   │   ├── target_construction.py     # Forward returns, direction, half-spread
│   │   └── feature_pipeline.py        # Orchestrates all feature engineering
│   ├── models/
│   │   ├── train_model.py             # Linear, Ridge, Lasso training + grids
│   │   ├── predict.py                 # Prediction utilities
│   │   └── model_utils.py             # Feature group definitions, ablation support
│   ├── validation/
│   │   ├── walk_forward.py            # Chronological split + walk-forward
│   │   └── metrics.py                 # IC, Sharpe, drawdown, decile spread, etc.
│   ├── backtest/
│   │   ├── strategy.py                # Decile long/short signal backtest
│   │   ├── cost_model.py              # Fee, slippage, spread cost scenarios
│   │   └── performance.py             # Equity curves, cost sensitivity plots
│   ├── analysis/
│   │   ├── eda.py                     # 10 EDA analyses (19 figures)
│   │   ├── signal_decay.py            # IC decay across horizons
│   │   ├── robustness.py              # Regime, ablation, cross-asset analysis
│   │   └── plots.py                   # Consistent styling and figure saving
│   └── utils/
│       ├── paths.py                   # Config loading, directory management
│       └── logging_utils.py           # Standardized logging
├── reports/
│   ├── figures/           # 30 publication-quality plots
│   ├── tables/            # Summary stats, walk-forward, backtest CSVs
│   └── research_memo.md   # 4–8 page research writeup
└── tests/
```

## How to Run

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Collect Data

```bash
# Collect live order book + trades via REST API (runs for specified duration)
python -c "
from src.data.download_binance_data import RESTCollector
import asyncio
collector = RESTCollector(symbols=['BTCUSDT','ETHUSDT'], duration_seconds=7200)
asyncio.run(collector.run())
"
```

### Step 2: Run Full Pipeline

```bash
python -c "
from src.data.load_data import load_raw_orderbook, load_raw_trades
from src.data.clean_data import run_cleaning_pipeline
from src.data.resample_data import resample_and_align
from src.features.feature_pipeline import run_feature_pipeline
from src.utils.paths import load_config, ensure_dir, get_data_dir

config = load_config()
for sym in config['assets']:
    ob = load_raw_orderbook(sym)
    tr = load_raw_trades(sym)
    clean_ob, clean_tr, _ = run_cleaning_pipeline(ob, tr, book_levels=10)
    aligned = resample_and_align(clean_ob, clean_tr, frequency='1s')
    featured = run_feature_pipeline(aligned, config)
    out = ensure_dir(get_data_dir('processed') / sym)
    featured.to_parquet(out / 'features.parquet', index=False)
"
```

### Step 3: Run Analysis

```bash
# EDA (generates 19 figures)
python -c "
from src.data.load_data import load_processed
from src.analysis.eda import run_full_eda
datasets = {s: load_processed(s, 'features') for s in ['BTCUSDT','ETHUSDT']}
run_full_eda(datasets)
"

# Modeling + Walk-forward + Backtest (see notebooks/ for interactive versions)
```

## Limitations

- **Non-contiguous days**: 7 days sampled on first-of-month; cannot test multi-day persistence or true rolling walk-forward
- **Simplified execution**: No queue position, market impact, or fill probability modeling
- **Linear models only**: Deliberately simple; non-linear models may capture additional signal
- **Crypto-specific**: Results may not transfer to equities, FX, or other markets
- **1s resolution**: Tardis data downsampled from tick-level; 100ms analysis might reveal finer signal structure
- **Free-tier data constraint**: Tardis free tier limits to first-of-month dates

## Future Work

- Obtain 30+ consecutive days for proper rolling walk-forward validation
- Add gradient boosting / neural network model comparison
- Implement realistic limit order execution simulation with queue position modeling
- Apply to equities (LOBSTER data) for cross-market validation
- Build time-conditional model (restrict trading to high-IC hours)
- Explore cross-asset signals (BTC book → ETH prediction)
- Quantify latency-adjusted signal decay

## Resume Bullets

- Researched short-horizon alpha signals on 7 days of limit order book data (~605K observations per asset) using order book imbalance, microprice, spread, and signed order-flow features; achieved walk-forward Pearson IC of 0.06–0.26 (t-stat up to 31.5) across 1–30s horizons for BTCUSDT and ETHUSDT.
- Built walk-forward validation and transaction-cost-aware backtesting framework for high-frequency mid-price prediction; analyzed gross/net Sharpe, turnover, hit rate, and cost sensitivity across 4 fee scenarios, demonstrating that no horizon survived medium costs over 7 trading days.
- Found that microstructure signals showed robust statistical predictive power (IC > 0 at all horizons with high significance) but were entirely non-tradable under realistic costs, quantifying the precise gap between statistical alpha and tradable alpha.

## STAR Interview Explanation

| Component | Description |
|-----------|-------------|
| **Situation** | Investigated whether short-term supply-demand imbalance visible in the limit order book could predict near-term mid-price movement in crypto futures markets. |
| **Task** | Built a complete research pipeline to clean L2 order book and trade data, engineer microstructure features, validate predictive power out-of-sample with chronological and walk-forward splits, and evaluate whether the signal survived realistic transaction costs. |
| **Action** | Collected 7 days of 10-level order book snapshots and tick-level trades (32M+ BTC, 54M+ ETH trades) from Tardis.dev historical archives. Constructed 58 features including order book imbalance, weighted imbalance, microprice deviation, spread, signed trade volume, volatility, and lagged returns. Defined targets at 1s, 5s, 10s, and 30s horizons. Evaluated Ridge regression via walk-forward validation (IC/rank IC), analyzed signal decay, ran feature ablation across 4 feature groups, and implemented a decile-sorted long-short backtest under 4 cost scenarios. Assessed robustness across volatility, liquidity, and time-of-day regimes for both assets. |
| **Result** | Demonstrated statistically significant predictability (IC = 0.06–0.26, t-stat up to 31.5) that decayed from 1s to 30s. Transaction costs eliminated profitability at all horizons (gross Sharpe 1.5–104 → net Sharpe deeply negative). Order book features proved more stable than trade features across 7 months of diverse market conditions. The signal was strongest in low-volatility regimes (IC = 0.26) and degraded sharply in high-volatility conditions (IC = 0.10). |
