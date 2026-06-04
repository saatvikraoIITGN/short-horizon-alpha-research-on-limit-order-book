# Short-Horizon Alpha Research on Limit Order Book Data

A rigorous quantitative research project investigating whether microstructure features derived from limit order book (LOB) snapshots and trade data predict short-horizon mid-price returns. The project demonstrates the full quant research loop — from hypothesis through data engineering, feature construction, chronological validation, transaction-cost-aware backtesting, and robustness analysis — using BTCUSDT and ETHUSDT on Binance Futures.

## Research Question

> Do limit order book features predict short-horizon mid-price returns? At which horizons does the signal work best, does predictive power decay quickly, and does the signal survive realistic transaction costs?

This matters for quantitative research because microstructure signals are among the most theoretically grounded alpha sources, yet the gap between statistical predictability and economic tradability is often underappreciated. This project explicitly quantifies that gap.

## Data

- **Assets**: BTCUSDT and ETHUSDT perpetual futures (Binance)
- **Order book**: 10-level bid/ask depth snapshots at ~1s frequency
- **Trades**: Tick-level with aggressor side classification
- **Collection**: REST API polling + historical archive download scripts included
- **Duration**: ~2 hours continuous (extensible to multi-day via the same pipeline)
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

1. **Statistically significant predictability**: Walk-forward Pearson IC of 0.16–0.35 across horizons and assets (all IC t-stats > 2.0)
2. **Signal decay confirmed**: IC peaks at 1–5s and decays by 30s, consistent with microstructure theory
3. **Trade features dominate**: Trade-flow features contribute more than static order book features alone
4. **Transaction costs eliminate short-horizon edge**: 1s gross Sharpe > 150 but deeply negative after costs
5. **30s horizon survives costs**: Net Sharpe > 37 (BTC) and > 65 (ETH) under "high" cost assumptions
6. **Signal strongest in low-vol, thin-liquidity regimes**: Where order book information is most concentrated
7. **Cross-asset consistency**: Both BTC and ETH show qualitatively identical patterns

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

- **Sample size**: 2 hours of data; ideally 7–30 days for robust inference
- **Single session**: Cannot assess day-to-day stability or event sensitivity
- **Simplified execution**: No queue position, market impact, or fill probability modeling
- **Linear models only**: Deliberately simple; non-linear models may capture additional signal
- **Crypto-specific**: Results may not transfer to equities, FX, or other markets
- **REST API latency**: ~1s polling; WebSocket streams would improve data fidelity

## Future Work

- Extend to 7–30 days of WebSocket-collected data
- Add gradient boosting / neural network model comparison
- Implement realistic limit order execution simulation
- Apply to equities (LOBSTER data) for cross-market validation
- Quantify latency-adjusted signal decay
- Explore non-linear feature interactions (e.g., OBI × volume shock)
