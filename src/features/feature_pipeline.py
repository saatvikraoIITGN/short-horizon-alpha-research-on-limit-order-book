"""
End-to-end feature pipeline.

Orchestrates: order-book features -> trade features -> return/volatility
features -> time controls -> prediction targets.  Saves the result to
``data/processed/<SYMBOL>/features.parquet``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.features.orderbook_features import compute_all_orderbook_features
from src.features.trade_features import compute_all_trade_features
from src.features.target_construction import compute_all_targets
from src.utils.logging_utils import setup_logger
from src.utils.paths import ensure_dir, get_data_dir, load_config

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Return & volatility features (backward-looking, no lookahead)
# ---------------------------------------------------------------------------

def compute_return_features(
    df: pd.DataFrame,
    lags: list = None,
    frequency: str = "1s",
) -> pd.DataFrame:
    """Lagged mid-price log-returns at each lag window."""
    if lags is None:
        lags = ["1s", "5s", "10s", "30s"]

    freq_td = pd.Timedelta(frequency)
    for lag in lags:
        shift = int(pd.Timedelta(lag) / freq_td)
        past_mid = df["mid_price"].shift(shift)
        df[f"return_lag_{lag}"] = np.log(df["mid_price"] / past_mid)

    return df


def compute_volatility_features(
    df: pd.DataFrame,
    windows: list = None,
    frequency: str = "1s",
) -> pd.DataFrame:
    """Realized volatility = rolling std of 1s log-returns over each window."""
    if windows is None:
        windows = ["10s", "30s", "60s"]

    if "return_lag_1s" not in df.columns:
        df = compute_return_features(df, lags=["1s"], frequency=frequency)

    freq_td = pd.Timedelta(frequency)
    for w in windows:
        n = int(pd.Timedelta(w) / freq_td)
        df[f"realized_vol_{w}"] = df["return_lag_1s"].rolling(n, min_periods=2).std()

    return df


# ---------------------------------------------------------------------------
# Rolling z-scores (normalise features for stationarity)
# ---------------------------------------------------------------------------

def compute_rolling_zscores(
    df: pd.DataFrame,
    columns: list = None,
    lookback: int = 300,
) -> pd.DataFrame:
    """
    z = (x - rolling_mean) / rolling_std

    Applied to key features so models see standardised, roughly stationary inputs.
    """
    if columns is None:
        candidates = ["return_lag_1s", "spread", "total_volume_1s", "obi_1"]
        columns = [c for c in candidates if c in df.columns]

    for col in columns:
        rm = df[col].rolling(lookback, min_periods=10).mean()
        rs = df[col].rolling(lookback, min_periods=10).std()
        df[f"zscore_{col}"] = ((df[col] - rm) / rs).where(rs > 0, 0.0)

    return df


# ---------------------------------------------------------------------------
# Time controls
# ---------------------------------------------------------------------------

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek

    # Session buckets (crypto trades 24/7, but activity varies)
    df["session"] = pd.cut(
        df["hour"],
        bins=[-1, 4, 8, 12, 16, 20, 24],
        labels=["asia_late", "asia_early", "europe_am", "us_am", "us_pm", "asia_evening"],
    )
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_feature_pipeline(
    aligned_df: pd.DataFrame,
    config: Optional[dict] = None,
    frequency: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the complete feature pipeline on a resampled-and-aligned DataFrame.

    Parameters
    ----------
    aligned_df : DataFrame produced by ``resample_and_align()``.
    config     : Project config dict.  Falls back to ``load_config()``.
    frequency  : Resample frequency (default from config).

    Returns
    -------
    DataFrame with all features **and** all targets appended.
    """
    if config is None:
        config = load_config()

    freq = frequency or config["data"]["resample_frequency"]
    feat_cfg = config.get("features", {})
    tgt_cfg = config.get("targets", {})
    book_levels = config["data"].get("book_levels", 5)

    imbalance_levels = feat_cfg.get("imbalance_levels", [1, 3, 5])
    trade_windows = feat_cfg.get("trade_windows", ["1s", "5s", "10s"])
    vol_windows = feat_cfg.get("volatility_windows", ["10s", "30s", "60s"])
    return_lags = feat_cfg.get("return_lags", ["1s", "5s", "10s", "30s"])
    horizons = tgt_cfg.get("horizons", ["1s", "5s", "10s", "30s"])

    df = aligned_df.copy()
    n0 = len(df.columns)

    logger.info("=" * 60)
    logger.info("FEATURE PIPELINE  |  rows=%d  initial_cols=%d", len(df), n0)
    logger.info("=" * 60)

    # 1. Order-book features
    df = compute_all_orderbook_features(df, imbalance_levels, book_levels)

    # 2. Trade-flow features
    df = compute_all_trade_features(df, trade_windows)

    # 3. Return & volatility
    df = compute_return_features(df, return_lags, freq)
    df = compute_volatility_features(df, vol_windows, freq)

    # 4. Rolling z-scores
    df = compute_rolling_zscores(df)

    # 5. Time controls
    df = compute_time_features(df)

    # 6. Targets (forward-looking)
    df = compute_all_targets(df, horizons, freq)

    logger.info("=" * 60)
    logger.info("DONE  |  rows=%d  cols=%d  (added %d)", len(df), len(df.columns), len(df.columns) - n0)
    logger.info("=" * 60)

    return df


# ---------------------------------------------------------------------------
# Convenience: load, feature-engineer, and save for one symbol
# ---------------------------------------------------------------------------

def build_features_for_symbol(symbol: str, config: Optional[dict] = None) -> pd.DataFrame:
    """Load interim aligned data, run feature pipeline, and save to processed/."""
    from src.data.load_data import load_interim

    if config is None:
        config = load_config()

    aligned = load_interim(symbol, name="aligned")
    featured = run_feature_pipeline(aligned, config)

    out_dir = ensure_dir(get_data_dir("processed") / symbol.upper())
    out_path = out_dir / "features.parquet"
    featured.to_parquet(out_path, index=False)
    logger.info("Saved %s  (%s)", out_path, featured.shape)

    return featured
