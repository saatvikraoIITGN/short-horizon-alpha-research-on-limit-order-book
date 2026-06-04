"""
Additional trade-flow features beyond the base aggregation in resample_data.

resample_data already produces: total_volume, buy/sell/signed_volume,
trade_imbalance, num_trades, avg_trade_size, vwap -- per rolling window.

This module adds:
- **large_trade_indicator** -- flags windows with outsized volume.
- **volume_shock** -- current volume relative to a longer lookback average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def compute_large_trade_indicator(
    df: pd.DataFrame,
    windows: list = None,
    threshold_mult: float = 3.0,
) -> pd.DataFrame:
    """
    Binary flag = 1 when avg_trade_size in *window* exceeds
    ``threshold_mult`` times the rolling median trade size.
    """
    if windows is None:
        windows = ["1s", "5s", "10s"]

    for w in windows:
        col = f"avg_trade_size_{w}"
        if col not in df.columns:
            continue
        rolling_med = df[col].rolling(300, min_periods=10).median()
        df[f"large_trade_{w}"] = (df[col] > threshold_mult * rolling_med).astype(np.int8)

    return df


def compute_volume_shock(
    df: pd.DataFrame,
    windows: list = None,
    lookback: int = 300,
) -> pd.DataFrame:
    """
    volume_shock = current_window_volume / rolling_mean_volume - 1

    Captures sudden spikes in trading activity relative to recent norms.
    """
    if windows is None:
        windows = ["1s", "5s", "10s"]

    for w in windows:
        col = f"total_volume_{w}"
        if col not in df.columns:
            continue
        rolling_avg = df[col].rolling(lookback, min_periods=10).mean()
        df[f"volume_shock_{w}"] = (df[col] / rolling_avg - 1).where(rolling_avg > 0, 0.0)

    return df


def compute_all_trade_features(
    df: pd.DataFrame,
    trade_windows: list = None,
) -> pd.DataFrame:
    """Compute all additional trade-flow features."""
    if trade_windows is None:
        trade_windows = ["1s", "5s", "10s"]

    df = compute_large_trade_indicator(df, windows=trade_windows)
    df = compute_volume_shock(df, windows=trade_windows)

    n_new = sum(1 for c in df.columns if c.startswith(("large_trade_", "volume_shock_")))
    logger.info("Trade features: %d new columns", n_new)
    return df
