"""
Prediction-target construction.

All targets are **forward-looking** and must never be used as features.

Regression targets:
    y_return_<h>   = log(mid[t+h] / mid[t])

Classification targets:
    y_direction_<h>    = 1 if return > 0, else 0
    y_half_spread_<h>  = +1 / 0 / -1  based on half-spread threshold
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def compute_future_returns(
    df: pd.DataFrame,
    horizons: list = None,
    frequency: str = "1s",
) -> pd.DataFrame:
    """
    Compute forward log-return and price-change for each horizon.

    Shift is computed as ``horizon / frequency`` rows forward.
    The last ``shift`` rows of each target will be NaN (no future data).
    """
    if horizons is None:
        horizons = ["1s", "5s", "10s", "30s"]

    freq_td = pd.Timedelta(frequency)

    for h in horizons:
        shift = int(pd.Timedelta(h) / freq_td)
        future_mid = df["mid_price"].shift(-shift)

        df[f"y_return_{h}"] = np.log(future_mid / df["mid_price"])
        df[f"y_price_change_{h}"] = future_mid - df["mid_price"]

    return df


def compute_direction_targets(
    df: pd.DataFrame,
    horizons: list = None,
) -> pd.DataFrame:
    """Binary direction: 1 if future return > 0, else 0."""
    if horizons is None:
        horizons = ["1s", "5s", "10s", "30s"]

    for h in horizons:
        ret = f"y_return_{h}"
        if ret in df.columns:
            df[f"y_direction_{h}"] = (df[ret] > 0).astype("Int8")
            df.loc[df[ret].isna(), f"y_direction_{h}"] = pd.NA

    return df


def compute_half_spread_targets(
    df: pd.DataFrame,
    horizons: list = None,
) -> pd.DataFrame:
    """
    Three-class target that filters out economically trivial moves.

    +1  if price_change > 0.5 * spread   (meaningful up move)
     0  if |price_change| <= 0.5 * spread (noise / scratch)
    -1  if price_change < -0.5 * spread   (meaningful down move)
    """
    if horizons is None:
        horizons = ["1s", "5s", "10s", "30s"]

    for h in horizons:
        change = f"y_price_change_{h}"
        if change not in df.columns:
            continue

        threshold = 0.5 * df["spread"]
        label = pd.Series(np.int8(0), index=df.index)
        label = label.where(~(df[change] > threshold), np.int8(1))
        label = label.where(~(df[change] < -threshold), np.int8(-1))
        label = label.where(df[change].notna(), pd.NA)

        df[f"y_half_spread_{h}"] = label

    return df


def compute_all_targets(
    df: pd.DataFrame,
    horizons: list = None,
    frequency: str = "1s",
) -> pd.DataFrame:
    """Compute every prediction target in one call."""
    if horizons is None:
        horizons = ["1s", "5s", "10s", "30s"]

    df = compute_future_returns(df, horizons, frequency)
    df = compute_direction_targets(df, horizons)
    df = compute_half_spread_targets(df, horizons)

    target_cols = [c for c in df.columns if c.startswith("y_")]
    logger.info("Targets: %d columns for %d horizons", len(target_cols), len(horizons))
    return df
