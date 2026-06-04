"""
Model utilities: feature column selection, coefficient extraction, and
common pre-processing shared across model types.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Feature groups for ablation studies
ORDERBOOK_FEATURES = [
    "obi_1", "obi_3", "obi_5", "weighted_obi", "microprice_deviation",
    "spread", "relative_spread", "bid_depth_total", "ask_depth_total",
    "total_depth", "top_bid_depth", "top_ask_depth",
    "bid_depth_slope", "ask_depth_slope",
    "depth_imbalance_1", "depth_imbalance_2", "depth_imbalance_3",
    "depth_imbalance_4", "depth_imbalance_5",
]

TRADE_FEATURES = [
    "trade_imbalance_1s", "trade_imbalance_5s", "trade_imbalance_10s",
    "signed_volume_1s", "signed_volume_5s", "signed_volume_10s",
    "total_volume_1s", "total_volume_5s", "total_volume_10s",
    "num_trades_1s", "avg_trade_size_1s",
    "large_trade_1s", "volume_shock_1s", "volume_shock_5s",
]

RETURN_VOL_FEATURES = [
    "return_lag_1s", "return_lag_5s", "return_lag_10s", "return_lag_30s",
    "realized_vol_10s", "realized_vol_30s", "realized_vol_60s",
    "zscore_return_lag_1s", "zscore_spread", "zscore_total_volume_1s",
    "zscore_obi_1",
]

ALL_FEATURES = ORDERBOOK_FEATURES + TRADE_FEATURES + RETURN_VOL_FEATURES

FEATURE_SETS = {
    "orderbook_only": ORDERBOOK_FEATURES,
    "trade_only": TRADE_FEATURES,
    "return_vol_only": RETURN_VOL_FEATURES,
    "all": ALL_FEATURES,
}


def get_feature_columns(df: pd.DataFrame, feature_set: str = "all") -> list:
    """Return the subset of feature columns that actually exist in *df*."""
    candidates = FEATURE_SETS.get(feature_set, ALL_FEATURES)
    return [c for c in candidates if c in df.columns]


def prepare_Xy(
    df: pd.DataFrame,
    target: str,
    feature_set: str = "all",
) -> tuple:
    """
    Extract aligned (X, y) arrays, dropping NaN rows.

    Returns ``(X, y, feature_names, valid_index)``.
    """
    feat_cols = get_feature_columns(df, feature_set)
    sub = df[feat_cols + [target]].dropna()

    X = sub[feat_cols].values.astype(np.float64)
    y = sub[target].values.astype(np.float64)

    return X, y, feat_cols, sub.index


def extract_coefficients(model, feature_names: list) -> pd.DataFrame:
    """Pull coefficients or feature importances from a fitted sklearn model."""
    if hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
    elif hasattr(model, "feature_importances_"):
        coefs = model.feature_importances_
    else:
        return pd.DataFrame()

    result = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    result["abs_coefficient"] = result["coefficient"].abs()
    return result.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
