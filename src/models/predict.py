"""
Generate predictions from trained models on new data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.model_utils import get_feature_columns
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def predict(
    pipeline,
    df: pd.DataFrame,
    feature_set: str = "all",
) -> pd.Series:
    """
    Run a fitted sklearn pipeline on *df* and return predictions indexed
    by the DataFrame's index (NaN rows excluded).
    """
    feat_cols = get_feature_columns(df, feature_set)
    sub = df[feat_cols].dropna()

    if sub.empty:
        return pd.Series(dtype=float)

    preds = pipeline.predict(sub.values.astype(np.float64))
    return pd.Series(preds, index=sub.index, name="y_pred")


def predict_proba(
    pipeline,
    df: pd.DataFrame,
    feature_set: str = "all",
) -> pd.Series:
    """Return P(class=1) for classification pipelines."""
    feat_cols = get_feature_columns(df, feature_set)
    sub = df[feat_cols].dropna()

    if sub.empty:
        return pd.Series(dtype=float)

    probs = pipeline.predict_proba(sub.values.astype(np.float64))[:, 1]
    return pd.Series(probs, index=sub.index, name="y_pred_proba")
