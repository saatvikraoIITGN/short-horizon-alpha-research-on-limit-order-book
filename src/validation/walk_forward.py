"""
Chronological validation: fixed split and walk-forward rolling.

Key principle: **never** use random train-test splits for time-series alpha
research.  All splits respect temporal ordering to prevent lookahead bias.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.train_model import train_and_evaluate, run_model_comparison
from src.validation.metrics import (
    pearson_ic, spearman_ic, decile_spread, ic_tstat, ic_information_ratio,
)
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Fixed chronological split (60 / 20 / 20)
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple:
    """
    Split a time-sorted DataFrame into train / validation / test by row count.

    Returns ``(train_df, val_df, test_df)``.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    i_val = int(n * train_frac)
    i_test = int(n * (train_frac + val_frac))

    train = df.iloc[:i_val].copy()
    val = df.iloc[i_val:i_test].copy()
    test = df.iloc[i_test:].copy()

    logger.info(
        "Chronological split: train=%d [%s..%s]  val=%d [%s..%s]  test=%d [%s..%s]",
        len(train), train["timestamp"].iloc[0], train["timestamp"].iloc[-1],
        len(val), val["timestamp"].iloc[0], val["timestamp"].iloc[-1],
        len(test), test["timestamp"].iloc[0], test["timestamp"].iloc[-1],
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_split(
    df: pd.DataFrame,
    train_seconds: int = 5 * 86400,
    val_seconds: int = 86400,
    test_seconds: int = 86400,
    step_seconds: Optional[int] = None,
) -> list:
    """
    Generate walk-forward (train, val, test) index triples.

    With default settings at 1s frequency:
      - train on 5 days, validate on day 6, test on day 7
      - roll forward by *step_seconds* (default = test_seconds)

    Returns list of ``(train_df, val_df, test_df)`` tuples.
    """
    if step_seconds is None:
        step_seconds = test_seconds

    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]
    t0 = ts.iloc[0]
    t_end = ts.iloc[-1]

    folds = []
    cursor = t0

    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(seconds=train_seconds)
        val_end = train_end + pd.Timedelta(seconds=val_seconds)
        test_end = val_end + pd.Timedelta(seconds=test_seconds)

        if test_end > t_end:
            break

        train_mask = (ts >= train_start) & (ts < train_end)
        val_mask = (ts >= train_end) & (ts < val_end)
        test_mask = (ts >= val_end) & (ts < test_end)

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) > 50 and len(val_df) > 10 and len(test_df) > 10:
            folds.append((train_df, val_df, test_df))

        cursor += pd.Timedelta(seconds=step_seconds)

    logger.info("Walk-forward: %d folds generated", len(folds))
    return folds


def walk_forward_splits_adaptive(
    df: pd.DataFrame,
    n_folds: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> list:
    """
    Adaptive walk-forward that divides available data into *n_folds*
    expanding-window folds.  Works well when total data is small.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    test_ratio = 1.0 - train_ratio - val_ratio
    fold_size = int(n * test_ratio / max(n_folds, 1))

    if fold_size < 20:
        fold_size = max(20, n // (n_folds + 2))

    folds = []
    for i in range(n_folds):
        test_end = n - i * fold_size
        test_start = test_end - fold_size
        if test_start <= 0:
            break

        remaining = test_start
        val_size = max(10, int(remaining * val_ratio / (train_ratio + val_ratio)))
        val_start = test_start - val_size
        train_end = val_start

        if train_end < 30:
            continue

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:test_start].copy()
        test_df = df.iloc[test_start:test_end].copy()

        folds.append((train_df, val_df, test_df))

    folds.reverse()
    logger.info("Adaptive walk-forward: %d folds (fold_size=%d)", len(folds), fold_size)
    return folds


# ---------------------------------------------------------------------------
# Walk-forward evaluator
# ---------------------------------------------------------------------------

def run_walk_forward(
    df: pd.DataFrame,
    target: str,
    model_name: str = "ridge",
    feature_set: str = "all",
    task: str = "regression",
    n_folds: int = 5,
) -> dict:
    """
    Run walk-forward validation and return fold-level + aggregate metrics.

    Uses adaptive splits that work on both small and large datasets.
    """
    folds = walk_forward_splits_adaptive(df, n_folds=n_folds)

    if not folds:
        logger.warning("No valid folds for walk-forward")
        return {"fold_metrics": pd.DataFrame(), "aggregate": {}}

    fold_rows = []
    all_preds = []
    all_actuals = []

    for i, (train_df, val_df, test_df) in enumerate(folds):
        result = train_and_evaluate(
            train_df, test_df, target, model_name, feature_set, task,
        )

        if "error" in result:
            continue

        row = {"fold": i, "train_size": result["train_size"], "test_size": result["test_size"]}
        row.update(result["metrics"])
        fold_rows.append(row)

        if "predictions" in result:
            preds = result["predictions"]
            actuals = test_df.loc[preds.index, target]
            all_preds.append(preds)
            all_actuals.append(actuals)

    fold_df = pd.DataFrame(fold_rows)

    # Aggregate across folds
    agg = {}
    if not fold_df.empty:
        for col in ["pearson_ic", "spearman_ic", "mse", "mae", "decile_spread"]:
            if col in fold_df.columns:
                series = fold_df[col].dropna()
                agg[f"mean_{col}"] = float(series.mean())
                agg[f"std_{col}"] = float(series.std())

        if "pearson_ic" in fold_df.columns:
            agg["ic_tstat"] = ic_tstat(fold_df["pearson_ic"])
            agg["ic_ir"] = ic_information_ratio(fold_df["pearson_ic"])

        agg["n_folds"] = len(fold_df)
        agg["total_test_rows"] = int(fold_df["test_size"].sum())

    logger.info(
        "Walk-forward %s/%s: %d folds  mean_IC=%.4f  IC_tstat=%.2f",
        model_name, target, agg.get("n_folds", 0),
        agg.get("mean_pearson_ic", 0), agg.get("ic_tstat", 0),
    )

    return {
        "fold_metrics": fold_df,
        "aggregate": agg,
        "model_name": model_name,
        "target": target,
        "feature_set": feature_set,
    }


# ---------------------------------------------------------------------------
# Full walk-forward grid
# ---------------------------------------------------------------------------

def run_walk_forward_grid(
    df: pd.DataFrame,
    horizons: list = None,
    model_names: list = None,
    feature_sets: list = None,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Run walk-forward for every (model, horizon, feature_set) and return
    a summary DataFrame with one row per configuration.
    """
    if horizons is None:
        horizons = ["1s", "5s", "10s", "30s"]
    if model_names is None:
        model_names = ["linear", "ridge", "lasso"]
    if feature_sets is None:
        feature_sets = ["all"]

    rows = []
    for h in horizons:
        target = f"y_return_{h}"
        for fs in feature_sets:
            for mn in model_names:
                result = run_walk_forward(df, target, mn, fs, n_folds=n_folds)
                agg = result["aggregate"]
                if agg:
                    row = {"model": mn, "horizon": h, "target": target, "feature_set": fs}
                    row.update(agg)
                    rows.append(row)

    return pd.DataFrame(rows)
