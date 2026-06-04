"""
Model training for short-horizon alpha prediction.

Emphasis is on simple, interpretable models (per spec):
- Linear regression (baseline)
- Ridge regression (regularised)
- Lasso regression (sparse feature selection)
- Logistic regression (direction classification)

Tree-based models (RF, XGBoost, LightGBM) are optional comparisons only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.model_utils import prepare_Xy, extract_coefficients
from src.validation.metrics import evaluate_predictions
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_regression_models(alpha: float = 1.0) -> dict:
    """Return a dict of name -> sklearn Pipeline for regression."""
    return {
        "linear": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]),
        "lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha * 0.01, max_iter=5000)),
        ]),
    }


def get_classification_models() -> dict:
    """Return a dict of name -> sklearn Pipeline for classification."""
    return {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]),
    }


# ---------------------------------------------------------------------------
# Single model train + evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    model_name: str = "ridge",
    feature_set: str = "all",
    task: str = "regression",
) -> dict:
    """
    Train a single model on *train_df*, predict on *test_df*, return metrics
    plus the fitted model and coefficients.
    """
    X_train, y_train, feat_names, _ = prepare_Xy(train_df, target, feature_set)
    X_test, y_test, _, test_idx = prepare_Xy(test_df, target, feature_set)

    if len(X_train) < 30 or len(X_test) < 10:
        logger.warning("Insufficient data for %s (train=%d, test=%d)", model_name, len(X_train), len(X_test))
        return {"model_name": model_name, "target": target, "error": "insufficient_data"}

    models = get_regression_models() if task == "regression" else get_classification_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    pipe = models[model_name]
    pipe.fit(X_train, y_train)

    if task == "regression":
        y_pred = pipe.predict(X_test)
    else:
        y_pred = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X_test)

    metrics = evaluate_predictions(y_test, y_pred, task=task)

    # Extract coefficients from the inner model
    inner_model = pipe.named_steps["model"]
    coefs = extract_coefficients(inner_model, feat_names)

    return {
        "model_name": model_name,
        "target": target,
        "feature_set": feature_set,
        "task": task,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
        "coefficients": coefs,
        "pipeline": pipe,
        "predictions": pd.Series(y_pred, index=test_idx, name="y_pred"),
    }


# ---------------------------------------------------------------------------
# Run all models for one target
# ---------------------------------------------------------------------------

def run_model_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    feature_set: str = "all",
    task: str = "regression",
) -> pd.DataFrame:
    """Train every registered model and return a comparison table."""
    models = get_regression_models() if task == "regression" else get_classification_models()

    rows = []
    for name in models:
        result = train_and_evaluate(train_df, test_df, target, name, feature_set, task)
        if "error" in result:
            continue
        row = {"model": name, "target": target, "feature_set": feature_set}
        row.update(result["metrics"])
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run all targets x all models (the full grid)
# ---------------------------------------------------------------------------

def run_full_model_grid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizons: list = None,
    feature_sets: list = None,
) -> pd.DataFrame:
    """
    Train regression models for every (horizon, feature_set) combination.

    Returns a single DataFrame with one row per (model, target, feature_set).
    """
    if horizons is None:
        horizons = ["1s", "5s", "10s", "30s"]
    if feature_sets is None:
        feature_sets = ["all"]

    all_rows = []
    for h in horizons:
        target = f"y_return_{h}"
        for fs in feature_sets:
            logger.info("Training | target=%s | features=%s", target, fs)
            comparison = run_model_comparison(train_df, test_df, target, fs, task="regression")
            all_rows.append(comparison)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)
