"""
Evaluation metrics for signal quality, statistical prediction, and trading.

Three categories (matching the project spec):
1. **Statistical prediction** -- IC, rank IC, MSE, MAE, AUC, accuracy, F1
2. **Signal quality** -- decile spread, t-stat, IC ratio, feature stability
3. **Trading / backtest** -- gross/net Sharpe, hit rate, turnover, drawdown
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Statistical prediction metrics
# ---------------------------------------------------------------------------

def pearson_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    return np.corrcoef(y_true[mask], y_pred[mask])[0, 1]


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    return stats.spearmanr(y_true[mask], y_pred[mask]).statistic


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 10:
        return {}

    residuals = yt - yp
    return {
        "pearson_ic": np.corrcoef(yt, yp)[0, 1],
        "spearman_ic": stats.spearmanr(yt, yp).statistic,
        "mse": float(np.mean(residuals ** 2)),
        "mae": float(np.mean(np.abs(residuals))),
        "n": int(len(yt)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics for binary (0/1) classification targets."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask].astype(int), y_pred[mask]
    if len(yt) < 10:
        return {}

    yp_binary = (yp > 0.5).astype(int) if yp.max() <= 1.0 else (yp > 0).astype(int)

    result = {
        "accuracy": accuracy_score(yt, yp_binary),
        "precision": precision_score(yt, yp_binary, zero_division=0),
        "recall": recall_score(yt, yp_binary, zero_division=0),
        "f1": f1_score(yt, yp_binary, zero_division=0),
        "n": int(len(yt)),
    }

    try:
        result["auc"] = roc_auc_score(yt, yp)
    except ValueError:
        result["auc"] = np.nan

    return result


# ---------------------------------------------------------------------------
# 2. Signal-quality metrics
# ---------------------------------------------------------------------------

def decile_spread(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Mean return in top decile minus mean return in bottom decile."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < n_bins * 3:
        return np.nan

    try:
        bins = pd.qcut(yp, n_bins, labels=False, duplicates="drop")
    except ValueError:
        return np.nan

    means = pd.Series(yt).groupby(bins).mean()
    if len(means) < 2:
        return np.nan
    return float(means.iloc[-1] - means.iloc[0])


def ic_tstat(ic_series: pd.Series) -> float:
    """t-statistic of a series of per-period ICs."""
    n = ic_series.dropna().shape[0]
    if n < 3:
        return np.nan
    return float(ic_series.mean() / (ic_series.std() / np.sqrt(n)))


def ic_information_ratio(ic_series: pd.Series) -> float:
    """IC mean / IC std -- annualised stability measure."""
    s = ic_series.dropna()
    if s.std() == 0 or len(s) < 3:
        return np.nan
    return float(s.mean() / s.std())


# ---------------------------------------------------------------------------
# 3. Trading / backtest metrics (used by backtest module too)
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252 * 86400) -> float:
    """Annualised Sharpe ratio.  *periods_per_year* = seconds in a trading year for 1s data."""
    r = returns[~np.isnan(returns)]
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def hit_rate(returns: np.ndarray) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) == 0:
        return np.nan
    return float((r > 0).sum() / len(r))


def max_drawdown(cum_returns: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum_returns)
    dd = (cum_returns - peak) / np.where(peak != 0, peak, 1)
    return float(dd.min())


def turnover(positions: np.ndarray) -> float:
    """Mean absolute position change per period."""
    diffs = np.abs(np.diff(positions))
    return float(diffs.mean()) if len(diffs) > 0 else 0.0


# ---------------------------------------------------------------------------
# Combined evaluator
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
) -> dict:
    """
    One-call evaluator returning all relevant metrics.

    *task*: ``"regression"`` or ``"classification"``.
    """
    if task == "regression":
        m = regression_metrics(y_true, y_pred)
        m["decile_spread"] = decile_spread(y_true, y_pred)
        return m
    else:
        return classification_metrics(y_true, y_pred)
