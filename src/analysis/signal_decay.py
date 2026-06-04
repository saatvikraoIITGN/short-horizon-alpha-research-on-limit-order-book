"""
Signal-decay analysis: how quickly does feature predictive power fade
as the prediction horizon increases?

Primary metric: **Information Coefficient (IC)** -- Pearson and Spearman
correlation between a feature and forward returns at each horizon.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.plots import HORIZON_COLORS, save_fig, set_style
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def compute_ic_table(
    df: pd.DataFrame,
    features: list,
    horizons: list = None,
) -> pd.DataFrame:
    """
    Compute Pearson IC and Spearman rank-IC for each (feature, horizon) pair.

    Returns a DataFrame with columns:
    feature, horizon, pearson_ic, spearman_ic, pearson_pval, spearman_pval, n
    """
    if horizons is None:
        horizons = ["1s", "5s", "10s", "30s"]

    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        for h in horizons:
            target = f"y_return_{h}"
            if target not in df.columns:
                continue

            mask = df[[feat, target]].dropna().index
            x = df.loc[mask, feat]
            y = df.loc[mask, target]

            if len(x) < 30:
                continue

            pr, p_pval = stats.pearsonr(x, y)
            sr, s_pval = stats.spearmanr(x, y)

            rows.append({
                "feature": feat,
                "horizon": h,
                "pearson_ic": pr,
                "spearman_ic": sr,
                "pearson_pval": p_pval,
                "spearman_pval": s_pval,
                "n": len(x),
            })

    return pd.DataFrame(rows)


def plot_signal_decay(
    df: pd.DataFrame,
    features: list = None,
    horizons: list = None,
    ic_type: str = "spearman_ic",
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plot IC (y-axis) vs horizon (x-axis) for each feature.

    This is the canonical signal-decay chart: strong signals at short
    horizons that fade toward zero indicate genuine short-lived alpha.
    """
    set_style()

    if features is None:
        features = ["obi_1", "obi_5", "weighted_obi", "microprice_deviation",
                     "trade_imbalance_1s", "signed_volume_1s"]
    if horizons is None:
        horizons = ["1s", "5s", "10s", "30s"]

    ic_table = compute_ic_table(df, features, horizons)

    if ic_table.empty:
        logger.warning("No IC values computed -- not enough data?")
        fig, ax = plt.subplots()
        ax.set_title("Signal Decay (insufficient data)")
        return fig

    horizon_secs = [pd.Timedelta(h).total_seconds() for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 6))

    for feat in features:
        sub = ic_table[ic_table["feature"] == feat]
        if sub.empty:
            continue
        x = [pd.Timedelta(h).total_seconds() for h in sub["horizon"]]
        y = sub[ic_type].values
        ax.plot(x, y, "o-", label=feat, linewidth=2, markersize=6)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Prediction horizon (seconds)")
    ax.set_ylabel(ic_type.replace("_", " ").title())
    ax.set_title(f"Signal Decay: IC vs Horizon{title_suffix}")
    ax.set_xticks(horizon_secs)
    ax.set_xticklabels(horizons)
    ax.legend(loc="best", framealpha=0.9)

    return fig


def plot_ic_barplot(
    df: pd.DataFrame,
    features: list = None,
    horizon: str = "5s",
) -> plt.Figure:
    """Bar chart of Spearman IC for all features at a single horizon."""
    set_style()

    if features is None:
        features = [
            "obi_1", "obi_3", "obi_5", "weighted_obi",
            "microprice_deviation", "trade_imbalance_1s", "trade_imbalance_5s",
            "signed_volume_1s", "return_lag_1s", "realized_vol_10s",
        ]

    ic_table = compute_ic_table(df, features, [horizon])
    if ic_table.empty:
        fig, ax = plt.subplots()
        ax.set_title(f"Feature IC @ {horizon} (insufficient data)")
        return fig

    ic_table = ic_table.sort_values("spearman_ic", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in ic_table["spearman_ic"]]
    ax.barh(ic_table["feature"], ic_table["spearman_ic"], color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Spearman Rank IC")
    ax.set_title(f"Feature Rank IC at {horizon} horizon")

    return fig
