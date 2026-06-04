"""
Exploratory Data Analysis -- the 10 analyses required by the project spec.

1.  Distribution of spreads
2.  Distribution of returns by horizon
3.  Distribution of order book imbalance
4.  Average future return by imbalance decile
5.  Average future return by trade imbalance decile
6.  Correlation matrix of major features
7.  Signal decay (delegated to signal_decay.py)
8.  Volatility and volume by time of day
9.  Spread and depth by time of day
10. BTCUSDT vs ETHUSDT comparison
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.plots import (
    ASSET_COLORS,
    HORIZON_COLORS,
    save_fig,
    set_style,
    get_tables_dir,
)
from src.analysis.signal_decay import plot_signal_decay, compute_ic_table
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


# ---- helpers ---------------------------------------------------------------

def _decile_return(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin *feature* into quantile deciles and compute mean *target* per bin."""
    tmp = df[[feature, target]].dropna()
    if len(tmp) < n_bins * 5:
        return pd.DataFrame()
    tmp["decile"] = pd.qcut(tmp[feature], n_bins, labels=False, duplicates="drop") + 1
    return tmp.groupby("decile")[target].agg(["mean", "std", "count"]).reset_index()


# ---- 1. Spread distribution -----------------------------------------------

def plot_spread_distribution(df: pd.DataFrame, symbol: str = "") -> plt.Figure:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df["spread"].dropna(), bins=80, edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Absolute spread")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Spread distribution")

    axes[1].hist(df["relative_spread"].dropna() * 1e4, bins=80, edgecolor="white", alpha=0.8, color="orange")
    axes[1].set_xlabel("Relative spread (bps)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Relative spread distribution")

    fig.suptitle(f"Spread Distributions{f' — {symbol}' if symbol else ''}", fontsize=13)
    fig.tight_layout()
    return fig


# ---- 2. Return distributions by horizon -----------------------------------

def plot_return_distributions(df: pd.DataFrame, symbol: str = "") -> plt.Figure:
    set_style()
    horizons = ["1s", "5s", "10s", "30s"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for h in horizons:
        col = f"y_return_{h}"
        if col not in df.columns:
            continue
        vals = df[col].dropna() * 1e4  # convert to bps
        ax.hist(vals, bins=100, alpha=0.45, label=h, color=HORIZON_COLORS.get(h))

    ax.set_xlabel("Future return (bps)")
    ax.set_ylabel("Count")
    ax.set_title(f"Return Distributions by Horizon{f' — {symbol}' if symbol else ''}")
    ax.legend(title="Horizon")
    return fig


# ---- 3. OBI distribution --------------------------------------------------

def plot_obi_distribution(df: pd.DataFrame, symbol: str = "") -> plt.Figure:
    set_style()
    obi_cols = [c for c in ["obi_1", "obi_3", "obi_5"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(10, 6))

    for col in obi_cols:
        ax.hist(df[col].dropna(), bins=80, alpha=0.45, label=col)

    ax.set_xlabel("Order Book Imbalance")
    ax.set_ylabel("Count")
    ax.set_title(f"OBI Distributions{f' — {symbol}' if symbol else ''}")
    ax.legend()
    return fig


# ---- 4. Average return by imbalance decile --------------------------------

def plot_return_by_obi_decile(
    df: pd.DataFrame,
    feature: str = "obi_1",
    horizons: list = None,
    symbol: str = "",
) -> plt.Figure:
    set_style()
    if horizons is None:
        horizons = ["1s", "5s", "10s", "30s"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    for i, h in enumerate(horizons[:4]):
        ax = axes[i]
        target = f"y_return_{h}"
        dec = _decile_return(df, feature, target)
        if dec.empty:
            ax.set_title(f"{h} (insufficient data)")
            continue
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in dec["mean"]]
        ax.bar(dec["decile"], dec["mean"] * 1e4, color=colors, edgecolor="white")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"{feature} decile")
        ax.set_ylabel("Mean return (bps)")
        ax.set_title(f"{h} horizon")

    fig.suptitle(
        f"Average Future Return by {feature} Decile{f' — {symbol}' if symbol else ''}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


# ---- 5. Average return by trade imbalance decile --------------------------

def plot_return_by_trade_imbalance_decile(
    df: pd.DataFrame,
    horizons: list = None,
    symbol: str = "",
) -> plt.Figure:
    return plot_return_by_obi_decile(
        df, feature="trade_imbalance_1s", horizons=horizons, symbol=symbol,
    )


# ---- 6. Correlation matrix ------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, symbol: str = "") -> plt.Figure:
    set_style()
    feature_cols = [
        "obi_1", "obi_3", "obi_5", "weighted_obi", "microprice_deviation",
        "spread", "relative_spread", "bid_depth_total", "ask_depth_total",
        "trade_imbalance_1s", "signed_volume_1s", "total_volume_1s",
        "return_lag_1s", "realized_vol_10s",
        "y_return_1s", "y_return_5s", "y_return_10s",
    ]
    cols = [c for c in feature_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, center=0, ax=ax, square=True,
        linewidths=0.5, annot_kws={"size": 8},
    )
    ax.set_title(f"Feature Correlation Matrix{f' — {symbol}' if symbol else ''}")
    fig.tight_layout()
    return fig


# ---- 8. Volatility & volume by time of day --------------------------------

def plot_volume_volatility_by_hour(df: pd.DataFrame, symbol: str = "") -> plt.Figure:
    set_style()
    if "hour" not in df.columns or df["hour"].nunique() < 2:
        fig, ax = plt.subplots()
        ax.set_title("Volume & Volatility by Hour (need multi-hour data)")
        return fig

    hourly = df.groupby("hour").agg(
        mean_volume=("total_volume_1s", "mean"),
        mean_vol=("realized_vol_10s", "mean"),
    ).dropna()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(hourly.index, hourly["mean_volume"], alpha=0.5, label="Mean 1s volume", color="steelblue")
    ax1.set_xlabel("Hour (UTC)")
    ax1.set_ylabel("Mean volume (1s window)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly["mean_vol"], "o-", color="tomato", label="Realized vol (10s)")
    ax2.set_ylabel("Realized volatility (10s)", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")

    fig.suptitle(f"Volume & Volatility by Hour{f' — {symbol}' if symbol else ''}", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    return fig


# ---- 9. Spread & depth by time of day -------------------------------------

def plot_spread_depth_by_hour(df: pd.DataFrame, symbol: str = "") -> plt.Figure:
    set_style()
    if "hour" not in df.columns or df["hour"].nunique() < 2:
        fig, ax = plt.subplots()
        ax.set_title("Spread & Depth by Hour (need multi-hour data)")
        return fig

    hourly = df.groupby("hour").agg(
        mean_spread=("spread", "mean"),
        mean_depth=("total_depth", "mean"),
    ).dropna()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(hourly.index, hourly["mean_depth"], alpha=0.5, label="Total depth (top-5)", color="teal")
    ax1.set_xlabel("Hour (UTC)")
    ax1.set_ylabel("Mean total depth", color="teal")
    ax1.tick_params(axis="y", labelcolor="teal")

    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly["mean_spread"], "s-", color="purple", label="Mean spread")
    ax2.set_ylabel("Mean spread", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    fig.suptitle(f"Spread & Depth by Hour{f' — {symbol}' if symbol else ''}", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    return fig


# ---- 10. BTC vs ETH comparison -------------------------------------------

def plot_asset_comparison(
    datasets: dict,
    metric_fn=None,
    title: str = "Asset Comparison",
) -> plt.Figure:
    """
    Side-by-side box/bar comparison of key statistics across assets.

    *datasets*: ``{"BTCUSDT": df_btc, "ETHUSDT": df_eth}``
    """
    set_style()
    metrics = ["obi_1", "spread", "trade_imbalance_1s", "return_lag_1s", "realized_vol_10s"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        data, labels = [], []
        for sym, df in datasets.items():
            if m in df.columns:
                data.append(df[m].dropna().values)
                labels.append(sym)
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
            for patch, sym in zip(bp["boxes"], labels):
                patch.set_facecolor(ASSET_COLORS.get(sym, "grey"))
                patch.set_alpha(0.6)
        ax.set_title(m, fontsize=10)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ---- Summary statistics table ---------------------------------------------

def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive stats for key features, suitable for the research memo."""
    cols = [
        "spread", "relative_spread", "obi_1", "obi_5", "weighted_obi",
        "microprice_deviation", "trade_imbalance_1s", "total_volume_1s",
        "return_lag_1s", "realized_vol_10s",
        "y_return_1s", "y_return_5s", "y_return_10s", "y_return_30s",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].describe().T


# ---- Full EDA runner -------------------------------------------------------

def run_full_eda(
    datasets: dict,
    horizons: list = None,
) -> list:
    """
    Run all 10 EDA analyses and save figures.

    Parameters
    ----------
    datasets : ``{"BTCUSDT": featured_df, "ETHUSDT": featured_df}``
    horizons : prediction horizons list.

    Returns list of saved file paths.
    """
    if horizons is None:
        horizons = ["1s", "5s", "10s", "30s"]

    saved = []

    for sym, df in datasets.items():
        tag = sym.lower()
        logger.info("EDA for %s  (%d rows)", sym, len(df))

        # 1. Spread
        fig = plot_spread_distribution(df, sym)
        saved.append(save_fig(fig, f"01_spread_dist_{tag}"))

        # 2. Returns
        fig = plot_return_distributions(df, sym)
        saved.append(save_fig(fig, f"02_return_dist_{tag}"))

        # 3. OBI
        fig = plot_obi_distribution(df, sym)
        saved.append(save_fig(fig, f"03_obi_dist_{tag}"))

        # 4. Return by OBI decile
        fig = plot_return_by_obi_decile(df, "obi_1", horizons, sym)
        saved.append(save_fig(fig, f"04_return_by_obi_decile_{tag}"))

        # 5. Return by trade imbalance decile
        fig = plot_return_by_trade_imbalance_decile(df, horizons, sym)
        saved.append(save_fig(fig, f"05_return_by_trade_imb_decile_{tag}"))

        # 6. Correlation matrix
        fig = plot_correlation_matrix(df, sym)
        saved.append(save_fig(fig, f"06_correlation_matrix_{tag}"))

        # 7. Signal decay
        fig = plot_signal_decay(df, title_suffix=f" — {sym}")
        saved.append(save_fig(fig, f"07_signal_decay_{tag}"))

        # 8. Volume & volatility by hour
        fig = plot_volume_volatility_by_hour(df, sym)
        saved.append(save_fig(fig, f"08_vol_volatility_by_hour_{tag}"))

        # 9. Spread & depth by hour
        fig = plot_spread_depth_by_hour(df, sym)
        saved.append(save_fig(fig, f"09_spread_depth_by_hour_{tag}"))

        # Summary table
        summary = compute_summary_table(df)
        tbl_path = get_tables_dir() / f"summary_stats_{tag}.csv"
        summary.to_csv(tbl_path)
        logger.info("  Summary table -> %s", tbl_path)

    # 10. Cross-asset comparison
    if len(datasets) >= 2:
        fig = plot_asset_comparison(datasets, title="BTCUSDT vs ETHUSDT")
        saved.append(save_fig(fig, "10_btc_vs_eth_comparison"))

    logger.info("EDA complete — %d figures saved", len(saved))
    return saved
