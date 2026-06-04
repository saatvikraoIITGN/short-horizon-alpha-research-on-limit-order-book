"""
Robustness analysis: the 10 checks required by the project spec.

11. Performance by asset (BTC vs ETH)
12. Performance by horizon (1s, 5s, 10s, 30s)
13. Performance by volatility regime
14. Performance by spread regime
15. Performance by liquidity regime
16. Performance by time of day
17. Performance across walk-forward folds
18. Sensitivity to transaction costs
19. Feature ablation study
20. Model comparison
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.plots import save_fig, set_style
from src.validation.metrics import pearson_ic, spearman_ic
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def regime_analysis(
    df: pd.DataFrame,
    predictions: pd.Series,
    target: str,
    regime_col: str,
    regime_name: str = "",
) -> pd.DataFrame:
    """
    Compute IC and other metrics within each regime bucket.

    *regime_col* should already exist in *df* as a categorical / discrete column.
    """
    common = df.index.intersection(predictions.index)
    sub = df.loc[common].copy()
    sub["y_pred"] = predictions.loc[common]
    sub = sub.dropna(subset=[target, "y_pred", regime_col])

    rows = []
    for regime, grp in sub.groupby(regime_col):
        if len(grp) < 30:
            continue
        yt = grp[target].values
        yp = grp["y_pred"].values
        rows.append({
            "regime": regime,
            "n": len(grp),
            "pearson_ic": pearson_ic(yt, yp),
            "spearman_ic": spearman_ic(yt, yp),
            "mean_return": float(yt.mean()),
            "std_return": float(yt.std()),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        logger.info("Regime analysis [%s]: %d regimes", regime_name or regime_col, len(result))
    return result


def _safe_qcut(series: pd.Series, q: int, labels: list) -> pd.Series:
    """qcut that gracefully handles too-few unique values."""
    try:
        return pd.qcut(series, q, labels=labels, duplicates="drop")
    except ValueError:
        try:
            n_unique = series.nunique()
            if n_unique < q:
                return pd.qcut(series.rank(method="first"), min(n_unique, q),
                                labels=labels[:min(n_unique, q)], duplicates="drop")
        except (ValueError, TypeError):
            pass
        return pd.Series(pd.NA, index=series.index)


def add_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility, spread, and liquidity regime labels to *df*."""
    out = df.copy()

    if "realized_vol_30s" in out.columns:
        out["vol_regime"] = _safe_qcut(out["realized_vol_30s"], 3, ["low_vol", "mid_vol", "high_vol"])

    if "spread" in out.columns:
        out["spread_regime"] = _safe_qcut(out["spread"], 3, ["tight", "mid", "wide"])

    if "total_depth" in out.columns:
        out["liq_regime"] = _safe_qcut(out["total_depth"], 3, ["thin", "mid", "deep"])

    return out


def run_robustness_suite(
    df: pd.DataFrame,
    predictions: pd.Series,
    target: str = "y_return_5s",
) -> dict:
    """
    Run all regime-based robustness checks and return a dict of DataFrames.
    """
    df = add_regime_columns(df)
    results = {}

    for regime_col, name in [
        ("vol_regime", "Volatility"),
        ("spread_regime", "Spread"),
        ("liq_regime", "Liquidity"),
        ("hour", "Time of Day"),
    ]:
        if regime_col in df.columns:
            results[name] = regime_analysis(df, predictions, target, regime_col, name)

    return results


def plot_regime_ic(
    regime_results: dict,
    title_suffix: str = "",
) -> plt.Figure:
    """Bar plots of Spearman IC by regime for each regime type."""
    set_style()
    regimes = {k: v for k, v in regime_results.items() if not v.empty}
    n = len(regimes)
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("No regime data")
        return fig

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, rdf) in zip(axes, regimes.items()):
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in rdf["spearman_ic"]]
        ax.bar(rdf["regime"].astype(str), rdf["spearman_ic"], color=colors, edgecolor="white")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name} Regime")
        ax.set_ylabel("Spearman IC")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"IC by Regime{title_suffix}", fontsize=13)
    fig.tight_layout()
    return fig
