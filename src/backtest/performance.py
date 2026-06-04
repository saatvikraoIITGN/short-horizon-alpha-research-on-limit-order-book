"""
Backtest performance reporting: summary tables, equity curves, cost sensitivity.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.plots import save_fig, set_style, HORIZON_COLORS
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def plot_equity_curve(
    backtest_result: dict,
    cost_scenarios: list = None,
    symbol: str = "",
) -> plt.Figure:
    """Plot cumulative PnL for gross and each net cost scenario."""
    set_style()
    if cost_scenarios is None:
        cost_scenarios = ["zero", "low", "medium", "high"]

    bt_df = backtest_result["backtest_df"]
    horizon = backtest_result["horizon"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Gross
    cum_gross = bt_df["gross_return"].cumsum()
    ax.plot(bt_df["timestamp"] if "timestamp" in bt_df.columns else cum_gross.index,
            cum_gross.values * 1e4, linewidth=2, label="Gross", color="black")

    colors = {"zero": "green", "low": "blue", "medium": "orange", "high": "red"}
    for sc in cost_scenarios:
        col = f"net_return_{sc}"
        if col in bt_df.columns:
            cum = bt_df[col].cumsum()
            ax.plot(bt_df["timestamp"] if "timestamp" in bt_df.columns else cum.index,
                    cum.values * 1e4, linewidth=1.5, label=f"Net ({sc})",
                    color=colors.get(sc, "grey"), linestyle="--")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative PnL (bps)")
    ax.set_title(f"Equity Curve — {horizon} horizon{f' — {symbol}' if symbol else ''}")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_cost_sensitivity(
    summary_df: pd.DataFrame,
    metric: str = "sharpe",
) -> plt.Figure:
    """Bar chart of Sharpe (or other metric) across cost scenarios and horizons."""
    set_style()

    horizons = summary_df["horizon"].unique()
    scenarios = summary_df["cost_scenario"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.15
    x_base = np.arange(len(horizons))

    for i, sc in enumerate(scenarios):
        sub = summary_df[summary_df["cost_scenario"] == sc]
        vals = []
        for h in horizons:
            row = sub[sub["horizon"] == h]
            vals.append(row[metric].values[0] if len(row) > 0 else 0)
        ax.bar(x_base + i * width, vals, width, label=sc)

    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Cost Sensitivity: {metric.replace('_', ' ').title()} by Scenario")
    ax.set_xticks(x_base + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(horizons)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.legend(title="Cost scenario")
    fig.tight_layout()
    return fig


def format_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Format the summary DataFrame for display / CSV export."""
    display_cols = [
        "horizon", "cost_scenario", "total_pnl", "mean_return_per_trade",
        "sharpe", "hit_rate", "max_drawdown", "n_trades",
    ]
    cols = [c for c in display_cols if c in summary_df.columns]
    out = summary_df[cols].copy()

    for c in ["total_pnl", "mean_return_per_trade"]:
        if c in out.columns:
            out[c] = (out[c] * 1e4).round(2)  # convert to bps

    for c in ["sharpe"]:
        if c in out.columns:
            out[c] = out[c].round(2)

    for c in ["hit_rate"]:
        if c in out.columns:
            out[c] = (out[c] * 100).round(1)  # percent

    for c in ["max_drawdown"]:
        if c in out.columns:
            out[c] = (out[c] * 100).round(2)

    rename = {
        "total_pnl": "PnL (bps)",
        "mean_return_per_trade": "Ret/Trade (bps)",
        "sharpe": "Sharpe",
        "hit_rate": "Hit %",
        "max_drawdown": "MaxDD %",
        "n_trades": "Trades",
    }
    out.rename(columns=rename, inplace=True)
    return out
