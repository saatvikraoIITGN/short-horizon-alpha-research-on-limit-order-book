"""
Reusable plotting utilities and style configuration.

All plots use a consistent research-paper style and save to ``reports/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.utils.paths import get_project_root, ensure_dir


def set_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def get_figures_dir() -> Path:
    return ensure_dir(get_project_root() / "reports" / "figures")


def get_tables_dir() -> Path:
    return ensure_dir(get_project_root() / "reports" / "tables")


def save_fig(fig: plt.Figure, name: str, close: bool = True) -> Path:
    """Save figure to ``reports/figures/<name>.png`` and optionally close it."""
    path = get_figures_dir() / f"{name}.png"
    fig.savefig(path)
    if close:
        plt.close(fig)
    return path


PALETTE = sns.color_palette("deep")
HORIZON_COLORS = {"1s": PALETTE[0], "5s": PALETTE[1], "10s": PALETTE[2], "30s": PALETTE[3]}
ASSET_COLORS = {"BTCUSDT": PALETTE[0], "ETHUSDT": PALETTE[1]}
