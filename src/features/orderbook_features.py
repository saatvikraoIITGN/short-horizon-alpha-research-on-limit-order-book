"""
Order-book features derived from L2 snapshot data.

Every feature is interpretable and rooted in market-microstructure intuition:
- **Imbalance** signals excess demand / supply pressure.
- **Microprice** is a size-weighted fair-value estimator.
- **Spread / depth** capture liquidity conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Mid-price & spread
# ---------------------------------------------------------------------------

def compute_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
    return df


def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute and relative spread."""
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["relative_spread"] = df["spread"] / df["mid_price"]
    return df


# ---------------------------------------------------------------------------
# Order-book imbalance (OBI)
# ---------------------------------------------------------------------------

def compute_obi(df: pd.DataFrame, levels: list = None) -> pd.DataFrame:
    """
    OBI_N = (sum bid_size 1..N  -  sum ask_size 1..N)
          / (sum bid_size 1..N  +  sum ask_size 1..N)

    Positive -> more resting buy interest -> short-term upward pressure.
    """
    if levels is None:
        levels = [1, 3, 5]

    for n in levels:
        bid_cols = [f"bid_size_{i}" for i in range(1, n + 1) if f"bid_size_{i}" in df.columns]
        ask_cols = [f"ask_size_{i}" for i in range(1, n + 1) if f"ask_size_{i}" in df.columns]
        if not bid_cols or not ask_cols:
            continue
        bid_sum = df[bid_cols].sum(axis=1)
        ask_sum = df[ask_cols].sum(axis=1)
        total = bid_sum + ask_sum
        df[f"obi_{n}"] = ((bid_sum - ask_sum) / total).where(total > 0, 0.0)

    return df


# ---------------------------------------------------------------------------
# Weighted OBI  (weight_i = 1/i)
# ---------------------------------------------------------------------------

def compute_weighted_obi(df: pd.DataFrame, max_level: int = 5) -> pd.DataFrame:
    """
    Gives higher weight to levels closer to the BBO, reflecting that
    top-of-book orders are more likely to execute and thus more informative.
    """
    w_bid = pd.Series(0.0, index=df.index)
    w_ask = pd.Series(0.0, index=df.index)
    for i in range(1, max_level + 1):
        w = 1.0 / i
        bc, ac = f"bid_size_{i}", f"ask_size_{i}"
        if bc in df.columns:
            w_bid += w * df[bc]
        if ac in df.columns:
            w_ask += w * df[ac]
    total = w_bid + w_ask
    df["weighted_obi"] = ((w_bid - w_ask) / total).where(total > 0, 0.0)
    return df


# ---------------------------------------------------------------------------
# Microprice
# ---------------------------------------------------------------------------

def compute_microprice(df: pd.DataFrame) -> pd.DataFrame:
    """
    microprice = (ask_price * bid_size + bid_price * ask_size)
               / (bid_size + ask_size)

    A size-weighted mid-price that tilts towards the side with more
    resting liquidity, serving as a better fair-value estimator.
    """
    total_size = df["bid_size_1"] + df["ask_size_1"]
    df["microprice"] = (
        (df["ask_price_1"] * df["bid_size_1"] + df["bid_price_1"] * df["ask_size_1"])
        / total_size
    ).where(total_size > 0, df["mid_price"])
    df["microprice_deviation"] = (df["microprice"] - df["mid_price"]) / df["mid_price"]
    return df


# ---------------------------------------------------------------------------
# Depth & liquidity
# ---------------------------------------------------------------------------

def compute_depth_features(df: pd.DataFrame, max_level: int = 5) -> pd.DataFrame:
    """
    Aggregate depth, top-of-book depth, depth slope, and per-level imbalance.
    """
    bid_cols = [f"bid_size_{i}" for i in range(1, max_level + 1) if f"bid_size_{i}" in df.columns]
    ask_cols = [f"ask_size_{i}" for i in range(1, max_level + 1) if f"ask_size_{i}" in df.columns]
    n_levels = min(len(bid_cols), len(ask_cols))

    df["bid_depth_total"] = df[bid_cols].sum(axis=1)
    df["ask_depth_total"] = df[ask_cols].sum(axis=1)
    df["total_depth"] = df["bid_depth_total"] + df["ask_depth_total"]

    df["top_bid_depth"] = df["bid_size_1"]
    df["top_ask_depth"] = df["ask_size_1"]

    # Depth slope: how fast size drops off from level 1 to level N
    if n_levels > 1:
        df["bid_depth_slope"] = (df[bid_cols[0]] - df[bid_cols[-1]]) / (n_levels - 1)
        df["ask_depth_slope"] = (df[ask_cols[0]] - df[ask_cols[-1]]) / (n_levels - 1)

    # Per-level imbalance
    for i in range(1, n_levels + 1):
        bc, ac = f"bid_size_{i}", f"ask_size_{i}"
        total = df[bc] + df[ac]
        df[f"depth_imbalance_{i}"] = ((df[bc] - df[ac]) / total).where(total > 0, 0.0)

    return df


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def compute_all_orderbook_features(
    df: pd.DataFrame,
    imbalance_levels: list = None,
    book_levels: int = 5,
) -> pd.DataFrame:
    """Compute every order-book feature in one call."""
    if imbalance_levels is None:
        imbalance_levels = [1, 3, 5]

    df = compute_mid_price(df)
    df = compute_spread_features(df)
    df = compute_obi(df, levels=imbalance_levels)
    df = compute_weighted_obi(df, max_level=book_levels)
    df = compute_microprice(df)
    df = compute_depth_features(df, max_level=book_levels)

    logger.info("Order-book features: %d new columns", sum(1 for c in df.columns if c.startswith((
        "mid_price", "spread", "relative_spread", "obi_", "weighted_obi",
        "microprice", "bid_depth", "ask_depth", "total_depth",
        "top_bid", "top_ask", "depth_imbalance",
    ))))
    return df
