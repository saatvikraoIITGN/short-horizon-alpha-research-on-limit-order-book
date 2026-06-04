"""
Resample order-book snapshots to regular intervals and aggregate trades into
rolling windows.

Primary frequency: 1 s  (optional: 100 ms, 5 s).

For each resampled timestamp:
- **Order book**: latest available snapshot at or before the timestamp (forward fill).
- **Trades**: aggregated into rolling windows of 1 s, 5 s, and 10 s ending at
  each timestamp using vectorised rolling sums (no Python-level row loop).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Order-book resampling
# ---------------------------------------------------------------------------

def resample_orderbook(df: pd.DataFrame, frequency: str = "1s") -> pd.DataFrame:
    """Forward-fill order-book snapshots onto a regular time grid."""
    df = df.copy().set_index("timestamp").sort_index()

    start = df.index.min().ceil(frequency)
    end = df.index.max().floor(frequency)
    grid = pd.date_range(start=start, end=end, freq=frequency)

    resampled = df.reindex(grid, method="ffill")
    resampled.index.name = "timestamp"
    resampled.reset_index(inplace=True)

    resampled.dropna(subset=["bid_price_1"], inplace=True)
    logger.info("Resampled OB: %d -> %d rows @ %s", len(df), len(resampled), frequency)
    return resampled


# ---------------------------------------------------------------------------
# Trade aggregation (vectorised)
# ---------------------------------------------------------------------------

def aggregate_trades(
    trades_df: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    frequency: str = "1s",
    windows: Optional[list] = None,
) -> pd.DataFrame:
    """
    Aggregate trade data into rolling windows aligned to *timestamps*.

    Strategy (avoids a Python loop over timestamps):
    1. Bin every trade into the enclosing ``frequency`` bucket.
    2. Group-aggregate each bucket (volume, buy/sell split, count, dollar volume).
    3. Re-index onto *timestamps* (fill missing buckets with zero).
    4. For each *window*, compute a rolling sum with ``n_periods = window / freq``.
    """
    if windows is None:
        windows = ["1s", "5s", "10s"]

    if trades_df.empty:
        cols = []
        for w in windows:
            cols += [
                f"total_volume_{w}", f"buy_volume_{w}", f"sell_volume_{w}",
                f"signed_volume_{w}", f"trade_imbalance_{w}", f"num_trades_{w}",
                f"avg_trade_size_{w}", f"vwap_{w}",
            ]
        return pd.DataFrame(0.0, index=timestamps, columns=cols)

    trades = trades_df.copy()

    # Classify aggressor side:
    #   is_buyer_maker=True  -> seller aggressed (sell trade)
    #   is_buyer_maker=False -> buyer aggressed  (buy trade)
    trades["buy_qty"] = trades["quantity"].where(~trades["is_buyer_maker"], 0.0)
    trades["sell_qty"] = trades["quantity"].where(trades["is_buyer_maker"], 0.0)
    trades["dollar_vol"] = trades["price"] * trades["quantity"]

    # Bin into frequency buckets
    trades["bin"] = trades["timestamp"].dt.floor(frequency)

    binned = (
        trades
        .groupby("bin")
        .agg(
            total_volume=("quantity", "sum"),
            buy_volume=("buy_qty", "sum"),
            sell_volume=("sell_qty", "sum"),
            num_trades=("quantity", "count"),
            dollar_volume=("dollar_vol", "sum"),
        )
    )

    # Align to the target timestamp grid
    binned = binned.reindex(timestamps, fill_value=0.0)

    freq_td = pd.Timedelta(frequency)
    parts: list[pd.DataFrame] = []

    for w in windows:
        n = max(1, int(pd.Timedelta(w) / freq_td))
        r = binned.rolling(n, min_periods=1).sum()

        p = pd.DataFrame({
            f"total_volume_{w}": r["total_volume"],
            f"buy_volume_{w}": r["buy_volume"],
            f"sell_volume_{w}": r["sell_volume"],
            f"signed_volume_{w}": r["buy_volume"] - r["sell_volume"],
            f"num_trades_{w}": r["num_trades"],
        })

        total = p[f"total_volume_{w}"]
        ntrades = p[f"num_trades_{w}"]
        p[f"trade_imbalance_{w}"] = (p[f"signed_volume_{w}"] / total).where(total > 0, 0.0)
        p[f"avg_trade_size_{w}"] = (total / ntrades).where(ntrades > 0, 0.0)
        p[f"vwap_{w}"] = (r["dollar_volume"] / total).where(total > 0, np.nan)

        parts.append(p)

    result = pd.concat(parts, axis=1)
    result.index.name = "timestamp"
    logger.info(
        "Aggregated trades: %d timestamps x %d cols (windows=%s)",
        len(result), len(result.columns), windows,
    )
    return result


# ---------------------------------------------------------------------------
# Combined: resample + align
# ---------------------------------------------------------------------------

def resample_and_align(
    orderbook_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    frequency: str = "1s",
    trade_windows: Optional[list] = None,
) -> pd.DataFrame:
    """
    Resample the order book and aggregate trades, then merge on timestamp.

    Returns a single DataFrame ready for feature engineering.
    """
    if trade_windows is None:
        trade_windows = ["1s", "5s", "10s"]

    logger.info("Resample-and-align | freq=%s | trade_windows=%s", frequency, trade_windows)

    ob = resample_orderbook(orderbook_df, frequency)
    ts_index = pd.DatetimeIndex(ob["timestamp"])
    ta = aggregate_trades(trades_df, ts_index, frequency, trade_windows)
    ta.reset_index(inplace=True)

    merged = pd.merge(ob, ta, on="timestamp", how="left")

    # Fill NaN trade columns with 0 (no trades in that window)
    trade_cols = [c for c in merged.columns if any(
        c.startswith(p) for p in (
            "total_volume_", "buy_volume_", "sell_volume_", "signed_volume_",
            "trade_imbalance_", "num_trades_", "avg_trade_size_",
        )
    )]
    merged[trade_cols] = merged[trade_cols].fillna(0.0)

    logger.info("Aligned dataset: %s", merged.shape)
    return merged
