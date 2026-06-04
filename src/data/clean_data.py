"""
Data cleaning pipeline for order-book snapshots and trades.

Bad high-frequency data can manufacture fake alpha.  Every cleaning step is
explicit, logged, and collected into a ``CleaningReport`` so the research
memo can document exactly what was removed and why.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Cleaning report
# ---------------------------------------------------------------------------

class CleaningReport:
    """Accumulate per-step cleaning statistics."""

    def __init__(self, initial_rows: int):
        self.initial_rows = initial_rows
        self.steps: list[dict] = []

    def log_step(self, name: str, before: int, after: int) -> None:
        removed = before - after
        pct = 100.0 * removed / max(before, 1)
        self.steps.append(
            {"step": name, "rows_before": before, "rows_after": after,
             "rows_removed": removed, "pct_removed": round(pct, 4)}
        )
        logger.info("  %-30s removed %6d rows (%.2f%%)", name, removed, pct)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    @property
    def total_removed(self) -> int:
        return self.initial_rows - (self.steps[-1]["rows_after"] if self.steps else self.initial_rows)


# ---------------------------------------------------------------------------
# Order-book cleaning
# ---------------------------------------------------------------------------

def clean_orderbook(
    df: pd.DataFrame,
    book_levels: int = 5,
    max_spread_multiple: float = 10.0,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Return ``(cleaned_df, report)`` after applying every check listed in the
    project spec: duplicates, missing values, crossed books, zero spreads,
    outlier spreads, invalid prices/sizes, and book-ordering violations.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    rpt = CleaningReport(len(df))

    # 1. Duplicate timestamps
    n = len(df)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    rpt.log_step("duplicate_timestamps", n, len(df))

    # 2. Missing top-of-book
    n = len(df)
    required = ["bid_price_1", "bid_size_1", "ask_price_1", "ask_size_1"]
    df.dropna(subset=[c for c in required if c in df.columns], inplace=True)
    rpt.log_step("missing_top_of_book", n, len(df))

    # 3. Crossed books (bid >= ask)
    n = len(df)
    df = df[df["bid_price_1"] < df["ask_price_1"]]
    rpt.log_step("crossed_books", n, len(df))

    # 4. Zero / negative spread
    n = len(df)
    spread = df["ask_price_1"] - df["bid_price_1"]
    df = df[spread > 0]
    rpt.log_step("zero_negative_spread", n, len(df))

    # 5. Outlier spread (> max_spread_multiple * median)
    n = len(df)
    spread = df["ask_price_1"] - df["bid_price_1"]
    med = spread.median()
    if med > 0:
        df = df[spread <= med * max_spread_multiple]
    rpt.log_step("outlier_spreads", n, len(df))

    # 6. Non-positive prices at any level
    n = len(df)
    for i in range(1, book_levels + 1):
        for side in ("bid", "ask"):
            c = f"{side}_price_{i}"
            if c in df.columns:
                df = df[df[c] > 0]
    rpt.log_step("invalid_prices", n, len(df))

    # 7. Negative sizes
    n = len(df)
    for i in range(1, book_levels + 1):
        for side in ("bid", "ask"):
            c = f"{side}_size_{i}"
            if c in df.columns:
                df = df[df[c] >= 0]
    rpt.log_step("negative_sizes", n, len(df))

    # 8. Book-level ordering (bid monotonically decreasing, ask increasing)
    n = len(df)
    mask = pd.Series(True, index=df.index)
    for i in range(1, book_levels):
        bp, bp_next = f"bid_price_{i}", f"bid_price_{i+1}"
        ap, ap_next = f"ask_price_{i}", f"ask_price_{i+1}"
        if bp in df.columns and bp_next in df.columns:
            mask &= df[bp] >= df[bp_next]
        if ap in df.columns and ap_next in df.columns:
            mask &= df[ap] <= df[ap_next]
    df = df[mask]
    rpt.log_step("book_ordering", n, len(df))

    df.reset_index(drop=True, inplace=True)
    logger.info(
        "OB cleaned: %d -> %d  (removed %d, %.2f%%)",
        rpt.initial_rows, len(df), rpt.total_removed,
        100 * rpt.total_removed / max(rpt.initial_rows, 1),
    )
    return df, rpt


# ---------------------------------------------------------------------------
# Trade cleaning
# ---------------------------------------------------------------------------

def clean_trades(
    df: pd.DataFrame,
    ob_start: Optional[pd.Timestamp] = None,
    ob_end: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean trade data.  Optionally restrict to the order-book time range so
    trades that cannot be aligned to snapshots are removed.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    rpt = CleaningReport(len(df))

    # 1. Duplicate trade IDs
    n = len(df)
    if "trade_id" in df.columns:
        df.drop_duplicates(subset=["trade_id"], keep="first", inplace=True)
    rpt.log_step("duplicate_trade_ids", n, len(df))

    # 2. Missing critical fields
    n = len(df)
    df.dropna(subset=["timestamp", "price", "quantity"], inplace=True)
    rpt.log_step("missing_fields", n, len(df))

    # 3. Zero / negative price or quantity
    n = len(df)
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]
    rpt.log_step("invalid_price_or_qty", n, len(df))

    # 4. Outlier trade prices (> 5 std from rolling median)
    n = len(df)
    if len(df) > 100:
        roll_med = df["price"].rolling(1000, min_periods=10, center=True).median()
        roll_std = df["price"].rolling(1000, min_periods=10, center=True).std()
        keep = ((df["price"] - roll_med).abs() <= 5 * roll_std).fillna(True)
        df = df[keep]
    rpt.log_step("outlier_prices", n, len(df))

    # 5. Align to order-book time window
    n = len(df)
    if ob_start is not None:
        df = df[df["timestamp"] >= ob_start]
    if ob_end is not None:
        df = df[df["timestamp"] <= ob_end]
    rpt.log_step("time_range_alignment", n, len(df))

    df.reset_index(drop=True, inplace=True)
    logger.info(
        "Trades cleaned: %d -> %d  (removed %d, %.2f%%)",
        rpt.initial_rows, len(df), rpt.total_removed,
        100 * rpt.total_removed / max(rpt.initial_rows, 1),
    )
    return df, rpt


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(
    orderbook_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    book_levels: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run full cleaning on both datasets.

    Returns ``(clean_ob, clean_trades, {"orderbook": report_df, "trades": report_df})``.
    """
    logger.info("=" * 60)
    logger.info("CLEANING PIPELINE  |  OB=%d  Trades=%d", len(orderbook_df), len(trades_df))
    logger.info("=" * 60)

    clean_ob, ob_rpt = clean_orderbook(orderbook_df, book_levels=book_levels)

    ob_start = clean_ob["timestamp"].min()
    ob_end = clean_ob["timestamp"].max()
    clean_tr, tr_rpt = clean_trades(trades_df, ob_start=ob_start, ob_end=ob_end)

    logger.info("=" * 60)
    logger.info("DONE  |  OB=%d  Trades=%d", len(clean_ob), len(clean_tr))
    logger.info("=" * 60)

    return clean_ob, clean_tr, {"orderbook": ob_rpt.summary(), "trades": tr_rpt.summary()}
