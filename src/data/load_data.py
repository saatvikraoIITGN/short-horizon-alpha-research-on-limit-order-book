"""Load raw and processed data from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logging_utils import setup_logger
from src.utils.paths import get_data_dir, load_config

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Raw data loaders
# ---------------------------------------------------------------------------

def load_raw_orderbook(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load raw order-book snapshots for *symbol* (optionally filtered by date range)."""
    directory = get_data_dir("raw") / symbol.upper() / "orderbook"
    return _load_parquet_range(directory, start_date, end_date)


def load_raw_trades(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load raw trade data for *symbol*."""
    directory = get_data_dir("raw") / symbol.upper() / "trades"
    return _load_parquet_range(directory, start_date, end_date)


# ---------------------------------------------------------------------------
# Processed / featured data
# ---------------------------------------------------------------------------

def load_processed(symbol: str, name: str = "features") -> pd.DataFrame:
    """Load a processed dataset (e.g. ``features.parquet``)."""
    path = get_data_dir("processed") / symbol.upper() / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    logger.info("Loading %s", path)
    return pd.read_parquet(path)


def load_interim(symbol: str, name: str = "aligned") -> pd.DataFrame:
    """Load an interim dataset (e.g. ``aligned.parquet``)."""
    path = get_data_dir("interim") / symbol.upper() / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Interim data not found: {path}")
    logger.info("Loading %s", path)
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def list_available_data(symbol: Optional[str] = None) -> dict:
    """Return a nested dict describing what raw data exists on disk."""
    raw_dir = get_data_dir("raw")
    config = load_config()
    symbols = [symbol.upper()] if symbol else [s.upper() for s in config["assets"]]

    inventory: dict = {}
    for sym in symbols:
        inventory[sym] = {}
        for dtype in ("orderbook", "trades"):
            d = raw_dir / sym / dtype
            if d.exists():
                files = sorted(d.glob("*.parquet"))
                dates = [f.stem for f in files]
                inventory[sym][dtype] = {
                    "dates": dates,
                    "num_files": len(files),
                }
            else:
                inventory[sym][dtype] = {"dates": [], "num_files": 0}
    return inventory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet_range(
    directory: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}")

    if start_date:
        files = [f for f in files if f.stem >= start_date]
    if end_date:
        files = [f for f in files if f.stem <= end_date]
    if not files:
        raise FileNotFoundError(
            f"No parquet files in range [{start_date}, {end_date}] under {directory}"
        )

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
        logger.info("  loaded %s  (%d rows)", f.name, len(df))

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    logger.info("Total: %d rows from %d file(s)", len(combined), len(files))
    return combined
