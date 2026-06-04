"""
Convert Tardis.dev CSV datasets into the project's standard Parquet format.

Tardis book_snapshot_25 → data/raw/{symbol}/orderbook/{date}.parquet
Tardis trades           → data/raw/{symbol}/trades/{date}.parquet
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.paths import ensure_dir, get_data_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

BOOK_LEVELS = 10


def _tardis_ts_to_datetime(series: pd.Series) -> pd.Series:
    """Tardis timestamps are microseconds since epoch."""
    return pd.to_datetime(series.astype(np.int64), unit="us", utc=True)


def convert_book_snapshot(
    csv_path: str | Path,
    symbol: str,
    date_str: str,
    levels: int = BOOK_LEVELS,
    sample_interval: str = "1s",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Read a Tardis book_snapshot_25 CSV and convert to our standard format.

    Downsamples to `sample_interval` (default 1s) by keeping the last
    snapshot per interval — equivalent to what forward-fill resampling does.
    """
    csv_path = Path(csv_path)
    logger.info(f"Reading {csv_path.name} ...")

    cols_to_use = ["timestamp"]
    rename_map = {}
    for i in range(levels):
        for side, our_prefix in [("asks", "ask"), ("bids", "bid")]:
            pcol = f"{side}[{i}].price"
            acol = f"{side}[{i}].amount"
            cols_to_use.extend([pcol, acol])
            rename_map[pcol] = f"{our_prefix}_price_{i+1}"
            rename_map[acol] = f"{our_prefix}_size_{i+1}"

    df = pd.read_csv(csv_path, usecols=cols_to_use, dtype={c: np.float64 for c in cols_to_use if c != "timestamp"})
    df["timestamp"] = _tardis_ts_to_datetime(df["timestamp"])
    df.rename(columns=rename_map, inplace=True)

    n_raw = len(df)

    if sample_interval:
        df = df.set_index("timestamp")
        df = df.resample(sample_interval).last().dropna(subset=["bid_price_1"])
        df = df.reset_index()

    logger.info(f"  {symbol} {date_str}: {n_raw:,} tick snapshots → {len(df):,} rows @ {sample_interval}")

    if output_dir is None:
        output_dir = ensure_dir(get_data_dir("raw") / symbol / "orderbook")
    else:
        output_dir = ensure_dir(Path(output_dir))

    out_path = output_dir / f"{date_str}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"  Saved → {out_path}")
    return df


def convert_trades(
    csv_path: str | Path,
    symbol: str,
    date_str: str,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Read a Tardis trades CSV and convert to our standard format."""
    csv_path = Path(csv_path)
    logger.info(f"Reading {csv_path.name} ...")

    df = pd.read_csv(
        csv_path,
        usecols=["timestamp", "id", "side", "price", "amount"],
        dtype={"id": np.int64, "price": np.float64, "amount": np.float64},
    )

    df["timestamp"] = _tardis_ts_to_datetime(df["timestamp"])
    df.rename(columns={"id": "trade_id", "amount": "quantity"}, inplace=True)
    # Tardis side="buy" means taker is buyer → is_buyer_maker=False
    df["is_buyer_maker"] = df["side"] == "sell"
    df["quote_quantity"] = df["price"] * df["quantity"]
    df.drop(columns=["side"], inplace=True)

    logger.info(f"  {symbol} {date_str}: {len(df):,} trades")

    if output_dir is None:
        output_dir = ensure_dir(get_data_dir("raw") / symbol / "trades")
    else:
        output_dir = ensure_dir(Path(output_dir))

    out_path = output_dir / f"{date_str}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"  Saved → {out_path}")
    return df


def convert_all_tardis(
    tardis_dir: str | Path = "datasets_tardis",
    symbols: Optional[list] = None,
    levels: int = BOOK_LEVELS,
    sample_interval: str = "1s",
) -> dict:
    """
    Scan a Tardis download directory and convert all files to our format.

    Returns a dict of {symbol: {date: {"ob": DataFrame, "tr": DataFrame}}}.
    """
    tardis_dir = Path(tardis_dir)
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]

    results = {}

    for fpath in sorted(tardis_dir.glob("*.csv.gz")):
        name = fpath.stem.replace(".csv", "")
        parts = name.split("_")
        # e.g. binance-futures_book_snapshot_25_2026-06-01_BTCUSDT
        #      binance-futures_trades_2026-06-01_BTCUSDT
        sym = parts[-1]
        date_str = parts[-2]

        if sym not in symbols:
            continue

        if sym not in results:
            results[sym] = {}
        if date_str not in results[sym]:
            results[sym][date_str] = {}

        if "book_snapshot" in name:
            df = convert_book_snapshot(fpath, sym, date_str, levels=levels, sample_interval=sample_interval)
            results[sym][date_str]["ob"] = df
        elif "trades" in name:
            df = convert_trades(fpath, sym, date_str)
            results[sym][date_str]["tr"] = df

    # Summary
    print(f"\n{'='*60}")
    print("TARDIS CONVERSION SUMMARY")
    print(f"{'='*60}")
    for sym in sorted(results):
        print(f"\n{sym}:")
        for date in sorted(results[sym]):
            ob_rows = len(results[sym][date].get("ob", []))
            tr_rows = len(results[sym][date].get("tr", []))
            print(f"  {date}: {ob_rows:>8,} OB rows, {tr_rows:>10,} trades")

    return results


if __name__ == "__main__":
    convert_all_tardis()
