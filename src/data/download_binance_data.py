"""
Download and collect Binance order book and trade data.

Three collection modes
---------------------
1. **websocket**  -- Real-time L2 depth snapshots + trades via WebSocket streams.
2. **rest**       -- Periodic REST API polling (simpler, good for testing).
3. **historical** -- Bulk download from https://data.binance.vision (trades for
   spot; trades *and* order-book depth for USDT-M futures).

Usage examples::

    # Collect live data via websocket for 24 hours
    python -m src.data.download_binance_data websocket --hours 24

    # Collect via REST polling for 1 hour (quick test)
    python -m src.data.download_binance_data rest --hours 1

    # Download 7 days of historical futures data (trades + depth)
    python -m src.data.download_binance_data historical \
        --start 2025-01-01 --end 2025-01-07 --market futures --data-type both
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import websockets

from typing import Optional

from src.utils.logging_utils import setup_logger
from src.utils.paths import ensure_dir, get_data_dir, load_config

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# WebSocket collector
# ---------------------------------------------------------------------------

class WebSocketCollector:
    """Collect order book snapshots and trades via Binance combined WebSocket stream."""

    def __init__(self, symbols: list[str], config: dict):
        self.symbols = [s.lower() for s in symbols]
        bcfg = config["data"].get("binance", {})
        self.ws_base = bcfg.get("ws_base_url", "wss://stream.binance.com:9443")
        self.depth_levels = bcfg.get("depth_levels", 10)
        self.depth_speed = bcfg.get("depth_update_speed", "100ms")
        self.output_dir = get_data_dir("raw")

        self.ob_buf: dict[str, list[dict]] = {s: [] for s in self.symbols}
        self.tr_buf: dict[str, list[dict]] = {s: [] for s in self.symbols}
        self.flush_every = 60  # seconds
        self.stats = {s: {"ob": 0, "tr": 0} for s in self.symbols}

    # -- stream URL ----------------------------------------------------------

    def _stream_url(self) -> str:
        streams = []
        for sym in self.symbols:
            streams.append(f"{sym}@depth{self.depth_levels}@{self.depth_speed}")
            streams.append(f"{sym}@trade")
        return f"{self.ws_base}/stream?streams={'/'.join(streams)}"

    # -- parsers -------------------------------------------------------------

    @staticmethod
    def _parse_depth(data: dict) -> dict:
        record: dict = {"timestamp": pd.Timestamp.now(tz="UTC")}
        for i, (price, qty) in enumerate(data.get("bids", []), start=1):
            record[f"bid_price_{i}"] = float(price)
            record[f"bid_size_{i}"] = float(qty)
        for i, (price, qty) in enumerate(data.get("asks", []), start=1):
            record[f"ask_price_{i}"] = float(price)
            record[f"ask_size_{i}"] = float(qty)
        return record

    @staticmethod
    def _parse_trade(data: dict) -> dict:
        return {
            "timestamp": pd.Timestamp(data["T"], unit="ms", tz="UTC"),
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "is_buyer_maker": data["m"],
            "trade_id": data["t"],
        }

    # -- flush to parquet ----------------------------------------------------

    def _flush(self) -> None:
        for sym in self.symbols:
            sym_upper = sym.upper()
            for buf, dtype in [(self.ob_buf, "orderbook"), (self.tr_buf, "trades")]:
                if not buf[sym]:
                    continue
                df = pd.DataFrame(buf[sym])
                date_str = df["timestamp"].iloc[0].strftime("%Y-%m-%d")
                out_dir = ensure_dir(self.output_dir / sym_upper / dtype)
                out_path = out_dir / f"{date_str}.parquet"

                if out_path.exists():
                    df = pd.concat([pd.read_parquet(out_path), df], ignore_index=True)
                df.to_parquet(out_path, index=False)
                buf[sym] = []

    # -- main loop -----------------------------------------------------------

    async def collect(self, duration_hours: float) -> None:
        url = self._stream_url()
        deadline = time.time() + duration_hours * 3600
        last_flush = time.time()

        logger.info("WS collection | symbols=%s | url=%s | hours=%.1f",
                     self.symbols, url, duration_hours)

        while time.time() < deadline:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info("WebSocket connected")
                    while time.time() < deadline:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            continue

                        payload = json.loads(raw)
                        stream = payload.get("stream", "")
                        data = payload.get("data", {})
                        sym = stream.split("@")[0]

                        if "@depth" in stream:
                            self.ob_buf[sym].append(self._parse_depth(data))
                            self.stats[sym]["ob"] += 1
                        elif "@trade" in stream:
                            self.tr_buf[sym].append(self._parse_trade(data))
                            self.stats[sym]["tr"] += 1

                        if time.time() - last_flush > self.flush_every:
                            self._flush()
                            last_flush = time.time()
                            self._log_stats()

            except (websockets.ConnectionClosed, ConnectionError, OSError) as exc:
                logger.warning("WS disconnected (%s). Reconnecting in 5 s ...", exc)
                self._flush()
                await asyncio.sleep(5)

        self._flush()
        self._log_stats()
        logger.info("WebSocket collection finished")

    def _log_stats(self) -> None:
        for sym in self.symbols:
            s = self.stats[sym]
            logger.info("%s | snapshots=%d  trades=%d", sym.upper(), s["ob"], s["tr"])


# ---------------------------------------------------------------------------
# REST poller
# ---------------------------------------------------------------------------

class RESTCollector:
    """Collect data by polling the Binance REST API at fixed intervals."""

    def __init__(self, symbols: list[str], config: dict):
        self.symbols = [s.upper() for s in symbols]
        bcfg = config["data"].get("binance", {})
        self.base_url = bcfg.get("rest_base_url", "https://api.binance.com")
        self.book_levels = bcfg.get("depth_levels", 10)
        self.output_dir = get_data_dir("raw")
        self.session = requests.Session()

    # -- fetchers ------------------------------------------------------------

    def _fetch_depth(self, symbol: str) -> Optional[dict]:
        try:
            r = self.session.get(
                f"{self.base_url}/api/v3/depth",
                params={"symbol": symbol, "limit": self.book_levels},
                timeout=5,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.warning("Depth fetch failed [%s]: %s", symbol, exc)
            return None

    def _fetch_trades(self, symbol: str, limit: int = 1000) -> Optional[list]:
        try:
            r = self.session.get(
                f"{self.base_url}/api/v3/trades",
                params={"symbol": symbol, "limit": limit},
                timeout=5,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.warning("Trades fetch failed [%s]: %s", symbol, exc)
            return None

    # -- parsers -------------------------------------------------------------

    @staticmethod
    def _parse_depth(data: dict) -> dict:
        now = pd.Timestamp.now(tz="UTC")
        rec: dict = {"timestamp": now}
        for i, (price, qty) in enumerate(data.get("bids", []), start=1):
            rec[f"bid_price_{i}"] = float(price)
            rec[f"bid_size_{i}"] = float(qty)
        for i, (price, qty) in enumerate(data.get("asks", []), start=1):
            rec[f"ask_price_{i}"] = float(price)
            rec[f"ask_size_{i}"] = float(qty)
        return rec

    @staticmethod
    def _parse_trades(raw_trades: list) -> list[dict]:
        return [
            {
                "timestamp": pd.Timestamp(t["time"], unit="ms", tz="UTC"),
                "price": float(t["price"]),
                "quantity": float(t["qty"]),
                "is_buyer_maker": t["isBuyerMaker"],
                "trade_id": t["id"],
            }
            for t in raw_trades
        ]

    # -- flush ---------------------------------------------------------------

    def _flush(self, symbol: str, records: list[dict], dtype: str) -> None:
        if not records:
            return
        df = pd.DataFrame(records)
        date_str = df["timestamp"].iloc[0].strftime("%Y-%m-%d")
        out_dir = ensure_dir(self.output_dir / symbol / dtype)
        out_path = out_dir / f"{date_str}.parquet"
        if out_path.exists():
            df = pd.concat([pd.read_parquet(out_path), df], ignore_index=True)
        df.to_parquet(out_path, index=False)

    # -- main loop -----------------------------------------------------------

    def collect(self, duration_hours: float, interval_sec: float = 1.0) -> None:
        deadline = time.time() + duration_hours * 3600
        ob_buf: dict[str, list] = {s: [] for s in self.symbols}
        tr_buf: dict[str, list] = {s: [] for s in self.symbols}
        seen_ids: dict[str, set] = {s: set() for s in self.symbols}

        flush_every = 60
        last_flush = time.time()
        polls = 0

        logger.info("REST collection | symbols=%s | interval=%.1fs | hours=%.1f",
                     self.symbols, interval_sec, duration_hours)

        while time.time() < deadline:
            t0 = time.time()
            for sym in self.symbols:
                depth = self._fetch_depth(sym)
                if depth:
                    ob_buf[sym].append(self._parse_depth(depth))

                raw_trades = self._fetch_trades(sym)
                if raw_trades:
                    for t in self._parse_trades(raw_trades):
                        tid = t["trade_id"]
                        if tid not in seen_ids[sym]:
                            seen_ids[sym].add(tid)
                            tr_buf[sym].append(t)
                    if len(seen_ids[sym]) > 100_000:
                        seen_ids[sym] = set(list(seen_ids[sym])[-50_000:])

            polls += 1

            if time.time() - last_flush > flush_every:
                for sym in self.symbols:
                    self._flush(sym, ob_buf[sym], "orderbook")
                    self._flush(sym, tr_buf[sym], "trades")
                    ob_buf[sym], tr_buf[sym] = [], []
                last_flush = time.time()
                logger.info("Flushed after %d polls", polls)

            sleep = max(0.0, interval_sec - (time.time() - t0))
            if sleep:
                time.sleep(sleep)

        for sym in self.symbols:
            self._flush(sym, ob_buf[sym], "orderbook")
            self._flush(sym, tr_buf[sym], "trades")
        logger.info("REST collection done | total polls=%d", polls)


# ---------------------------------------------------------------------------
# Historical bulk download from data.binance.vision
# ---------------------------------------------------------------------------

TRADE_COLS = [
    "trade_id", "price", "quantity", "quote_quantity",
    "timestamp_ms", "is_buyer_maker", "is_best_match",
]


def download_historical_trades(
    symbols: list[str],
    start_date: str,
    end_date: str,
    config: dict,
    market: str = "spot",
) -> None:
    """Download daily trade CSVs from data.binance.vision and save as parquet."""
    base = config["data"].get("binance", {}).get(
        "data_vision_base_url", "https://data.binance.vision"
    )
    out_root = get_data_dir("raw")

    if market == "futures":
        tpl = "{base}/data/futures/um/daily/trades/{sym}/{sym}-trades-{date}.zip"
    else:
        tpl = "{base}/data/spot/daily/trades/{sym}/{sym}-trades-{date}.zip"

    for sym in (s.upper() for s in symbols):
        cur = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        logger.info("Downloading trades for %s [%s -> %s] (%s)", sym, start_date, end_date, market)

        while cur <= end:
            ds = cur.strftime("%Y-%m-%d")
            url = tpl.format(base=base, sym=sym, date=ds)
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 404:
                    logger.warning("  %s: not available", ds)
                    cur += timedelta(days=1)
                    continue
                resp.raise_for_status()

                with ZipFile(BytesIO(resp.content)) as zf:
                    with zf.open(zf.namelist()[0]) as f:
                        df = pd.read_csv(f, header=None, names=TRADE_COLS)

                df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
                df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
                df = df[["timestamp", "trade_id", "price", "quantity", "is_buyer_maker"]]
                df["price"] = df["price"].astype(float)
                df["quantity"] = df["quantity"].astype(float)

                out_dir = ensure_dir(out_root / sym / "trades")
                df.to_parquet(out_dir / f"{ds}.parquet", index=False)
                logger.info("  %s: %d trades", ds, len(df))

            except requests.RequestException as exc:
                logger.error("  %s: download failed -- %s", ds, exc)

            cur += timedelta(days=1)


def download_historical_book_depth(
    symbols: list[str],
    start_date: str,
    end_date: str,
    config: dict,
    level: str = "S5",
) -> None:
    """
    Download daily book-depth snapshots from data.binance.vision (futures only).

    Binance publishes USDT-M futures depth at S5 (top 5) and S20 (top 20).
    """
    base = config["data"].get("binance", {}).get(
        "data_vision_base_url", "https://data.binance.vision"
    )
    out_root = get_data_dir("raw")
    book_levels = 5 if level == "S5" else 20

    tpl = (
        "{base}/data/futures/um/daily/bookDepth/"
        "{sym}/{sym}-bookDepth-{lvl}-{date}.zip"
    )

    for sym in (s.upper() for s in symbols):
        cur = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        logger.info("Downloading depth (%s) for %s [%s -> %s]",
                     level, sym, start_date, end_date)

        while cur <= end:
            ds = cur.strftime("%Y-%m-%d")
            url = tpl.format(base=base, sym=sym, lvl=level, date=ds)
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 404:
                    logger.warning("  %s: not available", ds)
                    cur += timedelta(days=1)
                    continue
                resp.raise_for_status()

                with ZipFile(BytesIO(resp.content)) as zf:
                    with zf.open(zf.namelist()[0]) as f:
                        raw = pd.read_csv(f, header=None)

                df = _parse_book_depth_csv(raw, book_levels)
                out_dir = ensure_dir(out_root / sym / "orderbook")
                df.to_parquet(out_dir / f"{ds}.parquet", index=False)
                logger.info("  %s: %d snapshots", ds, len(df))

            except requests.RequestException as exc:
                logger.error("  %s: download failed -- %s", ds, exc)
            except Exception as exc:
                logger.error("  %s: parse error -- %s", ds, exc)

            cur += timedelta(days=1)


def _parse_book_depth_csv(raw: pd.DataFrame, book_levels: int) -> pd.DataFrame:
    """
    Normalize a Binance Data Vision book-depth CSV.

    Common layouts
    - ``symbol, ts, first_update_id, last_update_id, bids..., asks...``
    - ``ts, bids..., asks...``

    After parsing, the output has columns:
    ``timestamp, bid_price_1, bid_size_1, ..., ask_price_1, ask_size_1, ...``
    """
    first_val = raw.iloc[0, 0]
    n_data = book_levels * 4  # bid_price, bid_size, ask_price, ask_size per level

    # Detect metadata columns
    if isinstance(first_val, str) and not first_val.replace(".", "").replace("-", "").isdigit():
        meta = ["symbol", "timestamp_ms", "first_update_id", "last_update_id"]
    else:
        n_extra = len(raw.columns) - n_data
        if n_extra == 2:
            meta = ["timestamp_ms", "update_id"]
        else:
            meta = ["timestamp_ms"]

    cols = list(meta)
    for i in range(1, book_levels + 1):
        cols.extend([f"bid_price_{i}", f"bid_size_{i}"])
    for i in range(1, book_levels + 1):
        cols.extend([f"ask_price_{i}", f"ask_size_{i}"])

    # Auto-detect fallback if column counts diverge
    if len(cols) != len(raw.columns):
        logger.warning(
            "Column mismatch (expected %d, got %d); falling back to auto-detect",
            len(cols), len(raw.columns),
        )
        cols = ["timestamp_ms"]
        remaining = len(raw.columns) - 1
        lvls = remaining // 4
        for i in range(1, lvls + 1):
            cols.extend([f"bid_price_{i}", f"bid_size_{i}"])
        for i in range(1, lvls + 1):
            cols.extend([f"ask_price_{i}", f"ask_size_{i}"])
        cols = cols[: len(raw.columns)]

    raw.columns = cols

    if "timestamp_ms" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp_ms"], unit="ms", utc=True)

    keep = ["timestamp"]
    for i in range(1, book_levels + 1):
        for c in (f"bid_price_{i}", f"bid_size_{i}", f"ask_price_{i}", f"ask_size_{i}"):
            if c in raw.columns:
                keep.append(c)

    out = raw[[c for c in keep if c in raw.columns]].copy()
    for c in out.columns:
        if c != "timestamp":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binance LOB data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- websocket -----------------------------------------------------------
    ws = sub.add_parser("websocket", help="Real-time WebSocket collection")
    ws.add_argument("--hours", type=float, required=True)
    ws.add_argument("--symbols", nargs="+", default=None)

    # -- rest ----------------------------------------------------------------
    rs = sub.add_parser("rest", help="REST API polling")
    rs.add_argument("--hours", type=float, required=True)
    rs.add_argument("--interval", type=float, default=1.0)
    rs.add_argument("--symbols", nargs="+", default=None)

    # -- historical ----------------------------------------------------------
    hi = sub.add_parser("historical", help="Bulk download from data.binance.vision")
    hi.add_argument("--start", required=True, help="YYYY-MM-DD")
    hi.add_argument("--end", required=True, help="YYYY-MM-DD")
    hi.add_argument("--symbols", nargs="+", default=None)
    hi.add_argument("--market", choices=["spot", "futures"], default="spot")
    hi.add_argument(
        "--data-type", choices=["trades", "depth", "both"], default="both",
        help="What to download (depth only available for futures)",
    )
    hi.add_argument("--depth-level", choices=["S5", "S20"], default="S5")

    args = parser.parse_args()
    config = load_config()
    symbols = args.symbols or config["assets"]

    if args.mode == "websocket":
        asyncio.run(WebSocketCollector(symbols, config).collect(args.hours))

    elif args.mode == "rest":
        RESTCollector(symbols, config).collect(args.hours, args.interval)

    elif args.mode == "historical":
        if args.data_type in ("trades", "both"):
            download_historical_trades(symbols, args.start, args.end, config, args.market)
        if args.data_type in ("depth", "both"):
            if args.market != "futures":
                logger.warning(
                    "Book depth archives exist only for futures. Switching market to futures."
                )
            download_historical_book_depth(
                symbols, args.start, args.end, config, level=args.depth_level,
            )


if __name__ == "__main__":
    main()
