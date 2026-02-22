"""Phase 1 - Data Acquisition & Preprocessing for BTC/USDT 1h data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ccxt
import pandas as pd


@dataclass
class DataConfig:
    """Configuration for historical OHLCV download and preprocessing."""

    exchange_id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    years: float = 4.0
    limit_per_call: int = 1000


class BTCDataPipeline:
    """Fetch and preprocess BTC/USDT OHLCV data using ccxt."""

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        self.config = config or DataConfig()
        self.exchange = self._build_exchange()

    def _build_exchange(self):
        if self.config.exchange_id != "binance":
            raise ValueError("This pipeline currently supports only the Binance exchange.")

        exchange = ccxt.binance({"enableRateLimit": True})
        exchange.load_markets()
        return exchange

    def fetch_ohlcv(self) -> pd.DataFrame:
        """Fetch OHLCV for the configured period using paginated ccxt calls."""
        timeframe_ms = self.exchange.parse_timeframe(self.config.timeframe) * 1000
        now_ms = self.exchange.milliseconds()
        since_ms = now_ms - int(self.config.years * 365.25 * 24 * 60 * 60 * 1000)

        all_rows = []
        cursor = since_ms

        while True:
            batch = self.exchange.fetch_ohlcv(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                since=cursor,
                limit=self.config.limit_per_call,
            )

            if not batch:
                break

            all_rows.extend(batch)
            last_ts = batch[-1][0]
            next_cursor = last_ts + timeframe_ms

            if next_cursor <= cursor or next_cursor >= now_ms:
                break

            cursor = next_cursor

        if not all_rows:
            raise RuntimeError("No OHLCV data returned from exchange.")

        raw = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        raw["datetime"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
        return raw

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLCV data: remove duplicates, enforce monotonic index, drop NaNs."""
        required_cols = {"datetime", "open", "high", "low", "close", "volume"}
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        clean = df.copy()
        clean = clean.drop_duplicates(subset=["datetime"], keep="last")
        clean = clean.sort_values("datetime")
        clean = clean.set_index("datetime")

        clean = clean[~clean.index.duplicated(keep="last")]
        clean = clean.dropna(how="any")

        if not clean.index.is_monotonic_increasing:
            raise RuntimeError("Datetime index is not strictly monotonic increasing.")

        return clean

    def run(self) -> pd.DataFrame:
        """Execute full Phase 1 pipeline."""
        raw = self.fetch_ohlcv()
        clean = self.preprocess(raw)
        return clean


if __name__ == "__main__":
    pipeline = BTCDataPipeline(DataConfig(years=4.0))
    dataset = pipeline.run()
    print(dataset.tail())
    print(f"Rows: {len(dataset):,} | Start: {dataset.index.min()} | End: {dataset.index.max()}")
