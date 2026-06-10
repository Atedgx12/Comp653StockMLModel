"""Loaders for the data sources used by the project."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase and ensure required columns exist."""
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    if "adjclose" in df.columns:
        df = df.rename(columns={"adjclose": "adj_close"})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Loader produced frame missing required columns: {missing}")
    return df


def _coerce_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a sorted DatetimeIndex named ``date``."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"]))
            df = df.drop(columns=["date"])
        else:
            df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


@dataclass
class KaggleStocksLoader:
    """Reads single ticker CSV files from the Huge Stock Market Kaggle dataset.

    The dataset stores one file per ticker as ``<ticker>.us.txt`` in CSV form
    with columns Date, Open, High, Low, Close, Volume, OpenInt. The loader
    accepts either that filename or a plain ``<TICKER>.csv``.
    """

    raw_root: Path

    def __post_init__(self) -> None:
        self.raw_root = Path(self.raw_root)

    def candidate_paths(self, ticker: str) -> list[Path]:
        ticker = ticker.strip().lower()
        return [
            self.raw_root / f"{ticker}.us.txt",
            self.raw_root / f"{ticker}.us.csv",
            self.raw_root / f"{ticker}.csv",
        ]

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        for candidate in self.candidate_paths(ticker):
            if candidate.exists():
                df = pd.read_csv(candidate)
                df = _normalize_columns(df)
                df = _coerce_index(df)
                df["ticker"] = ticker.upper()
                return df
        raise FileNotFoundError(
            f"No CSV for ticker {ticker} found under {self.raw_root}"
        )

    def list_available(self) -> list[str]:
        if not self.raw_root.exists():
            return []
        names: set[str] = set()
        for path in self.raw_root.iterdir():
            if path.suffix.lower() not in {".csv", ".txt"}:
                continue
            stem = path.stem
            if stem.endswith(".us"):
                stem = stem[: -len(".us")]
            names.add(stem.upper())
        return sorted(names)


@dataclass
class YFinanceLoader:
    """Thin wrapper over yfinance with retries and column normalization."""

    start: str
    end: str
    auto_adjust: bool = True
    interval: str = "1d"

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        import yfinance as yf

        logger.info("Downloading %s via yfinance", ticker)
        raw = yf.download(
            ticker,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=self.auto_adjust,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        # yfinance can return multi index columns when called with a list.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        df = _normalize_columns(raw)
        df = _coerce_index(df)
        df["ticker"] = ticker.upper()
        return df


def load_universe(
    loader: KaggleStocksLoader | YFinanceLoader,
    tickers: Iterable[str],
) -> pd.DataFrame:
    """Load every ticker in ``tickers`` and concatenate into a panel frame.

    The returned frame is sorted by (ticker, date) and indexed by date with a
    ticker column so downstream feature code can group by ticker without
    collapsing the time index.
    """
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            frames.append(loader.load_ticker(ticker))
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
    if not frames:
        raise RuntimeError("No tickers could be loaded")
    panel = pd.concat(frames, axis=0)
    panel = panel.sort_values(["ticker", "date"]) if "date" in panel.columns else panel.sort_index()
    return panel
