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
    """Load every ticker in ``tickers`` and concatenate into a panel frame."""
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


# ---------------------------------------------------------------------------
# Bulk S&P 500 download with parquet cache
# ---------------------------------------------------------------------------

SP500_HARDCODED: list[str] = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","BRK-B","JPM","JNJ","V",
    "PG","UNH","HD","MA","XOM","CVX","MRK","LLY","ABBV","PEP",
    "KO","AVGO","COST","MCD","WMT","BAC","TMO","CSCO","CRM","ABT",
    "ACN","CMCSA","DHR","VZ","ADBE","NEE","NKE","TXN","PM","RTX",
    "HON","AMGN","UNP","QCOM","LOW","ORCL","BMY","LIN","MDT","T",
    "SBUX","AMD","IBM","GE","CAT","SPGI","GS","MS","BLK","AXP",
    "DE","ISRG","ADP","GILD","MMM","MO","CI","TGT","INTU","NOW",
    "PLD","CB","ZTS","REGN","USB","DUK","SO","ITW","CL","EMR",
    "NSC","SHW","AON","HCA","FISV","ICE","EW","NFLX","MCO","EL",
    "MPC","PSA","VLO","TJX","WM","F","GM","INTC","MU","KLAC",
]


def get_sp500_tickers(use_hardcoded: bool = False) -> list[str]:
    """Return the current S&P 500 constituent list.

    Attempts to scrape Wikipedia; falls back to the 100-ticker hardcoded list
    when the network request fails or ``use_hardcoded`` is True.
    """
    if not use_hardcoded:
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        except Exception as exc:
            logger.warning("Wikipedia scrape failed (%s); using hardcoded list.", exc)
    return SP500_HARDCODED


def download_prices(
    tickers: list[str],
    cache_path: str | Path | None = None,
    n_years: int = 10,
) -> pd.DataFrame:
    """Download daily adjusted-close prices for ``tickers`` via yfinance.

    Results are stored as a wide DataFrame (index=date, columns=ticker) and
    cached to ``cache_path`` as a parquet file.  On subsequent calls the cache
    is read directly without hitting the network.

    Parameters
    ----------
    tickers : list of str
    cache_path : path-like, optional
        Where to read/write the parquet cache.  Pass ``None`` to skip caching.
    n_years : int
        Years of history to download (counting back from today).

    Returns
    -------
    pd.DataFrame
        Wide close-price frame: rows are dates, columns are ticker symbols.
    """
    import yfinance as yf

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached prices from %s", cache_path)
            return pd.read_parquet(cache_path)

    end   = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.DateOffset(years=n_years)).strftime("%Y-%m-%d")
    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]}) if len(tickers) == 1 else raw
    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index)
    close.index.name = "date"

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        close.to_parquet(cache_path)
        logger.info("Saved price cache to %s", cache_path)
    return close
