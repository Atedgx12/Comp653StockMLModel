"""Technical indicator implementations.

Every function here takes a per asset frame indexed by date with at least the
``open``, ``high``, ``low``, ``close``, and ``volume`` columns and returns a
new frame with additional indicator columns. All rolling windows use only
prior data to avoid look ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_per_asset(df: pd.DataFrame) -> None:
    """Guard against accidentally feeding the multi ticker panel directly."""
    if "ticker" in df.columns and df["ticker"].nunique() > 1:
        raise ValueError(
            "Technical indicator functions expect a single asset frame. "
            "Use build_features which groups by ticker before calling these."
        )


def add_log_returns(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Add log return columns at the requested look back periods."""
    _ensure_per_asset(df)
    periods = periods or [1, 5, 20]
    df = df.copy()
    log_close = np.log(df["close"])
    for k in periods:
        df[f"log_return_{k}"] = log_close.diff(k)
    return df


def add_ema(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Add exponential moving averages and price to EMA ratios."""
    _ensure_per_asset(df)
    df = df.copy()
    for p in periods:
        ema = df["close"].ewm(span=p, adjust=False, min_periods=p).mean()
        df[f"ema_{p}"] = ema
        df[f"close_over_ema_{p}"] = df["close"] / ema - 1.0
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD line, signal, and histogram."""
    _ensure_per_asset(df)
    df = df.copy()
    fast_ema = df["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = df["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = fast_ema - slow_ema
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = macd - sig
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder style relative strength index."""
    _ensure_per_asset(df)
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df[f"rsi_{period}"] = 100.0 - 100.0 / (1.0 + rs)
    return df


def add_realized_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Rolling standard deviation of one period log returns scaled to annual."""
    _ensure_per_asset(df)
    df = df.copy()
    if "log_return_1" not in df.columns:
        df = add_log_returns(df, periods=[1])
    vol = (
        df["log_return_1"].rolling(period, min_periods=period).std(ddof=0)
        * np.sqrt(252.0)
    )
    df[f"realized_vol_{period}"] = vol
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average true range scaled by closing price."""
    _ensure_per_asset(df)
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    df[f"atr_{period}"] = atr
    df[f"atr_{period}_pct"] = atr / df["close"]
    return df


def add_bollinger_width(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger band width and z score of close vs the rolling mean."""
    _ensure_per_asset(df)
    df = df.copy()
    mean = df["close"].rolling(period, min_periods=period).mean()
    std = df["close"].rolling(period, min_periods=period).std(ddof=0)
    df[f"bb_width_{period}"] = (2.0 * num_std * std) / mean
    df[f"bb_z_{period}"] = (df["close"] - mean) / std
    return df


def add_rate_of_change(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Simple rate of change indicators."""
    _ensure_per_asset(df)
    df = df.copy()
    for p in periods:
        df[f"roc_{p}"] = df["close"].pct_change(p)
    return df


def add_rolling_high_low(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Distance to the rolling high and low at the requested look backs."""
    _ensure_per_asset(df)
    df = df.copy()
    for p in periods:
        rh = df["close"].rolling(p, min_periods=p).max()
        rl = df["close"].rolling(p, min_periods=p).min()
        df[f"dist_to_high_{p}"] = df["close"] / rh - 1.0
        df[f"dist_to_low_{p}"] = df["close"] / rl - 1.0
    return df


def add_close_in_range(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Position of the latest close inside the rolling high to low range."""
    _ensure_per_asset(df)
    df = df.copy()
    rh = df["high"].rolling(period, min_periods=period).max()
    rl = df["low"].rolling(period, min_periods=period).min()
    rng = (rh - rl).replace(0.0, np.nan)
    df[f"close_in_range_{period}"] = (df["close"] - rl) / rng
    return df
