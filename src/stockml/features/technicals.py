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


# ---------------------------------------------------------------------------
# COMP 653 multi-timeframe pyramid (32 raw features per ticker)
# ---------------------------------------------------------------------------
# These features replicate the exact feature set used in pipeline_course.py:
#   returns  : ret1,2,3,5,10,20,60,120,252 (log), ret756 (3-yr reversal)
#   volatility: vol5,10,20,60,120,252 (rolling std of daily log-returns)
#   momentum  : mom5,10,20,60,120,252 (price / price-N-days-ago - 1)
#   structural: vol_ratio, ma50_ratio, ma200_ratio, ma50_200_cross, ret_accel
#   oscillator: rsi14
#   distance  : dist52h, dist52l, dist3yh, dist3yl
# Minimum price history required: 756 trading days (~3 years).

COURSE_FEATURE_NAMES: list[str] = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20",
    "ret60", "ret120", "ret252", "ret756",
    "vol5", "vol10", "vol20", "vol60", "vol120", "vol252",
    "mom5", "mom10", "mom20", "mom60", "mom120", "mom252",
    "vol_ratio", "ma50_ratio", "ma200_ratio", "ma50_200_cross", "ret_accel",
    "rsi14",
    "dist52h", "dist52l", "dist3yh", "dist3yl",
]


def _rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    loss  = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def build_course_features(close: pd.Series, min_len: int = 756) -> pd.DataFrame | None:
    """Compute the 32-feature multi-timeframe pyramid for one ticker.

    Parameters
    ----------
    close : pd.Series
        Daily adjusted-close prices indexed by date, sorted ascending.
    min_len : int
        Minimum number of price observations required.  Returns ``None``
        when the series is shorter.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns matching ``COURSE_FEATURE_NAMES`` plus a
        ``_fwd`` column (1-day forward log-return used as the label).
        Rows with NaN are dropped.
    """
    if len(close) < min_len:
        return None

    log_c = np.log(close.values.astype(float))
    n     = len(log_c)

    rows: dict[str, np.ndarray] = {}

    # Returns (log)
    for k in (1, 2, 3, 5, 10, 20, 60, 120, 252, 756):
        r = np.full(n, np.nan)
        r[k:] = log_c[k:] - log_c[:-k]
        rows[f"ret{k}"] = r

    # Volatility: rolling std of daily log returns
    daily_ret = np.concatenate([[np.nan], log_c[1:] - log_c[:-1]])
    s = pd.Series(daily_ret, index=close.index)
    for k in (5, 10, 20, 60, 120, 252):
        rows[f"vol{k}"] = s.rolling(k, min_periods=k).std().values

    # Momentum: price / price-k-days-ago - 1
    p = close.values.astype(float)
    for k in (5, 10, 20, 60, 120, 252):
        m = np.full(n, np.nan)
        m[k:] = p[k:] / p[:-k] - 1.0
        rows[f"mom{k}"] = m

    # Structural
    vol5  = pd.Series(daily_ret, index=close.index).rolling(5,  min_periods=5).std().values
    vol20 = pd.Series(daily_ret, index=close.index).rolling(20, min_periods=20).std().values
    rows["vol_ratio"] = np.where(vol20 > 0, vol5 / vol20, np.nan)

    ma50 = pd.Series(p, index=close.index).rolling(50,  min_periods=50).mean().values
    ma200= pd.Series(p, index=close.index).rolling(200, min_periods=200).mean().values
    rows["ma50_ratio"]    = np.where(ma50  > 0, p / ma50  - 1.0, np.nan)
    rows["ma200_ratio"]   = np.where(ma200 > 0, p / ma200 - 1.0, np.nan)
    rows["ma50_200_cross"]= np.where((ma50 > 0) & (ma200 > 0), ma50 / ma200 - 1.0, np.nan)

    ret5  = rows["ret5"]
    ret20 = rows["ret20"]
    rows["ret_accel"] = ret5 - ret20  # short-term momentum acceleration

    # RSI-14
    rows["rsi14"] = _rsi14(close).values

    # Distance-to-high / low features
    s_close = pd.Series(p, index=close.index)
    h52  = s_close.rolling(252, min_periods=252).max().values
    l52  = s_close.rolling(252, min_periods=252).min().values
    h3y  = s_close.rolling(756, min_periods=756).max().values
    l3y  = s_close.rolling(756, min_periods=756).min().values
    rows["dist52h"] = np.where(h52 > 0, p / h52 - 1.0, np.nan)
    rows["dist52l"] = np.where(l52 > 0, p / l52 - 1.0, np.nan)
    rows["dist3yh"] = np.where(h3y > 0, p / h3y - 1.0, np.nan)
    rows["dist3yl"] = np.where(l3y > 0, p / l3y - 1.0, np.nan)

    # Forward 1-day log return (label)
    fwd = np.full(n, np.nan)
    fwd[:-1] = log_c[1:] - log_c[:-1]
    rows["_fwd"] = fwd

    df = pd.DataFrame(rows, index=close.index)
    return df.dropna(subset=COURSE_FEATURE_NAMES)
