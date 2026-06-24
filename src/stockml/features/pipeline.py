"""Composable feature pipeline driven by the standard_technicals config."""
from __future__ import annotations

from typing import Any

import pandas as pd

from ..utils.logging import get_logger
from .regime import add_regime_features
from .technicals import (
    add_atr,
    add_bollinger_width,
    add_close_in_range,
    add_ema,
    add_log_returns,
    add_macd,
    add_rate_of_change,
    add_realized_volatility,
    add_rolling_high_low,
    add_rsi,
)

logger = get_logger(__name__)


def _per_asset(df: pd.DataFrame, fn, *args, **kwargs) -> pd.DataFrame:
    if "ticker" not in df.columns:
        return fn(df, *args, **kwargs)
    pieces: list[pd.DataFrame] = []
    for ticker, group in df.groupby("ticker", group_keys=False, sort=False):
        sub = group.drop(columns="ticker")
        out = fn(sub, *args, **kwargs)
        out = out.assign(ticker=ticker)
        pieces.append(out)
    return pd.concat(pieces)


def build_features(
    panel: pd.DataFrame,
    feature_cfg: dict[str, Any],
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the configured feature pipeline over a multi ticker panel."""
    groups = feature_cfg.get("groups", {})
    out = panel.copy()

    log_periods = sorted({1, *groups.get("trend_momentum", {}).get("roc_periods", [])})
    out = _per_asset(out, add_log_returns, periods=log_periods)

    tm = groups.get("trend_momentum", {})
    if tm.get("enabled", True):
        out = _per_asset(out, add_ema, periods=tm.get("ema_periods", [9, 21, 50, 200]))
        macd = tm.get("macd", {"fast": 12, "slow": 26, "signal": 9})
        out = _per_asset(out, add_macd, fast=macd["fast"], slow=macd["slow"], signal=macd["signal"])
        out = _per_asset(out, add_rsi, period=tm.get("rsi_period", 14))
        out = _per_asset(out, add_rate_of_change, periods=tm.get("roc_periods", [5, 10, 20]))

    vol = groups.get("volatility", {})
    if vol.get("enabled", True):
        out = _per_asset(out, add_realized_volatility, period=vol.get("realized_vol_period", 20))
        out = _per_asset(out, add_atr, period=vol.get("atr_period", 14))
        bb = vol.get("bollinger", {"period": 20, "num_std": 2.0})
        out = _per_asset(out, add_bollinger_width, period=bb["period"], num_std=bb["num_std"])

    ps = groups.get("price_structure", {})
    if ps.get("enabled", True):
        out = _per_asset(
            out, add_rolling_high_low, periods=ps.get("rolling_high_low_periods", [20, 60, 252])
        )
        out = _per_asset(out, add_close_in_range, period=ps.get("close_in_range_period", 20))

    regime = groups.get("regime", {})
    if regime.get("enabled", True) and market is not None:
        out = add_regime_features(
            out,
            market=market,
            realized_vol_period=regime.get("market_realized_vol_period", 20),
            vol_buckets=tuple(regime.get("vol_buckets", [0.10, 0.20, 0.40])),
        )

    return out


# ---------------------------------------------------------------------------
# COMP 653 cross-sectional pipeline
# ---------------------------------------------------------------------------

from .technicals import COURSE_FEATURE_NAMES, build_course_features  # noqa: E402


def build_cross_sectional_features(
    close: pd.DataFrame,
    min_ticker_len: int = 756,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the 32-feature cross-sectional panel from a wide close-price frame.

    Parameters
    ----------
    close : pd.DataFrame
        Wide adjusted-close prices: rows = dates (sorted), columns = tickers.
    min_ticker_len : int
        Minimum price history per ticker (default 756 = ~3 years).

    Returns
    -------
    features : pd.DataFrame
        Stacked long frame with columns [*COURSE_FEATURE_NAMES, "_fwd"] plus
        cross-sectionally ranked versions of all feature columns (suffix
        ``_rank``).  Index is (date, ticker).
    feature_names : list of str
        Names of the ranked feature columns.
    """
    pieces: list[pd.DataFrame] = []
    for ticker in close.columns:
        df = build_course_features(close[ticker].dropna(), min_len=min_ticker_len)
        if df is None:
            continue
        df["ticker"] = ticker
        pieces.append(df)

    if not pieces:
        raise RuntimeError("No tickers had enough history")

    panel = pd.concat(pieces).sort_index()

    # Cross-sectional rank (percentile) per date
    ranked_names = [f"{c}_rank" for c in COURSE_FEATURE_NAMES]
    panel[ranked_names] = (
        panel.groupby(level=0)[COURSE_FEATURE_NAMES]
        .rank(pct=True)
    )
    return panel, ranked_names
