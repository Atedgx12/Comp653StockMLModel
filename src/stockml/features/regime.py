"""Regime features summarizing the broad market state.

These features address the nonstationarity challenge by giving the model a
direct signal of whether the current environment looks like a trending bull,
a high volatility correction, or something else. Without explicit regime
inputs the model only sees per asset technicals and cannot easily share
information about the macro state across assets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _market_log_returns(market: pd.DataFrame) -> pd.Series:
    return np.log(market["close"]).diff()


def add_regime_features(
    panel: pd.DataFrame,
    market: pd.DataFrame | None,
    realized_vol_period: int = 20,
    vol_buckets: tuple[float, ...] = (0.10, 0.20, 0.40),
) -> pd.DataFrame:
    """Append market wide volatility and trend features to every asset row.

    The market frame is the broad index proxy (for example SPY) loaded with
    the same ingestion path. When ``market`` is ``None`` the function returns
    the input unchanged so unit tests for asset level features stay isolated.
    """
    if market is None or market.empty:
        return panel
    panel = panel.copy()

    if not isinstance(market.index, pd.DatetimeIndex):
        raise TypeError("market frame must be indexed by date")
    market_returns = _market_log_returns(market)
    market_vol = market_returns.rolling(realized_vol_period, min_periods=realized_vol_period).std(
        ddof=0
    ) * np.sqrt(252.0)
    market_trend = (
        market["close"].rolling(50, min_periods=50).mean()
        / market["close"].rolling(200, min_periods=200).mean()
        - 1.0
    )

    bucket_edges = list(vol_buckets)
    market_vol_bucket = pd.cut(
        market_vol,
        bins=[-np.inf, *bucket_edges, np.inf],
        labels=False,
    ).astype("float")

    regime = pd.DataFrame(
        {
            "market_realized_vol": market_vol,
            "market_trend_50_200": market_trend,
            "market_vol_bucket": market_vol_bucket,
        }
    )

    panel = panel.merge(regime, left_index=True, right_index=True, how="left")
    panel[regime.columns] = panel[regime.columns].ffill()
    return panel
