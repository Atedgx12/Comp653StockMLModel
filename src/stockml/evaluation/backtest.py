"""Backtest summary statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Cumulative equity curve from a per period return series."""
    return initial_value * (1.0 + returns.fillna(0.0)).cumprod()


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio with a zero risk free rate."""
    r = returns.dropna()
    if r.std() == 0 or len(r) == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())


def long_short_returns(
    pred: pd.DataFrame,
    realized_return_column: str = "y_logret_h1",
    quantile: float = 0.2,
) -> pd.Series:
    """Form a daily long short portfolio from cross sectional predictions."""
    if "ticker" not in pred.columns:
        return pd.Series(dtype=float)
    daily: list[float] = []
    dates: list[pd.Timestamp] = []
    for date, group in pred.groupby(pred.index):
        if len(group) < 5:
            continue
        long_cut = group["prediction"].quantile(1.0 - quantile)
        short_cut = group["prediction"].quantile(quantile)
        long_leg = group.loc[group["prediction"] >= long_cut, realized_return_column].mean()
        short_leg = group.loc[group["prediction"] <= short_cut, realized_return_column].mean()
        daily.append(long_leg - short_leg)
        dates.append(date)
    return pd.Series(daily, index=pd.DatetimeIndex(dates), name="ls_return")
