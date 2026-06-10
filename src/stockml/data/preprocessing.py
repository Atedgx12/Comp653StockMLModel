"""Cleaning and outlier handling applied before feature engineering.

The course feedback explicitly asked for an explicit treatment of missing
values and outliers, so each routine here is small, named for what it does,
and unit testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def drop_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where reported volume is zero.

    Zero volume rows are common in delisted or thinly traded names and they
    distort the realized volatility and average true range calculations
    because the price did not actually move during the session.
    """
    if "volume" not in df.columns:
        return df
    mask = df["volume"] > 0
    if mask.all():
        return df
    dropped = (~mask).sum()
    logger.info("Dropping %s zero volume rows", int(dropped))
    return df.loc[mask].copy()


def forward_fill(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    limit: int = 1,
) -> pd.DataFrame:
    """Forward fill missing values up to ``limit`` consecutive sessions.

    Used for fundamental columns that are reported less frequently than the
    daily price series. Limiting the fill prevents the model from seeing
    stale fundamentals from an arbitrary number of sessions ago, which would
    leak into a regime change.
    """
    cols = columns or [c for c in df.columns if c not in {"ticker"}]
    df = df.copy()
    df[cols] = df[cols].ffill(limit=limit)
    return df


def winsorize_returns(
    df: pd.DataFrame,
    return_col: str = "log_return_1",
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
) -> pd.DataFrame:
    """Clip extreme returns to the requested quantiles per ticker.

    Returns are computed using adjusted close prices, but data errors,
    suspended-trading reopens, or hard limit moves can still produce
    multi sigma outliers that drive gradient boosting splits and dominate
    sequence model gradients. Per ticker winsorization preserves the cross
    sectional distribution while removing single point pathologies.
    """
    if return_col not in df.columns:
        return df
    df = df.copy()
    if "ticker" in df.columns:
        grouped = df.groupby("ticker", group_keys=False)
        df[return_col] = grouped[return_col].apply(
            lambda s: s.clip(lower=s.quantile(quantile_low), upper=s.quantile(quantile_high))
        )
    else:
        low = df[return_col].quantile(quantile_low)
        high = df[return_col].quantile(quantile_high)
        df[return_col] = df[return_col].clip(lower=low, upper=high)
    return df


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small summary frame of missing value counts per column."""
    counts = df.isna().sum()
    pct = counts / max(len(df), 1)
    return pd.DataFrame({"missing": counts, "pct": pct}).sort_values("missing", ascending=False)


def median_impute(
    df: pd.DataFrame,
    columns: list[str],
    by_ticker: bool = True,
) -> pd.DataFrame:
    """Replace residual missing values with the per ticker median.

    Forward fill is the first defense and handles the dominant pattern of
    missing values. Median imputation handles columns that are missing at
    the very beginning of an asset's history where forward fill has nothing
    to copy from.
    """
    df = df.copy()
    if by_ticker and "ticker" in df.columns:
        for col in columns:
            df[col] = df.groupby("ticker")[col].transform(
                lambda s: s.fillna(s.median())
            )
    else:
        for col in columns:
            med = df[col].median()
            df[col] = df[col].fillna(med)
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df
