"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_panel() -> pd.DataFrame:
    """Two ticker synthetic panel suitable for feature and label tests."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    panels = []
    for ticker in ["AAA", "BBB"]:
        rets = rng.normal(0.0005, 0.011, len(dates))
        close = 50.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.001, len(dates))),
                "high": close * (1 + np.abs(rng.normal(0, 0.004, len(dates)))),
                "low": close * (1 - np.abs(rng.normal(0, 0.004, len(dates)))),
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, len(dates)),
                "ticker": ticker,
            },
            index=dates,
        )
        df.index.name = "date"
        panels.append(df)
    return pd.concat(panels)


@pytest.fixture()
def single_asset(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    return (
        synthetic_panel[synthetic_panel["ticker"] == "AAA"]
        .drop(columns=["ticker"])
        .copy()
    )
