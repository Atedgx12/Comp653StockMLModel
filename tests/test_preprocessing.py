import numpy as np
import pandas as pd

from stockml.data.preprocessing import (
    drop_zero_volume,
    forward_fill,
    median_impute,
    winsorize_returns,
)


def test_drop_zero_volume_removes_zero_rows():
    df = pd.DataFrame(
        {"close": [1, 2, 3], "volume": [0, 100, 0]},
        index=pd.bdate_range("2020-01-01", periods=3),
    )
    df.index.name = "date"
    out = drop_zero_volume(df)
    assert len(out) == 1
    assert out["volume"].iloc[0] == 100


def test_forward_fill_respects_limit():
    df = pd.DataFrame(
        {"a": [1.0, np.nan, np.nan, np.nan, 5.0]},
        index=pd.bdate_range("2020-01-01", periods=5),
    )
    df.index.name = "date"
    out = forward_fill(df, columns=["a"], limit=1)
    assert out["a"].iloc[0] == 1.0
    assert out["a"].iloc[1] == 1.0  # one fill forward
    assert pd.isna(out["a"].iloc[2])  # limit reached
    assert pd.isna(out["a"].iloc[3])
    assert out["a"].iloc[4] == 5.0


def test_winsorize_returns_clips_extremes_per_ticker():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 1000)
    rets[0] = 5.0
    rets[1] = -5.0
    df = pd.DataFrame(
        {"log_return_1": rets, "ticker": "AAA"},
        index=pd.bdate_range("2018-01-01", periods=1000),
    )
    df.index.name = "date"
    out = winsorize_returns(df, return_col="log_return_1", quantile_low=0.005, quantile_high=0.995)
    assert out["log_return_1"].max() < 5.0
    assert out["log_return_1"].min() > -5.0


def test_median_impute_replaces_nans():
    df = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, np.nan], "ticker": ["X", "X", "X", "X"]},
        index=pd.bdate_range("2020-01-01", periods=4),
    )
    df.index.name = "date"
    out = median_impute(df, columns=["a"])
    assert out["a"].isna().sum() == 0
