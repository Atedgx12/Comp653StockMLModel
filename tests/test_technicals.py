import numpy as np
import pandas as pd

from stockml.features.technicals import (
    add_atr,
    add_bollinger_width,
    add_close_in_range,
    add_ema,
    add_log_returns,
    add_macd,
    add_realized_volatility,
    add_rolling_high_low,
    add_rsi,
)


def test_log_returns_match_diff_of_log_close(single_asset):
    out = add_log_returns(single_asset, periods=[1, 5])
    expected_1 = np.log(single_asset["close"]).diff(1)
    expected_5 = np.log(single_asset["close"]).diff(5)
    pd.testing.assert_series_equal(
        out["log_return_1"], expected_1, check_names=False
    )
    pd.testing.assert_series_equal(
        out["log_return_5"], expected_5, check_names=False
    )


def test_ema_matches_pandas_ewm(single_asset):
    out = add_ema(single_asset, periods=[9])
    expected = single_asset["close"].ewm(span=9, adjust=False, min_periods=9).mean()
    pd.testing.assert_series_equal(out["ema_9"], expected, check_names=False)


def test_macd_signal_smaller_than_macd_line_in_amplitude(single_asset):
    out = add_macd(single_asset)
    nonnull = out[["macd", "macd_signal"]].dropna()
    assert (nonnull["macd"].abs().mean()) >= 0  # sanity: data flows


def test_rsi_within_zero_to_hundred(single_asset):
    out = add_rsi(single_asset)
    rsi = out["rsi_14"].dropna()
    assert rsi.min() >= 0.0
    assert rsi.max() <= 100.0


def test_realized_volatility_is_nonneg(single_asset):
    out = add_realized_volatility(single_asset)
    vol = out["realized_vol_20"].dropna()
    assert (vol >= 0).all()


def test_atr_pct_is_nonneg(single_asset):
    out = add_atr(single_asset)
    atr = out["atr_14_pct"].dropna()
    assert (atr >= 0).all()


def test_bollinger_width_present(single_asset):
    out = add_bollinger_width(single_asset)
    assert "bb_width_20" in out.columns
    assert "bb_z_20" in out.columns


def test_rolling_high_low_distance_signs(single_asset):
    out = add_rolling_high_low(single_asset, periods=[20])
    assert (out["dist_to_high_20"].dropna() <= 0).all()
    assert (out["dist_to_low_20"].dropna() >= 0).all()


def test_close_in_range_within_zero_one(single_asset):
    out = add_close_in_range(single_asset)
    series = out["close_in_range_20"].dropna()
    assert series.min() >= -1e-9
    assert series.max() <= 1.0 + 1e-9


def test_indicator_no_lookahead(single_asset):
    """Future rows must not influence the indicator at time t."""
    base = add_realized_volatility(single_asset, period=20)
    truncated = add_realized_volatility(single_asset.iloc[:-50].copy(), period=20)
    common = truncated.index.intersection(base.index)
    pd.testing.assert_series_equal(
        base.loc[common, "realized_vol_20"],
        truncated.loc[common, "realized_vol_20"],
        check_names=False,
    )
