import numpy as np
import pandas as pd

from stockml.labels.returns import (
    binary_direction_labels,
    multi_horizon_return_labels,
    quantile_return_labels,
    regime_class_labels,
    sequence_return_labels,
)


def test_binary_label_matches_future_close(single_asset):
    sa = single_asset.assign(ticker="AAA")
    out = binary_direction_labels(sa, horizon=1)
    last_idx = out.index[-1]
    assert pd.isna(out.loc[last_idx, "y_binary_h1"])
    nonnull = out["y_binary_h1"].dropna()
    sample = nonnull.index[10]
    sample_pos = out.index.get_loc(sample)
    expected = int(out["close"].iloc[sample_pos + 1] > out["close"].iloc[sample_pos])
    assert int(nonnull.loc[sample]) == expected


def test_multi_horizon_returns_sign_matches(single_asset):
    sa = single_asset.assign(ticker="AAA")
    out = multi_horizon_return_labels(sa, horizons=[1, 5, 20])
    for h in [1, 5, 20]:
        col = f"y_logret_h{h}"
        nonnull = out[col].dropna()
        idx = nonnull.index[10]
        pos = out.index.get_loc(idx)
        future_close = out["close"].iloc[pos + h]
        cur_close = out["close"].iloc[pos]
        assert np.sign(np.log(future_close / cur_close)) == np.sign(nonnull.loc[idx])


def test_quantile_labels_have_expected_columns(single_asset):
    sa = single_asset.assign(ticker="AAA")
    out = quantile_return_labels(sa, horizons=[1, 5])
    assert "y_quantile_h1" in out.columns
    assert "y_quantile_h5" in out.columns


def test_regime_label_has_finite_classes(single_asset):
    sa = single_asset.assign(ticker="AAA")
    out = regime_class_labels(sa, horizon=5)
    nonnull = out["y_regime_h5"].dropna()
    assert nonnull.nunique() >= 2


def test_sequence_label_columns_renamed(single_asset):
    sa = single_asset.assign(ticker="AAA")
    out = sequence_return_labels(sa, horizons=[1, 5])
    assert "y_seq_h1" in out.columns
    assert "y_seq_h5" in out.columns
