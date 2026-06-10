"""Forward looking label generators.

Each function emits labels aligned to the prediction time. The future
information used to compute the label is sourced exclusively from rows that
follow the prediction time, so a model trained on these labels learns to
forecast and not to read the present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _per_asset_apply(panel: pd.DataFrame, fn) -> pd.DataFrame:
    if "ticker" not in panel.columns:
        return fn(panel)
    pieces: list[pd.DataFrame] = []
    for ticker, group in panel.groupby("ticker", group_keys=False, sort=False):
        sub = group.drop(columns="ticker")
        out = fn(sub)
        out = out.assign(ticker=ticker)
        pieces.append(out)
    return pd.concat(pieces)


def binary_direction_labels(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Sanity check binary up/down label retained from the original proposal."""

    def _one(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        future = df["close"].shift(-horizon)
        label = (future > df["close"]).astype("Int64")
        label[future.isna()] = pd.NA
        df[f"y_binary_h{horizon}"] = label
        return df

    return _per_asset_apply(panel, _one)


def multi_horizon_return_labels(
    panel: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame:
    """Signed log returns for each requested horizon.

    Provides strictly more information than a binary sign label and is the
    primary regression target for the gradient boosting and linear families.
    """

    def _one(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        log_close = np.log(df["close"])
        for h in horizons:
            df[f"y_logret_h{h}"] = log_close.shift(-h) - log_close
        return df

    return _per_asset_apply(panel, _one)


def quantile_return_labels(
    panel: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame:
    """Realized future returns serve as the targets for quantile heads.

    A quantile loss with three quantile heads is fit later in the trainer.
    The label itself is the same realized signed log return; the quantile
    structure lives in the loss function.
    """
    return multi_horizon_return_labels(panel, horizons=horizons).rename(
        columns={f"y_logret_h{h}": f"y_quantile_h{h}" for h in horizons}
    )


def regime_class_labels(
    panel: pd.DataFrame,
    horizon: int = 5,
    direction_thresholds: tuple[float, float] = (-0.005, 0.005),
    volatility_buckets: tuple[float, ...] = (0.15, 0.30),
) -> pd.DataFrame:
    """Combine direction and realized volatility into a small class label.

    The result is a categorical target that retains directional information
    while also expressing whether the realized environment was calm or
    turbulent. This is the richer classification target promised in the
    revised proposal.
    """
    lower, upper = direction_thresholds

    def _one(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        log_close = np.log(df["close"])
        future = log_close.shift(-horizon) - log_close
        direction = np.where(future > upper, 2, np.where(future < lower, 0, 1))
        realized_vol = (
            df["close"].pct_change().rolling(horizon, min_periods=horizon).std()
            * np.sqrt(252.0)
        ).shift(-horizon)
        bins = [-np.inf, *list(volatility_buckets), np.inf]
        vol_bucket = pd.cut(realized_vol, bins=bins, labels=False)
        df[f"y_regime_h{horizon}"] = direction * len(volatility_buckets) + vol_bucket
        return df

    return _per_asset_apply(panel, _one)


def sequence_return_labels(
    panel: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Same as ``multi_horizon_return_labels`` but with a sequence aware naming.

    The trainer for sequence models stacks the inputs into a window of length
    ``input_window`` and predicts the same forward log returns. Reusing the
    multi horizon target keeps the metric definitions consistent across model
    families.
    """
    return multi_horizon_return_labels(panel, horizons=horizons).rename(
        columns={f"y_logret_h{h}": f"y_seq_h{h}" for h in horizons}
    )
