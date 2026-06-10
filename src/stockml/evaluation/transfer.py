"""Transfer learning evaluation: pretrain on equities, fine tune on crypto.

The function exposed here is deliberately simple. It takes an already
trained model and a target dataset, optionally calls ``fit`` again on the
target slice as a fine tuning step, and reports the metric delta between
zero shot and fine tuned predictions on the target data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.base import BaseModel
from ..training.metrics import classification_report, regression_report


@dataclass
class TransferReport:
    zero_shot: dict[str, float]
    fine_tuned: dict[str, float]
    n_target_train: int
    n_target_test: int


def _metrics(task: str, y_true, y_pred, y_proba=None) -> dict[str, float]:
    if task == "classification":
        return classification_report(y_true, y_pred, y_proba=y_proba)
    return regression_report(y_true, y_pred)


def transfer_evaluate(
    pretrained_model: BaseModel,
    target_panel: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    task: str,
    finetune_fraction: float = 0.5,
) -> TransferReport:
    df = target_panel[[*feature_columns, label_column]].dropna()
    df = df.sort_index()
    cut = int(len(df) * finetune_fraction)
    train = df.iloc[:cut]
    test = df.iloc[cut:]

    X_test = test[feature_columns].to_numpy()
    y_test = test[label_column].to_numpy()

    zs_pred = pretrained_model.predict(X_test)
    zs_proba = None
    if task == "classification":
        try:
            zs_proba = pretrained_model.predict_proba(X_test)
        except NotImplementedError:
            zs_proba = None
    zero_shot = _metrics(task, y_test, zs_pred, y_proba=zs_proba)

    pretrained_model.fit(
        train[feature_columns].to_numpy(),
        train[label_column].to_numpy(),
        feature_names=feature_columns,
    )
    ft_pred = pretrained_model.predict(X_test)
    ft_proba = None
    if task == "classification":
        try:
            ft_proba = pretrained_model.predict_proba(X_test)
        except NotImplementedError:
            ft_proba = None
    fine_tuned = _metrics(task, y_test, ft_pred, y_proba=ft_proba)

    return TransferReport(
        zero_shot=zero_shot,
        fine_tuned=fine_tuned,
        n_target_train=len(train),
        n_target_test=len(test),
    )


def _ensure_numpy_referenced() -> None:
    _ = np.array([])
