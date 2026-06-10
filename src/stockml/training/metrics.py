"""Metric implementations."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predicted and realized returns."""
    if len(y_true) < 2:
        return float("nan")
    rt = np.argsort(np.argsort(y_true))
    rp = np.argsort(np.argsort(y_pred))
    return float(np.corrcoef(rt, rp)[0, 1])


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of forecasts that get the sign of the realized return right."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Quantile (pinball) loss at the requested quantile level."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if y_proba.ndim == 1:
        return float(np.mean((y_proba - y_true) ** 2))
    n = len(y_true)
    onehot = np.zeros_like(y_proba)
    onehot[np.arange(n), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "ic": information_coefficient(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: Iterable[int] | None = None,
) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            labels_arg = list(labels) if labels else None
            out["log_loss"] = float(log_loss(y_true, y_proba, labels=labels_arg))
        except ValueError:
            out["log_loss"] = float("nan")
        out["brier"] = brier_score(np.asarray(y_true), y_proba)
    return out
