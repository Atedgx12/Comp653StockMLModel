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


def roc_auc_wilcoxon(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary ROC-AUC via the Wilcoxon rank-sum statistic.  O(n log n).

    Equivalent to sklearn ``roc_auc_score`` for binary labels but avoids the
    trapezoidal approximation overhead.  The relationship is::

        AUC = (sum of ranks of positives - n1*(n1+1)/2) / (n0 * n1)

    where ``n1 = sum(y_true == 1)`` and ``n0 = sum(y_true == 0)``.

    Parameters
    ----------
    y_true  : array-like of 0/1 labels
    y_score : array-like of continuous predicted scores

    Returns
    -------
    float in [0, 1]
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    # Average ranks for ties
    unique_scores, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        avg_ranks = np.zeros(len(unique_scores))
        for i, s in enumerate(unique_scores):
            mask = y_score == s
            avg_ranks[i] = ranks[mask].mean()
        ranks = avg_ranks[inverse]
    u_stat = float(ranks[y_true == 1].sum()) - n1 * (n1 + 1) / 2.0
    return u_stat / (n0 * n1)


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
            out["log_loss"] = float(log_loss(y_true, y_proba, labels=list(labels) if labels else None))
        except ValueError:
            out["log_loss"] = float("nan")
        out["brier"] = brier_score(np.asarray(y_true), y_proba)
    return out
