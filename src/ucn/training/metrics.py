"""Evaluation metrics — no sklearn dependency."""
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true.astype(int) == y_pred.astype(int)).mean())


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via Wilcoxon rank-sum — O(n log n), no external deps."""
    y_true = y_true.astype(int)
    n_pos  = int(y_true.sum())
    n_neg  = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order    = np.argsort(y_score)
    ranks    = np.arange(1, len(y_true) + 1, dtype=np.float64)
    rank_sum = ranks[y_true[order] == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
