"""Feature pruning: correlation-based and mutual-information-based selection."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def drop_correlated_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop features with absolute pairwise correlation above ``threshold``.

    Returns the reduced frame and the list of retained feature names. The
    selection is greedy: features are sorted by their average absolute
    correlation against the rest, and the most redundant feature is removed
    first until no pair exceeds the threshold.
    """
    feats = list(feature_columns)
    if len(feats) < 2:
        return df, feats
    corr = df[feats].corr().abs()
    np.fill_diagonal(corr.values, 0.0)
    while True:
        max_pair = corr.unstack().sort_values(ascending=False).index[0]
        max_val = corr.loc[max_pair[0], max_pair[1]]
        if max_val < threshold:
            break
        a, b = max_pair
        a_avg = corr[a].mean()
        b_avg = corr[b].mean()
        drop = a if a_avg >= b_avg else b
        feats.remove(drop)
        corr = corr.drop(index=drop, columns=drop)
    out = df.drop(columns=[c for c in feature_columns if c not in feats])
    return out, feats


# ---------------------------------------------------------------------------
# Mutual-information based feature selection (COMP 653 Module 2)
# ---------------------------------------------------------------------------

def _entropy_bins(x: np.ndarray, bins: int = 20) -> float:
    """Marginal entropy H(X) via equal-width histogram."""
    counts, _ = np.histogram(x, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """I(X ; Y) = H(X) + H(Y) - H(X, Y) via 2-D histogram."""
    h_x = _entropy_bins(x, bins)
    h_y = _entropy_bins(y, bins)
    counts2d, _, _ = np.histogram2d(x, y, bins=bins)
    probs2d = counts2d / counts2d.sum()
    probs2d = probs2d[probs2d > 0]
    h_xy = float(-np.sum(probs2d * np.log2(probs2d)))
    return max(0.0, h_x + h_y - h_xy)


def select_features_by_mi(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    k: int = 20,
    bins: int = 20,
    verbose: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Select the top ``k`` features by mutual information with ``y``.

    Implements the Module 2 histogram MI estimator::

        I(X_j ; Y) = H(X_j) + H(Y) - H(X_j, Y)

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    feature_names : sequence of str
    k : int
        Number of features to retain.
    bins : int
        Number of histogram bins for entropy estimation.
    verbose : bool
        Print the top-k MI scores when True.

    Returns
    -------
    X_selected : ndarray of shape (n_samples, k)
    selected_names : list of str
    """
    scores = np.array([
        _mutual_information(X[:, j], y.astype(float), bins)
        for j in range(X.shape[1])
    ])
    order = np.argsort(scores)[::-1]
    top_k = order[:k]
    selected_names = [feature_names[i] for i in top_k]
    if verbose:
        print(f"  Top {k} features by I(X_j ; Y):")
        for i in top_k:
            print(f"    {feature_names[i]:<32s}  I = {scores[i]:.4f} bits")
    return X[:, top_k], selected_names
