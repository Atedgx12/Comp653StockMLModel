"""Correlation-based feature pruning."""
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
