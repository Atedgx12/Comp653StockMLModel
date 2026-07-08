"""
Information Theory utilities — Module 2, Lec 2-6.
H(X), I(X;Y) via histogram-based entropy estimation.
"""
import numpy as np
from typing import List, Tuple


def entropy(x: np.ndarray, bins: int = 20) -> float:
    counts, _ = np.histogram(x, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    hx  = entropy(x, bins)
    hy  = entropy(y, bins)
    counts2d, _, _ = np.histogram2d(x, y, bins=bins)
    p2d = counts2d / counts2d.sum()
    p2d_pos = p2d[p2d > 0]
    hxy = float(-np.sum(p2d_pos * np.log2(p2d_pos)))
    return max(0.0, hx + hy - hxy)


def select_features_by_mi(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    k: int = 20,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Rank features by I(X_j ; Y) and return top-k names and their MI scores.

    Returns
    -------
    selected : list of str   top-k feature names
    scores   : list of float corresponding MI values (bits)
    """
    if verbose:
        print(f"\n[MI Feature Selection] Computing MI for {X.shape[1]} features ...",
              flush=True)

    scored: List[Tuple[str, float]] = []
    for j in range(X.shape[1]):
        mi = mutual_information(X[:, j], y.astype(float))
        scored.append((feature_names[j], mi))

    scored.sort(key=lambda t: t[1], reverse=True)

    if verbose:
        print("  Top features by I(X_j ; Y):")
        for name, mi in scored[:10]:
            print(f"    {name:30s}  I = {mi:.4f} bits")

    top = scored[:k]
    return [n for n, _ in top], [s for _, s in top]
