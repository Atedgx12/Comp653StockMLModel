"""
Sample weighting utilities for temporal models.
Computes per-sample weights that emphasise recent data over older data,
addressing the regime mismatch between a long training history and a
recent out-of-sample test period.
"""
import numpy as np
import pandas as pd
from typing import Union


def exponential_time_weights(
    dates: Union[np.ndarray, pd.Index],
    decay: float = 2.0,
) -> np.ndarray:
    """
    Assign an exponential weight to each sample based on its recency.

    The most recent date always receives weight 1.0.  Older dates receive
    exp(-decay * (1 - relative_rank)) where relative_rank is in [0, 1].

    Parameters
    ----------
    dates  : array of dates for each sample (not required to be sorted)
    decay  : controls how fast older samples are down-weighted.
             0.0  = uniform weights (no decay)
             1.0  = mild decay, oldest sample has weight ~0.37
             2.0  = moderate decay, oldest sample has weight ~0.14
             4.0  = strong decay, oldest sample has weight ~0.02

    Returns
    -------
    weights : float array, shape (n_samples,), values in (0, 1]
              Already normalised so mean = 1.0 (total weight = n_samples).
    """
    if decay == 0.0:
        return np.ones(len(dates))

    dates_arr = np.array(dates, dtype="datetime64[D]").astype(np.int64)
    d_min, d_max = dates_arr.min(), dates_arr.max()
    span = max(d_max - d_min, 1)
    relative_rank = (dates_arr - d_min) / span          # 0 = oldest, 1 = newest
    raw = np.exp(-decay * (1.0 - relative_rank))
    # Normalise so mean weight = 1 (keeps effective LR unchanged)
    return raw / raw.mean()


def recent_mi_weights(
    dates: np.ndarray,
    recent_frac: float = 0.30,
) -> np.ndarray:
    """
    Binary weight: samples in the most recent `recent_frac` of dates
    get weight 2.0, older samples get weight 1.0.
    Useful when you want to double the influence of the recent regime
    without a continuous decay.
    """
    unique_sorted = np.sort(np.unique(dates))
    cutoff = unique_sorted[int(len(unique_sorted) * (1.0 - recent_frac))]
    weights = np.where(dates >= cutoff, 2.0, 1.0)
    return weights / weights.mean()
