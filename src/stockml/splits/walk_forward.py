"""Walk forward folds with embargo and purge.

Walk forward validation evaluates a model on data strictly later than its
training set, mimicking how the model would actually be deployed in
production. Random shuffle splits leak information across time and inflate
performance estimates, especially with the autocorrelated return series in
finance.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """A single walk forward fold."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def name(self) -> str:
        return f"{self.test_start.date()}_to_{self.test_end.date()}"


def expanding_walk_forward_folds(
    index: pd.DatetimeIndex,
    initial_train_years: int = 5,
    validation_years: int = 1,
    test_years: int = 1,
    step_years: int = 1,
) -> Iterator[Fold]:
    """Generate expanding walk forward folds covering ``index``.

    The training window grows over time while the validation and test windows
    slide forward. Both validation and test ranges sit strictly later than
    the training window.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    index = index.sort_values().unique()
    if len(index) == 0:
        return
    earliest = index[0]
    latest = index[-1]

    one_year = pd.DateOffset(years=1)
    train_start = pd.Timestamp(earliest)
    train_end = train_start + initial_train_years * one_year
    while True:
        val_start = train_end
        val_end = val_start + validation_years * one_year
        test_start = val_end
        test_end = test_start + test_years * one_year
        if test_end > pd.Timestamp(latest) + pd.Timedelta(days=1):
            return
        yield Fold(
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )
        train_end = train_end + step_years * one_year


def purge_and_embargo(
    train_index: pd.DatetimeIndex,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    purge_days: int = 20,
    embargo_days: int = 20,
) -> pd.DatetimeIndex:
    """Drop training rows that overlap the test horizon or the embargo gap.

    Purging removes training observations whose forward looking labels reach
    into the test window. The embargo removes training rows immediately after
    the test window so the next fold cannot peek at residual autocorrelation.
    """
    purge_start = test_start - pd.Timedelta(days=purge_days)
    embargo_end = test_end + pd.Timedelta(days=embargo_days)
    mask = (train_index < purge_start) | (train_index > embargo_end)
    return train_index[mask]
