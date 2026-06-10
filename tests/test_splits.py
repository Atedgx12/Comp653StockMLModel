import pandas as pd

from stockml.splits.walk_forward import expanding_walk_forward_folds, purge_and_embargo


def test_walk_forward_folds_are_disjoint_in_test():
    idx = pd.bdate_range("2010-01-01", "2024-12-31")
    folds = list(expanding_walk_forward_folds(idx, initial_train_years=4, validation_years=1, test_years=1, step_years=1))
    test_ranges = [(f.test_start, f.test_end) for f in folds]
    for i in range(len(test_ranges) - 1):
        assert test_ranges[i][1] <= test_ranges[i + 1][0]


def test_walk_forward_train_precedes_test():
    idx = pd.bdate_range("2010-01-01", "2024-12-31")
    folds = list(expanding_walk_forward_folds(idx))
    for f in folds:
        assert f.train_end <= f.val_start <= f.val_end <= f.test_start <= f.test_end


def test_purge_and_embargo_removes_overlapping_rows():
    idx = pd.bdate_range("2018-01-01", "2018-06-01")
    test_start = pd.Timestamp("2018-04-01")
    test_end = pd.Timestamp("2018-05-01")
    purged = purge_and_embargo(idx, test_start, test_end, purge_days=20, embargo_days=20)
    before = purged < test_start - pd.Timedelta(days=20)
    after = purged > test_end + pd.Timedelta(days=20)
    assert (before | after).all()
    inside = (purged >= test_start - pd.Timedelta(days=20)) & (
        purged <= test_end + pd.Timedelta(days=20)
    )
    assert not inside.any()
