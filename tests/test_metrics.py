import numpy as np

from stockml.training.metrics import (
    brier_score,
    classification_report,
    directional_accuracy,
    information_coefficient,
    pinball_loss,
    regression_report,
)


def test_information_coefficient_perfect_rank():
    y = np.arange(100, dtype=float)
    p = y * 2.0 + 1.0
    assert information_coefficient(y, p) > 0.99


def test_directional_accuracy_perfect():
    y = np.array([0.01, -0.02, 0.03, -0.04])
    p = np.array([0.5, -0.5, 0.5, -0.5])
    assert directional_accuracy(y, p) == 1.0


def test_pinball_loss_zero_for_exact():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, quantile=0.5) == 0.0


def test_brier_binary_zero_for_exact():
    y = np.array([1, 0, 1])
    p = np.array([1.0, 0.0, 1.0])
    assert brier_score(y, p) == 0.0


def test_regression_report_keys():
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    p = y + rng.normal(0, 0.1, size=100)
    rep = regression_report(y, p)
    for k in ["rmse", "mae", "r2", "ic", "directional_accuracy"]:
        assert k in rep


def test_classification_report_keys():
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 1, 1, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    rep = classification_report(y, p, y_proba=proba, labels=[0, 1])
    for k in ["accuracy", "balanced_accuracy", "log_loss", "brier"]:
        assert k in rep
