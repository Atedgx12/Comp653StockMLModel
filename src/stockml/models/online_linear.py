"""Online linear regressor with periodic refit on a rolling window.

This model directly addresses the nonstationarity challenge raised in the
proposal feedback. Instead of training once on the full historical window,
it refits at fixed intervals using only the most recent ``rolling_window``
years of data, which lets it forget regimes that no longer apply.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .base import BaseModel


class OnlineLinearRegressor(BaseModel):
    """Rolling refit ridge regressor."""

    name = "online_linear"

    def __init__(
        self,
        alpha: float = 1.0,
        refit_every_days: int = 21,
        rolling_window_years: int = 3,
        use_regime_features: bool = True,
    ) -> None:
        self.alpha = alpha
        self.refit_every_days = refit_every_days
        self.rolling_window_years = rolling_window_years
        self.use_regime_features = use_regime_features
        self._estimator = Ridge(alpha=alpha, fit_intercept=True)
        self._feature_names: list[str] | None = None
        self._fit_count: int = 0

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        self._feature_names = list(feature_names) if feature_names is not None else None
        self._estimator.fit(X_train, y_train)
        self._fit_count = 1
        return self

    def partial_refit(self, X_window: np.ndarray, y_window: np.ndarray) -> None:
        """Refit on the most recent rolling window during evaluation."""
        self._estimator.fit(X_window, y_window)
        self._fit_count += 1

    def predict(self, X):
        return self._estimator.predict(X)

    def feature_importances(self) -> dict[str, float] | None:
        if self._feature_names is None:
            return None
        return {
            n: float(c)
            for n, c in zip(self._feature_names, self._estimator.coef_, strict=False)
        }
