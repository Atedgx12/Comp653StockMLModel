"""Linear and logistic baselines."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet, LogisticRegression

from .base import BaseModel


class LinearRegressor(BaseModel):
    """Elastic net regressor wrapped in the project's BaseModel interface."""

    name = "linear"

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        max_iter: int = 5000,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self._estimator = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            random_state=42,
        )
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        feature_names=None,
    ) -> "LinearRegressor":
        self._feature_names = list(feature_names) if feature_names is not None else None
        self._estimator.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def feature_importances(self) -> dict[str, float] | None:
        if self._feature_names is None:
            return None
        return {n: float(c) for n, c in zip(self._feature_names, self._estimator.coef_, strict=False)}


class LogisticClassifier(BaseModel):
    """Multinomial logistic regression baseline."""

    name = "logistic"

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 5000,
        **_: object,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self._estimator = LogisticRegression(
            C=1.0 / max(alpha, 1e-8),
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            multi_class="auto",
            random_state=42,
        )
        self._feature_names: list[str] | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        self._feature_names = list(feature_names) if feature_names is not None else None
        self._estimator.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)

    def feature_importances(self) -> dict[str, float] | None:
        if self._feature_names is None or self._estimator.coef_.ndim != 2:
            return None
        avg = np.mean(np.abs(self._estimator.coef_), axis=0)
        return {n: float(c) for n, c in zip(self._feature_names, avg, strict=False)}
