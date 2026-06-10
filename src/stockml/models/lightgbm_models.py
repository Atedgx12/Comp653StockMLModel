"""LightGBM regression and classification wrappers."""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

from .base import BaseModel


class _LightGBMBase(BaseModel):
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = dict(params)
        self.num_iterations = int(self.params.pop("num_iterations", 1000))
        self.early_stopping_rounds = int(self.params.pop("early_stopping_rounds", 100))
        self._booster: lgb.Booster | None = None
        self._feature_names: list[str] | None = None

    def feature_importances(self) -> dict[str, float] | None:
        if self._booster is None or self._feature_names is None:
            return None
        importance = self._booster.feature_importance(importance_type="gain")
        return {n: float(v) for n, v in zip(self._feature_names, importance, strict=False)}


class LightGBMRegressor(_LightGBMBase):
    """Gradient boosting regressor with early stopping."""

    name = "lightgbm_regressor"

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        self._feature_names = list(feature_names) if feature_names is not None else None
        train_set = lgb.Dataset(X_train, label=y_train, feature_name=self._feature_names)
        valid_sets = [train_set]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(
                X_val, label=y_val, reference=train_set, feature_name=self._feature_names
            )
            valid_sets.append(val_set)
            valid_names.append("val")

        params = dict(self.params)
        params.setdefault("objective", "regression")
        params.setdefault("metric", "rmse")

        callbacks: list = []
        if X_val is not None and self.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))

        self._booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_iterations,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X):
        if self._booster is None:
            raise RuntimeError("Model has not been fit")
        return self._booster.predict(X)


class LightGBMClassifier(_LightGBMBase):
    """Gradient boosting classifier with multiclass softmax."""

    name = "lightgbm_classifier"

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        self._feature_names = list(feature_names) if feature_names is not None else None
        n_classes = int(np.max(y_train) - np.min(y_train) + 1)
        params = dict(self.params)
        if n_classes <= 2:
            params.setdefault("objective", "binary")
            params.setdefault("metric", "binary_logloss")
        else:
            params.setdefault("objective", "multiclass")
            params.setdefault("metric", "multi_logloss")
            params["num_class"] = n_classes
        train_set = lgb.Dataset(X_train, label=y_train, feature_name=self._feature_names)
        valid_sets = [train_set]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(
                X_val, label=y_val, reference=train_set, feature_name=self._feature_names
            )
            valid_sets.append(val_set)
            valid_names.append("val")

        callbacks: list = []
        if X_val is not None and self.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))

        self._booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_iterations,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self._n_classes = n_classes
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        if proba.ndim == 1:
            return (proba >= 0.5).astype(int)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        if self._booster is None:
            raise RuntimeError("Model has not been fit")
        return self._booster.predict(X)
