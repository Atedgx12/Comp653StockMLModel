"""Common model interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class BaseModel(ABC):
    """Minimal interface every learner in the project must satisfy."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> BaseModel:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - not all models
        raise NotImplementedError(
            f"{type(self).__name__} does not implement predict_proba"
        )

    def feature_importances(self) -> dict[str, float] | None:
        return None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        return joblib.load(path)

    def get_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
