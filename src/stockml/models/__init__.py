"""Model implementations grouped by family.

Each model exposes the ``BaseModel`` interface so the training driver can
swap families behind a single config switch.
"""
from .base import BaseModel
from .linear import LinearRegressor, LogisticClassifier
from .neural import UnifiedCourseNetwork
from .online_linear import OnlineLinearRegressor

__all__ = [
    "BaseModel",
    "LightGBMClassifier",
    "LightGBMRegressor",
    "LinearRegressor",
    "LogisticClassifier",
    "OnlineLinearRegressor",
    "UnifiedCourseNetwork",
]


def build_model(model_cfg: dict, task: str) -> BaseModel:
    """Factory that materializes a model instance from a config dict."""
    family = model_cfg["family"]
    params = dict(model_cfg.get("params", {}))
    if family == "linear":
        return LogisticClassifier(**params) if task == "classification" else LinearRegressor(**params)
    if family == "gbm":
        from .lightgbm_models import LightGBMClassifier, LightGBMRegressor

        if task == "classification":
            return LightGBMClassifier(params)
        return LightGBMRegressor(params)
    if family == "online_linear":
        return OnlineLinearRegressor(**params)
    if family == "unified_course_network":
        return UnifiedCourseNetwork(**params)
    if family in {"tcn", "transformer", "lstm"}:
        from .neural import build_torch_model

        return build_torch_model(family, params, task=task)
    raise ValueError(f"Unknown model family: {family}")


def __getattr__(name: str):
    """Load optional LightGBM wrappers only when they are requested."""
    if name in {"LightGBMClassifier", "LightGBMRegressor"}:
        from .lightgbm_models import LightGBMClassifier, LightGBMRegressor

        return {
            "LightGBMClassifier": LightGBMClassifier,
            "LightGBMRegressor": LightGBMRegressor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
