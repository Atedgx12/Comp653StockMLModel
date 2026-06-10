"""Model implementations grouped by family.

Each model exposes the ``BaseModel`` interface so the training driver can
swap families behind a single config switch.
"""
from .base import BaseModel
from .lightgbm_models import LightGBMClassifier, LightGBMRegressor
from .linear import LinearRegressor, LogisticClassifier
from .online_linear import OnlineLinearRegressor

__all__ = [
    "BaseModel",
    "LightGBMClassifier",
    "LightGBMRegressor",
    "LinearRegressor",
    "LogisticClassifier",
    "OnlineLinearRegressor",
]


def build_model(model_cfg: dict, task: str) -> BaseModel:
    """Factory that materializes a model instance from a config dict."""
    family = model_cfg["family"]
    params = dict(model_cfg.get("params", {}))
    if family == "linear":
        if task == "classification":
            return LogisticClassifier(**params)
        return LinearRegressor(**params)
    if family == "gbm":
        if task == "classification":
            return LightGBMClassifier(params)
        return LightGBMRegressor(params)
    if family == "online_linear":
        return OnlineLinearRegressor(**params)
    if family in {"tcn", "transformer", "lstm"}:
        from .neural import build_torch_model  # noqa: PLC0415

        return build_torch_model(family, params, task=task)
    raise ValueError(f"Unknown model family: {family}")
