"""Label generators for the four task formulations."""
from .returns import (
    binary_direction_labels,
    multi_horizon_return_labels,
    quantile_return_labels,
    regime_class_labels,
    sequence_return_labels,
)

__all__ = [
    "binary_direction_labels",
    "multi_horizon_return_labels",
    "quantile_return_labels",
    "regime_class_labels",
    "sequence_return_labels",
]
