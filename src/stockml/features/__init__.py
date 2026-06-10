"""Feature engineering: technical indicators and regime features."""
from .technicals import (
    add_atr,
    add_bollinger_width,
    add_ema,
    add_log_returns,
    add_macd,
    add_realized_volatility,
    add_rate_of_change,
    add_rsi,
)
from .regime import add_regime_features
from .pipeline import build_features
from .pruning import drop_correlated_features

__all__ = [
    "add_log_returns",
    "add_ema",
    "add_macd",
    "add_rsi",
    "add_realized_volatility",
    "add_atr",
    "add_bollinger_width",
    "add_rate_of_change",
    "add_regime_features",
    "build_features",
    "drop_correlated_features",
]
