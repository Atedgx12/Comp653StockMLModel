"""Feature engineering: technical indicators and regime features."""
from .pipeline import build_features
from .pruning import drop_correlated_features
from .regime import add_regime_features
from .technicals import (
    add_atr,
    add_bollinger_width,
    add_ema,
    add_log_returns,
    add_macd,
    add_rate_of_change,
    add_realized_volatility,
    add_rsi,
)

__all__ = [
    "add_atr",
    "add_bollinger_width",
    "add_ema",
    "add_log_returns",
    "add_macd",
    "add_rate_of_change",
    "add_realized_volatility",
    "add_regime_features",
    "add_rsi",
    "build_features",
    "drop_correlated_features",
]
