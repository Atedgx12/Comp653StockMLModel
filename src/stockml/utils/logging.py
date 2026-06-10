"""Logging configuration helpers."""
from __future__ import annotations

import logging
import sys
from typing import Final

_DEFAULT_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging with a single stream handler.

    Idempotent: if a stream handler is already attached the call is a no op so
    test runners do not accumulate duplicate handlers across modules.
    """
    if isinstance(level, str):
        level = level.upper()
    root = logging.getLogger()
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if has_stream:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module level logger with a stable namespace."""
    return logging.getLogger(name)
