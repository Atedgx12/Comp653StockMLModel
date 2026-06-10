"""Filesystem and config IO helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root inferred from this file's location."""
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    """Create the directory at path if missing and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top level YAML structure must be a mapping at {path}")
    return data


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a dictionary to YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
