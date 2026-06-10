"""Project command line interface.

The CLI keeps things simple: one subcommand to run the demo synthetic
pipeline end to end, one to print the resolved Hydra config, and one to
print the package version. Real training jobs are run from notebooks or
the dedicated training script which composes the same building blocks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data.preprocessing import drop_zero_volume, winsorize_returns
from .features.pipeline import build_features
from .labels.returns import multi_horizon_return_labels
from .models import build_model
from .training.trainer import train_walk_forward
from .utils.io import read_yaml
from .utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    cfg_root = Path(__file__).resolve().parents[2] / "configs"
    feature_cfg = read_yaml(cfg_root / "features" / "standard_technicals.yaml")
    splits_cfg = read_yaml(cfg_root / "splits" / "walk_forward.yaml")
    model_cfg = read_yaml(cfg_root / "model" / "lightgbm.yaml")

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010-01-01", "2024-12-31")
    panels = []
    for ticker in ["AAA", "BBB", "CCC"]:
        rets = rng.normal(0.0005, 0.012, len(dates))
        close = 50.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.001, len(dates))),
                "high": close * (1 + np.abs(rng.normal(0, 0.005, len(dates)))),
                "low": close * (1 - np.abs(rng.normal(0, 0.005, len(dates)))),
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, len(dates)),
                "ticker": ticker,
            },
            index=dates,
        )
        df.index.name = "date"
        panels.append(df)
    panel = pd.concat(panels)
    panel = drop_zero_volume(panel)
    feats = build_features(panel, feature_cfg, market=None)
    feats = multi_horizon_return_labels(feats, horizons=[1, 5, 20])
    feats = winsorize_returns(feats, return_col="log_return_1")

    feature_columns = [
        c
        for c in feats.columns
        if c
        not in {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ticker",
            "y_logret_h1",
            "y_logret_h5",
            "y_logret_h20",
        }
        and not c.startswith("log_return_")
    ]

    model = build_model(model_cfg, task="regression")
    result = train_walk_forward(
        feats,
        feature_columns=feature_columns,
        label_column="y_logret_h5",
        model=model,
        task="regression",
        splits_cfg=splits_cfg,
    )
    df_metrics = result.aggregate()
    print(df_metrics.to_string(index=False) if not df_metrics.empty else "(no folds)")
    return 0


def _cmd_train(_: argparse.Namespace) -> int:
    print(
        "The full training command is not wired yet. Use the demo subcommand "
        "or a notebook driver. Configure runs under configs/ and call "
        "stockml.training.train_walk_forward directly."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stockml")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version").set_defaults(func=_cmd_version)
    sub.add_parser("demo").set_defaults(func=_cmd_demo)
    sub.add_parser("train").set_defaults(func=_cmd_train)

    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
