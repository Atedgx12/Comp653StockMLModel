from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stockml.data.preprocessing import drop_zero_volume, winsorize_returns
from stockml.features.pipeline import build_features
from stockml.labels.returns import multi_horizon_return_labels
from stockml.models import build_model
from stockml.training.trainer import train_walk_forward
from stockml.utils.io import project_root, read_yaml
from stockml.utils.logging import configure_logging
from stockml.utils.seeding import set_global_seed


def main() -> None:
    configure_logging("INFO")
    set_global_seed(42)

    cfg_root: Path = project_root() / "configs"
    feature_cfg = read_yaml(cfg_root / "features" / "standard_technicals.yaml")
    splits_cfg = read_yaml(cfg_root / "splits" / "walk_forward.yaml")
    model_cfg = read_yaml(cfg_root / "model" / "lightgbm.yaml")

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010-01-01", "2024-12-31")
    panels = []
    for ticker in ["AAA", "BBB", "CCC", "DDD"]:
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

    drop_cols = {
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
    feature_columns = [
        c for c in feats.columns if c not in drop_cols and not c.startswith("log_return_")
    ]

    model = build_model(model_cfg, task="regression")
    result = train_walk_forward(
        feats,
        feature_columns=feature_columns,
        label_column="y_logret_h5",
        model=model,
        task="regression",
        splits_cfg=splits_cfg,
        artifacts_dir="artifacts/lightgbm_demo",
    )
    print(result.aggregate().to_string(index=False) if result.fold_results else "(no folds)")


if __name__ == "__main__":
    main()
