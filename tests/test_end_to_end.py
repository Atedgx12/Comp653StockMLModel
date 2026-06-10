import pandas as pd

from stockml.features.pipeline import build_features
from stockml.labels.returns import multi_horizon_return_labels
from stockml.models import build_model
from stockml.training.trainer import train_walk_forward
from stockml.utils.io import project_root, read_yaml


def test_end_to_end_lightgbm_runs(synthetic_panel):
    cfg_root = project_root() / "configs"
    feature_cfg = read_yaml(cfg_root / "features" / "standard_technicals.yaml")
    splits_cfg = read_yaml(cfg_root / "splits" / "walk_forward.yaml")
    model_cfg = read_yaml(cfg_root / "model" / "lightgbm.yaml")
    splits_cfg["initial_train_years"] = 3
    splits_cfg["validation_years"] = 1
    splits_cfg["test_years"] = 1
    splits_cfg["step_years"] = 1
    model_cfg["params"]["num_iterations"] = 50
    model_cfg["params"]["early_stopping_rounds"] = 5

    feats = build_features(synthetic_panel, feature_cfg, market=None)
    feats = multi_horizon_return_labels(feats, horizons=[5])
    feats = feats.dropna(subset=["y_logret_h5"])

    drop = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ticker",
        "y_logret_h5",
    }
    feature_columns = [
        c for c in feats.columns if c not in drop and not c.startswith("log_return_")
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
    assert isinstance(df_metrics, pd.DataFrame)
    assert len(df_metrics) >= 1
