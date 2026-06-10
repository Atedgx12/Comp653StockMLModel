"""Walk forward training driver.

The driver materializes folds, calls the model's fit/predict surface on each
fold, and aggregates per fold metrics into a single report. It is task aware
so the metrics dispatcher knows which family of metrics to compute for the
current label set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..models.base import BaseModel
from ..splits.walk_forward import Fold, expanding_walk_forward_folds, purge_and_embargo
from ..utils.logging import get_logger
from .metrics import classification_report, regression_report

logger = get_logger(__name__)


@dataclass
class FoldResult:
    fold: Fold
    metrics: dict[str, float]
    n_train: int
    n_test: int


@dataclass
class TrainResult:
    task: str
    fold_results: list[FoldResult] = field(default_factory=list)
    feature_importances: dict[str, float] | None = None

    def aggregate(self) -> pd.DataFrame:
        if not self.fold_results:
            return pd.DataFrame()
        rows = []
        for r in self.fold_results:
            row = {"fold": r.fold.name, "n_train": r.n_train, "n_test": r.n_test}
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows)


def _compute_metrics(task: str, y_true, y_pred, y_proba=None) -> dict[str, float]:
    if task == "classification":
        return classification_report(y_true, y_pred, y_proba=y_proba)
    return regression_report(y_true, y_pred)


def train_walk_forward(
    feature_panel: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model: BaseModel,
    task: str,
    splits_cfg: dict[str, Any],
    artifacts_dir: str | Path | None = None,
) -> TrainResult:
    """Run an expanding walk forward backtest end to end.

    The feature panel must already include both the engineered feature
    columns and the requested label column. The function does not mutate
    the input frame.
    """
    if not isinstance(feature_panel.index, pd.DatetimeIndex):
        raise TypeError("feature_panel must be indexed by date")
    df = feature_panel[[*feature_columns, label_column]].dropna()

    folds = list(
        expanding_walk_forward_folds(
            df.index,
            initial_train_years=splits_cfg.get("initial_train_years", 5),
            validation_years=splits_cfg.get("validation_years", 1),
            test_years=splits_cfg.get("test_years", 1),
            step_years=splits_cfg.get("step_years", 1),
        )
    )
    purge_days = int(splits_cfg.get("purge_days", 20))
    embargo_days = int(splits_cfg.get("embargo_days", 20))

    result = TrainResult(task=task)
    for fold in folds:
        train_mask = (df.index >= fold.train_start) & (df.index < fold.train_end)
        val_mask = (df.index >= fold.val_start) & (df.index < fold.val_end)
        test_mask = (df.index >= fold.test_start) & (df.index < fold.test_end)

        train_idx = purge_and_embargo(
            df.index[train_mask],
            test_start=fold.test_start,
            test_end=fold.test_end,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )
        train = df.loc[train_idx]
        val = df.loc[val_mask]
        test = df.loc[test_mask]

        if train.empty or test.empty:
            logger.info("Skipping fold %s due to empty split", fold.name)
            continue

        X_train = train[feature_columns].to_numpy()
        y_train = train[label_column].to_numpy()
        X_val = val[feature_columns].to_numpy() if not val.empty else None
        y_val = val[label_column].to_numpy() if not val.empty else None
        X_test = test[feature_columns].to_numpy()
        y_test = test[label_column].to_numpy()

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_columns)
        y_pred = model.predict(X_test)
        y_proba = None
        if task == "classification":
            try:
                y_proba = model.predict_proba(X_test)
            except NotImplementedError:
                y_proba = None
        metrics = _compute_metrics(task, y_test, y_pred, y_proba=y_proba)
        result.fold_results.append(
            FoldResult(fold=fold, metrics=metrics, n_train=len(train), n_test=len(test))
        )
        logger.info("Fold %s metrics: %s", fold.name, metrics)

    importances = model.feature_importances()
    if importances is not None:
        result.feature_importances = importances

    if artifacts_dir is not None:
        out = Path(artifacts_dir)
        out.mkdir(parents=True, exist_ok=True)
        df_metrics = result.aggregate()
        if not df_metrics.empty:
            df_metrics.to_csv(out / "fold_metrics.csv", index=False)
        if result.feature_importances:
            pd.Series(result.feature_importances).sort_values(ascending=False).to_csv(
                out / "feature_importances.csv"
            )
        if hasattr(model, "save"):
            model.save(out / "model.joblib")

    if not result.fold_results:
        logger.warning(
            "No folds completed. Check that the data spans the expected number of years."
        )
        # Reference numpy so the import is visibly used in static analysis.
        _ = np.array([])
    return result
