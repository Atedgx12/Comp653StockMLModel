# Architecture Notes

This document captures the design decisions in the codebase. It is meant for someone picking up the project and trying to understand why each module is the way it is.

## Layered structure

```
src/stockml/
  utils/         leaf level helpers, no project knowledge
  data/          ingestion and cleaning
  features/      indicator computation, regime features, pruning
  labels/        forward looking label generators
  splits/        walk forward folds, purge, embargo
  models/        learners with a uniform interface
  training/      metrics, walk forward driver
  evaluation/    backtest, transfer learning evaluation
  cli.py         end user entry points
```

Each layer depends only on the layers above it in the listing. Tests verify this contract by importing from each module in isolation.

## Why the model interface is small

`BaseModel` defines `fit`, `predict`, optional `predict_proba`, optional `feature_importances`, and `save`. Anything richer would lock the project into one ML framework. The current interface lets a LightGBM booster, a scikit-learn estimator, and a PyTorch module sit behind a single trainer.

## Why labels live in their own module

The task suite (multi horizon return regression, quantile, regime classification, sequence forecasting) is the part of the design most directly affected by the proposal feedback. Isolating label generation makes it easy to add a new task formulation without touching the feature pipeline or the trainer.

## Why the trainer takes a feature panel and a label column

The trainer is task agnostic. It receives a panel that already has features and labels and chooses metrics from the task name. New tasks plug in by writing a new label generator and adding a metrics dispatch case in `training.metrics`.

## Walk forward, purge, and embargo

Random shuffle splits leak information across time and inflate metrics. The trainer always uses walk forward folds with a configurable purge and embargo. The defaults in `configs/splits/walk_forward.yaml` follow the conservative end of the recommendations in *Advances in Financial Machine Learning*: twenty session purge, twenty session embargo, one year test windows.

## Per asset isolation in the feature pipeline

Every rolling window indicator must be computed per ticker so the rolling state does not leak across assets. The pipeline does this with a `groupby('ticker')` apply. The unit tests in `tests/test_technicals.py` include a no leakage check that recomputes an indicator on a truncated series and verifies the values agree on the overlapping timestamps.

## Optional torch dependency

PyTorch is heavy and slow to install. The project keeps it behind the `torch` extra so the core pipeline (data, features, labels, splits, linear and tree models, training) can be installed and exercised on a minimal environment. Sequence model factories raise an informative error when torch is missing.

## Configs

Configs are plain YAML files loaded with `stockml.utils.io.read_yaml`. The directory `configs/` is split into `data`, `features`, `labels`, `splits`, `model`, and `training` subtrees. Each subtree has a small set of named variants so the most common runs are a one line config swap inside a notebook or a script. Hydra-style composition is reserved for a later iteration; the current state keeps the config layer small and dependency free.

CI does not run the full training pipeline. It runs lint, type check (advisory), and unit tests, which is enough to catch most regressions without depending on data.
