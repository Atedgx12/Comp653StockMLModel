# Comp653 Stock ML Model

A generalized cross-asset prediction model for the COMP 653 Statistical Machine Learning course project. The system pretrains on a broad universe of US equities and fine tunes on narrower targets, including a cryptocurrency transfer evaluation. The codebase ships four learning task formulations and six model families so the cross asset generalization behavior can be compared across heterogeneous learners.

## Why this exists
The original course proposal framed the task as binary up or down classification of a single horizon. Course feedback noted that this formulation is too narrow to support a transfer learning narrative, so the project was reformulated. The current task suite is:

1. **Multi horizon return regression.** Predict signed log returns at horizons of 1, 5, and 20 sessions. The regression target carries strictly more information than a sign label and supports risk adjusted ranking downstream.
2. **Quantile regression.** Predict the 10th, 50th, and 90th percentiles of the future return distribution at the same horizons. This produces an explicit uncertainty estimate.
3. **Multi class regime classification.** Combine direction with realized volatility into a small set of regime labels. Richer than binary classification, easier to calibrate than full regression.
4. **Autoregressive sequence forecasting.** Treat the technical indicator window as a sequence and predict the next one to twenty session returns directly. This is the formulation that benefits most from transfer learning.

The binary up or down task is retained only as a sanity check baseline.

## Model families
| Family | Status | Role |
|---|---|---|
| Linear / logistic | Implemented | Interpretable baseline |
| LightGBM | Implemented | Tabular gradient boosting baseline |
| Online linear with regime features | Implemented | Direct response to nonstationarity |
| Temporal CNN | Stub | Local pattern extraction over rolling windows |
| Transformer encoder | Stub | Long range cross feature attention |
| LSTM | Stub | Classical recurrent baseline |

The neural network families are scaffolded behind the optional `torch` extra. Install with `pip install -e .[torch]` to enable them.

## Repository layout
```
configs/                  # YAML config tree for data, features, models, training
data/                     # Local cache for raw/interim/processed data (gitignored)
docs/                     # Proposal revisions, literature review, design notes
notebooks/                # EDA and result analysis notebooks
src/stockml/              # Python package
  data/                   # Ingestion, splitting, missing value handling
  features/               # Technical indicators and regime features
  labels/                 # Label generators for the four task formulations
  models/                 # Model implementations
  training/               # Training loop, metrics, walk forward CV
  evaluation/             # Backtest, calibration, transfer evaluation
  utils/                  # Logging, config, IO helpers
tests/                    # Unit tests for every module
```

## Quickstart
```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest -q
```

To enable neural models:
```powershell
pip install -e .[torch]
```

## Reproducing a run
The CLI ships a synthetic-data demo that exercises every layer end to end:
```powershell
stockml demo
```
For real runs, use the scripts under `scripts/` or write a notebook driver. Configs in `configs/` declare data sources, feature blocks, label generators, model hyperparameters, and walk forward split policy. The configs are loaded with `stockml.utils.io.read_yaml` so swapping a model or a label set is a one line edit.

## Course context
This repository is the implementation deliverable for the COMP 653 course project. The proposal revision that addresses the instructor's written feedback lives at [docs/proposal_revised.md](docs/proposal_revised.md). The supporting literature review lives at [docs/literature_review.md](docs/literature_review.md).
