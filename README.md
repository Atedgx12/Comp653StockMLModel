# COMP 653 Multi-Scale Volatility Term Structure Model

[![CI](https://github.com/Atedgx12/Comp653StockMLModel/actions/workflows/ci.yml/badge.svg)](https://github.com/Atedgx12/Comp653StockMLModel/actions/workflows/ci.yml)

**Course:** COMP 653 Statistical Machine Learning, Summer 2026, Rice University

**Authors:** Zachary Powell (`zp21@rice.edu`) and Josh Levy (`jl500@rice.edu`)

This repository predicts cross-sectional realized-volatility rank and return
quantile bands across six intraday and six daily horizons. The final
`MultiScaleTermStructureNet` uses one NumPy/CuPy LSTM branch per horizon,
temporal and cross-branch attention, a shared trunk, and conformal quantile
calibration. The earlier `stockml` baselines and unified course network remain
available for comparison.

## Reported final-run results

These AUC values are the held-out results reported in the final course report.
CI verifies packaging and model behavior on synthetic data; it does not
download the historical universe or reproduce a full training run.

| Scale | Horizon | Multi-Scale + Attention AUC |
|---|---:|---:|
| Intraday | 5m | 0.739 |
| Intraday | 15m | 0.829 |
| Intraday | 30m | 0.915 |
| Intraday | 60m | 0.960 |
| Intraday | 120m | 0.981 |
| Intraday | 240m | 0.989 |
| Daily | 1d | 0.702 |
| Daily | 5d | 0.856 |
| Daily | 10d | 0.909 |
| Daily | 30d | 0.957 |
| Daily | 90d | 0.979 |
| Daily | 180d | 0.983 |

The report also gives a mean LightGBM AUC of 0.515 on the daily
cross-sectional baseline. See [the architecture reference](docs/architecture.md)
for the model and loss definitions, and [the evaluation notes](docs/evaluation_notes.md)
for the steps needed before treating the reported values as reproduced
estimates.

## Setup

Python 3.11 and 3.12 are supported.

```bash
git clone https://github.com/Atedgx12/Comp653StockMLModel.git
cd Comp653StockMLModel
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

LightGBM requires an OpenMP runtime. On macOS, install it with
`brew install libomp` before running the LightGBM baseline. The final
multi-scale model itself uses NumPy on CPU and does not require LightGBM.

For an NVIDIA CUDA 12 GPU, install the optional CuPy backend and request it at
runtime:

```bash
python -m pip install -e ".[gpu]"
UCN_GPU=1 python scripts/multiscale_run.py --help
```

If the CUDA runtime cannot be initialized, the backend reports the problem and
falls back to NumPy.

## Run the final models

Start by checking the available options; these commands do not download data:

```bash
python scripts/multiscale_run.py --help
python scripts/intraday_run.py --help
```

Example CPU runs using the final-report training settings are:

```bash
mkdir -p artifacts

UCN_OUT=artifacts python scripts/multiscale_run.py \
  --start 2010-01-01 \
  --epochs 3000 \
  --label-pct 0.3 \
  --additivity-lambda 0.5

UCN_OUT=artifacts python scripts/intraday_run.py \
  --interval 5m \
  --period 60d \
  --epochs 3000 \
  --label-pct 0.3 \
  --additivity-lambda 0.5
```

The runners fetch market data with `yfinance`, cache tabular data under the
selected output directory, and save `.npz` checkpoints there. Results can vary
as the live data source, available constituents, and provider retention windows
change. Set `UCN_GPU=1` in front of either command to use the installed CuPy
backend.

The retained baseline pipeline can be run with:

```bash
python scripts/pipeline_course.py
```

## Repository structure

```text
src/ucn/                    final from-scratch course models
  backend.py                NumPy/CuPy device selection
  models/multiscale.py      dual-attention term-structure network
  data/                     ingestion, features, context, and data store
src/stockml/                reusable baselines and training utilities
scripts/
  multiscale_run.py         daily 1d-to-180d final runner
  intraday_run.py           intraday 5m-to-240m final runner
configs/                    baseline data, feature, model, and split configs
models/                     retained trained checkpoints
docs/                       architecture, derivations, and evaluation notes
tests/                      unit, runner-import, and model checkpoint tests
```

## Development checks

```bash
ruff check src tests
mypy src/stockml
pytest --cov=stockml --cov=ucn --cov-report=term-missing
```

CI runs lint and type checks once and executes the tests on Python 3.11 and
3.12. Contributions should keep generated datasets and large local artifacts
out of Git; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Course alignment

| Component | Course topic |
|---|---|
| Entropy and mutual-information feature selection | Module 2 |
| Temporal splitting and cross-sectional evaluation | Module 3 |
| Logistic regression and Naive Bayes branches | Module 5 |
| MLP, LSTM, attention, and backpropagation | Module 5 |
| Adam optimization and regularization | Module 6 |
| Quantile loss and conformal calibration | Final extension |
