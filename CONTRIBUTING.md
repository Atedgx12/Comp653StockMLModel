# Contributing

These conventions keep changes reviewable and the two model packages usable
across supported environments.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

To enable PyTorch backed sequence models:

```bash
python -m pip install -e ".[torch]"
```

## Running checks before commit

```bash
ruff check src tests
mypy src/stockml
pytest --cov=stockml --cov=ucn --cov-report=term-missing
```

## Data

Raw data is gitignored. See [data/README.md](data/README.md) for the directory layout and the staging instructions for each external dataset.

## Coding conventions

- Public functions have type annotations on every parameter and return value.
- Per asset rolling computations live behind `groupby('ticker')` so state never leaks across tickers.
- Forward looking targets always use `shift(-h)` and never `shift(h)`.
- Tests use deterministic synthetic data and must not make live network calls.
- Scientific result changes include the exact command, data manifest, split
  dates, seed, and machine-readable per-horizon metrics.
