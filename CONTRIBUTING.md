# Contributing

This is a single author course project, but the conventions below keep the codebase clean if a collaborator joins later.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

To enable PyTorch backed sequence models:

```powershell
pip install -e ".[torch]"
```

## Running checks before commit

```powershell
ruff check src tests
mypy src               # advisory; not blocking
pytest -q
```

## Data

Raw data is gitignored. See [data/README.md](data/README.md) for the directory layout and the staging instructions for each external dataset.

## Coding conventions

- Public functions have type annotations on every parameter and return value.
- Per asset rolling computations live behind `groupby('ticker')` so state never leaks across tickers.
- Forward looking targets always use `shift(-h)` and never `shift(h)`.
- Tests are unit level and use the synthetic panel fixture in `tests/conftest.py`. End to end tests use the same fixture so the suite stays under thirty seconds on commodity hardware.
