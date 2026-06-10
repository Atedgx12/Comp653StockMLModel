# data/

Raw and processed data is gitignored. This file documents the directory layout and how to populate each subfolder.

```
data/
  raw/         human staged downloads, never written by code
  interim/     intermediate parquet shards, written by ingestion
  processed/   final feature panels, written by feature pipeline
  external/    third party reference data, market index downloads
```

## Equities pretraining (Kaggle Huge Stock Market Dataset)

1. Download the dataset from <https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs>.
2. Unzip into `data/raw/Stocks/` so the CSV files are at `data/raw/Stocks/aapl.us.txt`, `data/raw/Stocks/msft.us.txt`, and so on.
3. Confirm a sample with:
   ```python
   from stockml.data.ingestion import KaggleStocksLoader
   loader = KaggleStocksLoader(raw_root="data/raw/Stocks")
   df = loader.load_ticker("AAPL")
   df.tail()
   ```

## Yahoo Finance (yfinance)

1. Network access is required at run time.
2. Run a fetch with:
   ```python
   from stockml.data.ingestion import YFinanceLoader
   loader = YFinanceLoader(start="2010-01-01", end="2024-12-31")
   df = loader.load_ticker("AAPL")
   df.tail()
   ```

## Cryptocurrency

For the transfer evaluation use the same `YFinanceLoader` with crypto tickers like `BTC-USD`, `ETH-USD`, `SOL-USD`. The loader handles the cryptocurrency session by aligning to the UTC midnight close that yfinance returns.

## Market index proxy

The regime feature group expects an index proxy. Pull the SPY series with the same yfinance loader and pass it as the `market` argument to `build_features`.

## Reproducibility

Every fetch is parameterized through the configs under `configs/data/`. Keep the requested date range stable across runs so results stay comparable.
