"""
Build the DuckDB feature store for all horizons.
Run once; subsequent main.py runs use --use-store for instant data loading.

Usage:
    python build_store.py                              # all horizons
    python build_store.py --horizons 1 20 63 90 126   # specific horizons
    python build_store.py --start 2010-01-01           # from 2010
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ucn.data.ingestion import get_tickers, download_prices, download_volume, fetch_sentiment
from ucn.data.store import build_store

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    p = argparse.ArgumentParser(description="Build DuckDB feature store")
    p.add_argument("--horizons", nargs="+", type=int,
                   default=[1, 20, 63, 90, 126])
    p.add_argument("--start",    default="2010-01-01")
    p.add_argument("--db",       default=os.path.join(OUT_DIR, "features.duckdb"))
    args = p.parse_args()

    print(f"Building feature store: {args.db}")
    print(f"  Horizons : {args.horizons}")
    print(f"  Start    : {args.start}")

    tickers = get_tickers()
    close   = download_prices(tickers, start=args.start, cache_dir=OUT_DIR)
    sent_df = fetch_sentiment(tickers, close.index, cache_dir=OUT_DIR)
    vol_df  = download_volume(tickers, start=args.start, cache_dir=OUT_DIR)

    store = build_store(close, sent_df, vol_df,
                        horizons=args.horizons,
                        db_path=args.db)
    store.summary()
    store.close()
    print("\nDone. Use --use-store in future main.py runs for instant loading.")

if __name__ == "__main__":
    main()
