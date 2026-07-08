"""
Quantile price range runner: predict the price band, not the price.

Trains a QuantileTermStructureNet on the daily features to predict the 5, 25,
50, 75, and 95 percent quantiles of the forward log return at horizons of 1, 5,
10, 30, 90, and 180 days.  The quantile returns are turned into a price range by
multiplying today's price by their exponential, which gives an honest
prediction cone: the band the price should occupy at each future horizon.

Validation is by coverage.  A well calibrated 90 percent band should contain
about 90 percent of the realized returns, and the band should widen with
horizon.

Usage:
    set UCN_GPU=1& python quantile_run.py --start 2010-01-01 --epochs 400
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ucn.data.ingestion import (get_tickers, download_prices, download_volume,
                                 fetch_sentiment, filter_universe)
from ucn.data.features import make_features
from ucn.data.market_context import build_correlation_clusters
from ucn.models.quantile_net import QuantileTermStructureNet, DEFAULT_QUANTILES

HORIZONS = [1, 5, 10, 30, 90, 180]
OUT_DIR = os.environ.get("UCN_OUT", ROOT)


def ascii_bars(labels, values, title, width=46, fmt="{:.4f}"):
    print("\n" + title, flush=True)
    vmax = max(values) if max(values) > 0 else 1.0
    vmin = min(min(values), 0.0)
    span = (vmax - vmin) or 1.0
    for lab, v in zip(labels, values):
        n = int(round(width * (v - vmin) / span))
        print(f"  {str(lab):>6} | {'#' * n:<{width}} {fmt.format(v)}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--stride", type=int, default=9)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--min-dollar-vol", type=float, default=50_000_000.0)
    p.add_argument("--cv-frac", type=float, default=0.80)
    return p.parse_args()


def build_forward_returns(close, index, ticker_col, horizons):
    """Forward log returns per horizon aligned to the feature rows."""
    rows = []
    for ticker in close.columns:
        c = close[ticker].dropna()
        if len(c) < 300:
            continue
        d = {"date": c.index, "ticker": ticker}
        for h in horizons:
            d[f"r{h}"] = np.log(c.shift(-h) / c).values
        rows.append(pd.DataFrame(d))
    allf = pd.concat(rows, ignore_index=True)
    key = pd.DataFrame({"date": pd.to_datetime(index), "ticker": ticker_col})
    key["_order"] = np.arange(len(key))
    merged = key.merge(allf, on=["date", "ticker"], how="left").sort_values("_order")
    return merged[[f"r{h}" for h in horizons]].values


def main():
    args = parse_args()
    print("=" * 65)
    print(" Quantile Price Range Model (prediction cone)")
    print("=" * 65, flush=True)

    tickers = get_tickers()
    close = download_prices(tickers, start=args.start, cache_dir=OUT_DIR)
    vol   = download_volume(tickers, cache_dir=OUT_DIR)
    sent  = fetch_sentiment(tickers, close.index, cache_dir=OUT_DIR)
    close = filter_universe(close, vol, drop_delisted=True,
                            min_dollar_vol=args.min_dollar_vol)
    vol = vol.reindex(columns=close.columns)

    ie = str(pd.Timestamp(close.index[len(close.index)//2]).date())
    cmap = build_correlation_clusters(close, insample_end=ie)

    X_df, _, feat_names, has_sent = make_features(
        close, sent, vol, horizon=30, stride=args.stride, use_nomadic=True,
        use_hierarchy=True, sector_map=cmap, target="vol")
    dates = X_df.index.values
    ticker_col = X_df.pop("_ticker").values
    X = X_df.values.astype(np.float64)

    print(f"\n[Targets] Forward log returns at horizons {HORIZONS} ...",
          flush=True)
    Y = build_forward_returns(close, dates, ticker_col, HORIZONS)
    keep = ~np.isnan(Y).any(axis=1)
    X, Y, dates = X[keep], Y[keep], dates[keep]
    print(f"  rows: {X.shape[0]:,}", flush=True)

    uniq = np.sort(np.unique(dates))
    split = uniq[int(args.cv_frac * len(uniq))]
    purge = int(np.ceil(max(HORIZONS) / max(args.stride, 1)))
    purge_dt = uniq[max(0, int(args.cv_frac*len(uniq)) - purge)]
    tr = dates < purge_dt; te = dates >= split
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
    Xtr = (X[tr]-mu)/sd; Xte = (X[te]-mu)/sd
    Ytr, Yte = Y[tr], Y[te]
    print(f"  train={tr.sum():,}  test={te.sum():,}  purge={purge}", flush=True)

    print("\n[Train] Quantile term structure model ...", flush=True)
    net = QuantileTermStructureNet(horizons=HORIZONS, hidden_sizes=(128, 64),
                                   epochs=args.epochs, patience=60, verbose=20)
    net.fit(Xtr, Ytr)
    q = net.predict_quantiles(Xte)   # (N, H, Q)

    qs = DEFAULT_QUANTILES
    lo, hi = 0, len(qs) - 1
    print("\n=== Prediction cone: coverage and band width by horizon ===")
    print(f"{'Horizon':>8} {'90% cover':>10} {'50% cover':>10} "
          f"{'band width':>12} {'price band +/-':>16}")
    print("-" * 60)
    widths = []
    for j, h in enumerate(HORIZONS):
        cov90 = float(((Yte[:, j] >= q[:, j, lo]) &
                       (Yte[:, j] <= q[:, j, hi])).mean())
        cov50 = float(((Yte[:, j] >= q[:, j, 1]) &
                       (Yte[:, j] <= q[:, j, 3])).mean())
        width = float((q[:, j, hi] - q[:, j, lo]).mean())
        widths.append(width)
        # Convert the average half-band return into a price percentage.
        half = float(np.expm1((q[:, j, hi] - q[:, j, lo]).mean() / 2.0)) * 100
        print(f"{h:>7}d {cov90:>10.3f} {cov50:>10.3f} {width:>12.4f} "
              f"{half:>14.1f}%")

    ascii_bars([f"{h}d" for h in HORIZONS], widths,
               "[Cone] 5-95% return band width by horizon (widens with time):",
               fmt="{:.4f}")

    # Example price cone for one recent name.
    print("\n=== Example price cone (median stock, price = 100) ===")
    med = np.argsort(Yte[:, -1])[len(Yte)//2]
    print(f"{'Horizon':>8} {'5%':>8} {'25%':>8} {'50%':>8} {'75%':>8} {'95%':>8}")
    print("-" * 52)
    for j, h in enumerate(HORIZONS):
        band = 100.0 * np.exp(q[med, j, :])
        print(f"{h:>7}d " + " ".join(f"{b:>7.2f}" for b in band))


if __name__ == "__main__":
    main()
