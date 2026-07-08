"""
Volatility term structure runner.

Trains a single VolTermStructureNet to predict forward realized volatility at
several horizons at once, with the curvature coupling that ties the horizons
together.  The features come from the existing pipeline.  The labels are the
forward realized volatility at each horizon, split at the cross sectional
median on every date so every name carries a label at every horizon.

Usage:
    set UCN_GPU=1& python term_structure_run.py --start 2010-01-01
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
from ucn.models.term_structure import VolTermStructureNet, DEFAULT_HORIZONS
from ucn.training.metrics import roc_auc

OUT_DIR = os.environ.get("UCN_OUT", ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--stride", type=int, default=9)
    p.add_argument("--ref-horizon", type=int, default=30,
                   help="Reference horizon used only to build the feature rows.")
    p.add_argument("--smooth-lambda", type=float, default=0.3,
                   help="Strength of the term structure curvature coupling.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--min-dollar-vol", type=float, default=50_000_000.0)
    p.add_argument("--cv-frac", type=float, default=0.80)
    return p.parse_args()


def build_multihorizon_labels(close, index, ticker_col, horizons):
    """Binary high/low volatility labels per horizon aligned to feature rows.

    For each ticker I compute forward realized volatility at every horizon,
    then rank it across the cross section on each date and split at the median.
    The result is a matrix with one column per horizon aligned row for row with
    the feature matrix.
    """
    idx_frames = []
    for ticker in close.columns:
        c = close[ticker].dropna()
        if len(c) < 300:
            continue
        r1 = np.log(c / c.shift(1))
        per_h = {}
        for h in horizons:
            per_h[h] = r1.rolling(h).std().shift(-h)
        df = pd.DataFrame(per_h)
        df["_ticker"] = ticker
        idx_frames.append(df)
    allf = pd.concat(idx_frames)
    # Rank each horizon cross sectionally per date, then median split.
    Y = {}
    for h in horizons:
        Y[h] = allf[h].groupby(allf.index).rank(pct=True)

    # Align to the requested feature rows by (date, ticker).
    key = pd.MultiIndex.from_arrays([index, ticker_col])
    out = np.full((len(index), len(horizons)), np.nan)
    for j, h in enumerate(horizons):
        s = Y[h].copy()
        s.index = pd.MultiIndex.from_arrays([allf.index, allf["_ticker"]])
        vals = s.reindex(key).values
        out[:, j] = (vals >= 0.5).astype(float)
        out[np.isnan(vals), j] = np.nan
    return out


def purged_split(dates, cv_frac, purge):
    unique_dates = np.sort(np.unique(dates))
    split_idx = int(cv_frac * len(unique_dates))
    split_dt = unique_dates[split_idx]
    purge_dt = unique_dates[max(0, split_idx - purge)]
    tr = dates < purge_dt
    te = dates >= split_dt
    return tr, te


def main():
    args = parse_args()
    print("=" * 65)
    print(" Volatility Term Structure — multi-horizon coupled model")
    print("=" * 65, flush=True)

    tickers = get_tickers()
    close   = download_prices(tickers, start=args.start, cache_dir=OUT_DIR)
    vol_df  = download_volume(tickers, cache_dir=OUT_DIR)
    sent_df = fetch_sentiment(tickers, close.index, cache_dir=OUT_DIR)

    close = filter_universe(close, vol_df, drop_delisted=True,
                            min_dollar_vol=args.min_dollar_vol)
    vol_df = vol_df.reindex(columns=close.columns)

    insample_end = str(pd.Timestamp(close.index[len(close.index)//2]).date())
    cluster_map = build_correlation_clusters(close, insample_end=insample_end)

    # Feature rows come from the standard pipeline at a reference horizon.
    X_df, _, feat_names, has_sent = make_features(
        close, sent_df, vol_df, horizon=args.ref_horizon, stride=args.stride,
        use_nomadic=True, use_hierarchy=True, sector_map=cluster_map,
        target="vol")
    dates = X_df.index.values
    ticker_col = X_df.pop("_ticker").values if "_ticker" in X_df.columns else None
    X = X_df.values.astype(np.float64)

    horizons = DEFAULT_HORIZONS
    print(f"\n[Labels] Building forward volatility labels at horizons "
          f"{horizons} ...", flush=True)
    Y = build_multihorizon_labels(close, dates, ticker_col, horizons)

    # Drop rows where any horizon label is missing (near the end of the sample).
    keep = ~np.isnan(Y).any(axis=1)
    X, Y, dates = X[keep], Y[keep], dates[keep]
    print(f"  Feature matrix: {X.shape}  labels: {Y.shape}", flush=True)

    purge = int(np.ceil(max(horizons) / max(args.stride, 1)))
    tr, te = purged_split(dates, args.cv_frac, purge)
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
    Xtr = (X[tr]-mu)/sd; Xte = (X[te]-mu)/sd
    Ytr, Yte = Y[tr], Y[te]
    print(f"  train={tr.sum():,}  test={te.sum():,}  purge={purge}", flush=True)

    print("\n[Train] Coupled term structure model ...", flush=True)
    net = VolTermStructureNet(horizons=horizons, hidden_sizes=(128, 64),
                              smooth_lambda=args.smooth_lambda,
                              epochs=args.epochs, patience=30, verbose=20)
    net.fit(Xtr, Ytr)
    P = net.predict_proba(Xte)

    print("\n=== Per-horizon test AUC ===")
    for j, h in enumerate(horizons):
        print(f"  {h:>4}d   AUC={roc_auc(Yte[:, j], P[:, j]):.4f}")
    curv = float(np.mean((P[:, 2:] - 2*P[:, 1:-1] + P[:, :-2])**2))
    print(f"\n  Term-structure curvature on test: {curv:.5f} "
          f"(lower = smoother coupled curve)")
    print(f"\nArtifacts dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
