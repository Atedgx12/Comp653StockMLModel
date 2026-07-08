"""
Multi scale term structure runner.

Builds six day window sequences per stock date, one per horizon window, from
compact daily dynamics, then trains the MultiScaleTermStructureNet whose six
LSTM branches read those windows, fuse their cross scale drift, and predict the
forward volatility term structure.

Usage:
    set UCN_GPU=1& python multiscale_run.py --start 2010-01-01
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ucn.data.ingestion import (get_tickers, download_prices, download_volume,
                                 filter_universe)
from ucn.models.multiscale import MultiScaleTermStructureNet, DEFAULT_WINDOWS
from ucn.training.metrics import roc_auc

OUT_DIR = os.environ.get("UCN_OUT", ROOT)
T_MAX = 20   # cap on timesteps per branch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--stride", type=int, default=9)
    p.add_argument("--smooth-lambda", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--min-dollar-vol", type=float, default=50_000_000.0)
    p.add_argument("--cv-frac", type=float, default=0.80)
    return p.parse_args()


def daily_seq_features(close, vol):
    """Compact daily dynamics per ticker used inside the window sequences."""
    feats = {}
    for t in close.columns:
        c = close[t].dropna()
        if len(c) < 300:
            continue
        r1 = np.log(c / c.shift(1))
        df = pd.DataFrame({
            "r1":     r1,
            "absr":   r1.abs(),
            "rvol5":  r1.rolling(5).std(),
            "rvol10": r1.rolling(10).std(),
            "rvol20": r1.rolling(20).std(),
            "mom10":  (c / c.shift(10) - 1.0),
            "dvol":   (np.log(vol[t].reindex(c.index) + 1.0).diff()
                       if t in vol.columns else 0.0 * r1),
        }).fillna(0.0)
        feats[t] = df
    return feats


def sample_window(arr, w, t_max):
    """Take the last w rows of arr and sample to at most t_max evenly."""
    seg = arr[-w:]
    if len(seg) <= t_max:
        if len(seg) < t_max:
            pad = np.repeat(seg[:1], t_max - len(seg), axis=0)
            seg = np.concatenate([pad, seg], axis=0)
        return seg
    idx = np.linspace(0, len(seg) - 1, t_max).round().astype(int)
    return seg[idx]


def build_multiscale_sequences(close, vol, index, ticker_col, windows):
    """Return a list of B sequence tensors aligned to the feature rows."""
    daily = daily_seq_features(close, vol)
    d = next(iter(daily.values())).shape[1] if daily else 4
    steps = [min(w, T_MAX) for w in windows]
    B = len(windows)
    N = len(index)
    seqs = [np.zeros((N, steps[b], d), dtype=np.float32) for b in range(B)]
    valid = np.zeros(N, dtype=bool)

    dates = pd.to_datetime(index)
    by_ticker: dict = {}
    for i in range(N):
        by_ticker.setdefault(ticker_col[i], []).append(i)

    for tk, rows in by_ticker.items():
        if tk not in daily:
            continue
        df = daily[tk]
        arr = df.values.astype(np.float32)
        pos_of = {dt: p for p, dt in enumerate(df.index)}
        for i in rows:
            dt = dates[i]
            pos = pos_of.get(dt)
            if pos is None or pos < max(windows):
                continue
            hist = arr[:pos + 1]
            for b, w in enumerate(windows):
                seqs[b][i] = sample_window(hist, w, steps[b])
            valid[i] = True
    return seqs, valid


def build_labels(close, index, ticker_col, horizons):
    rows = []
    for ticker in close.columns:
        c = close[ticker].dropna()
        if len(c) < 300:
            continue
        r1 = np.log(c / c.shift(1))
        d = {"date": c.index, "ticker": ticker}
        for h in horizons:
            if h <= 1:
                d[f"h{h}"] = r1.shift(-1).abs().values
            else:
                d[f"h{h}"] = r1.rolling(h).std().shift(-h).values
        rows.append(pd.DataFrame(d))
    allf = pd.concat(rows, ignore_index=True)
    for h in horizons:
        rank = allf.groupby("date")[f"h{h}"].rank(pct=True)
        allf[f"y{h}"] = (rank >= 0.5).astype(float)
        allf.loc[allf[f"h{h}"].isna(), f"y{h}"] = np.nan
    lab = allf[["date", "ticker"] + [f"y{h}" for h in horizons]]
    key = pd.DataFrame({"date": pd.to_datetime(index), "ticker": ticker_col})
    key["_order"] = np.arange(len(key))
    merged = key.merge(lab, on=["date", "ticker"], how="left").sort_values("_order")
    return merged[[f"y{h}" for h in horizons]].values


def main():
    args = parse_args()
    print("=" * 65)
    print(" Multi-Scale Volatility Term Structure (window LSTM branches)")
    print("=" * 65, flush=True)

    windows = DEFAULT_WINDOWS
    tickers = get_tickers()
    close = download_prices(tickers, start=args.start, cache_dir=OUT_DIR)
    vol   = download_volume(tickers, cache_dir=OUT_DIR)
    close = filter_universe(close, vol, drop_delisted=True,
                            min_dollar_vol=args.min_dollar_vol)
    vol = vol.reindex(columns=close.columns)

    # Target rows: every stride-th trading day for every surviving ticker.
    print("[Rows] Building the stock-date grid ...", flush=True)
    all_dates = np.sort(close.index.unique())
    keep_dates = set(all_dates[::args.stride])
    rows_idx, rows_tk = [], []
    for t in close.columns:
        c = close[t].dropna()
        for dt in c.index:
            if dt in keep_dates:
                rows_idx.append(dt)
                rows_tk.append(t)
    index = np.array(rows_idx, dtype="datetime64[ns]")
    ticker_col = np.array(rows_tk, dtype=object)
    print(f"  {len(index):,} stock-date rows", flush=True)

    print("[Labels] Forward volatility at horizons "
          f"{windows} ...", flush=True)
    Y = build_labels(close, index, ticker_col, windows)

    print("[Sequences] Building six day-window branches ...", flush=True)
    seqs, valid = build_multiscale_sequences(close, vol, index, ticker_col, windows)

    keep = valid & ~np.isnan(Y).any(axis=1)
    seqs = [s[keep] for s in seqs]
    Y = Y[keep]
    index = index[keep]
    print(f"  valid rows: {keep.sum():,}   branch shapes: "
          f"{[s.shape for s in seqs]}", flush=True)

    # Purged temporal split.
    uniq = np.sort(np.unique(index))
    split_idx = int(args.cv_frac * len(uniq))
    split_dt = uniq[split_idx]
    purge = int(np.ceil(max(windows) / max(args.stride, 1)))
    purge_dt = uniq[max(0, split_idx - purge)]
    tr = index < purge_dt
    te = index >= split_dt
    print(f"  train={tr.sum():,}  test={te.sum():,}  purge={purge}", flush=True)

    # Standardize each branch by its training mean and std.
    seq_tr, seq_te = [], []
    for s in seqs:
        mu = s[tr].reshape(-1, s.shape[2]).mean(0)
        sd = s[tr].reshape(-1, s.shape[2]).std(0) + 1e-9
        seq_tr.append((s[tr] - mu) / sd)
        seq_te.append((s[te] - mu) / sd)
    Ytr, Yte = Y[tr], Y[te]

    print("\n[Train] Multi-scale coupled model ...", flush=True)
    net = MultiScaleTermStructureNet(windows=windows, hidden=24,
                                     trunk_sizes=(128, 64),
                                     smooth_lambda=args.smooth_lambda,
                                     epochs=args.epochs, patience=25, verbose=20)
    net.fit(seq_tr, Ytr)
    P = net.predict_proba(seq_te)

    print("\n=== Per-horizon test AUC (multi-scale) ===")
    for b, w in enumerate(windows):
        print(f"  {w:>4}d   AUC={roc_auc(Yte[:, b], P[:, b]):.4f}")
    curv = float(np.mean((P[:, 2:] - 2*P[:, 1:-1] + P[:, :-2])**2))
    print(f"\n  Term-structure curvature on test: {curv:.5f}")


if __name__ == "__main__":
    main()
