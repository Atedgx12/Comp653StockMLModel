"""
Intraday volatility term structure runner.

Same idea as the daily term structure, but on the intraday time scale.  The
horizons are 1, 5, 15, 30, 60, and 240 minutes.  Free intraday data only goes
back about eleven days at one minute resolution, so this is a proof of concept
on a recent window and a reduced liquid universe rather than a long history
study.  Forward realized volatility is measured within each trading day so it
never crosses the overnight gap.

Usage:
    set UCN_GPU=1& python intraday_run.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ucn.data.ingestion import get_tickers
from ucn.models.multiscale import MultiScaleTermStructureNet
from ucn.training.metrics import roc_auc

OUT_DIR = os.environ.get("UCN_OUT", ROOT)
WINDOWS_MIN = [1, 5, 15, 30, 60, 240]   # horizons in minutes
T_MAX = 20


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
    p.add_argument("--n-tickers", type=int, default=60,
                   help="Number of liquid names to pull minute bars for.")
    p.add_argument("--stride-min", type=int, default=5,
                   help="Sample every this many minutes to reduce overlap.")
    p.add_argument("--smooth-lambda", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--cv-frac", type=float, default=0.80)
    return p.parse_args()


def download_minute(tickers, cache):
    if os.path.exists(cache):
        print("Loading cached minute bars ...", flush=True)
        return pd.read_parquet(cache)
    print(f"Downloading 1m bars for {len(tickers)} tickers ...", flush=True)
    raw = yf.download(tickers, period="7d", interval="1m",
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna(axis=1, how="all")
    close.to_parquet(cache)
    print(f"  {close.shape[1]} tickers x {close.shape[0]} minute bars", flush=True)
    return close


def within_day_fwd_vol(r1, day_id, h):
    """Forward realized vol over the next h minutes, within the same day."""
    fwd = r1.rolling(h).std().shift(-h)
    # Invalidate windows whose end crosses into the next trading day.
    end_day = pd.Series(day_id, index=r1.index).shift(-h)
    fwd[end_day.values != day_id] = np.nan
    return fwd


def main():
    args = parse_args()
    print("=" * 65)
    print(" Intraday Volatility Term Structure (1m to 240m horizons)")
    print("=" * 65, flush=True)

    tickers = get_tickers()[:args.n_tickers]
    close = download_minute(tickers, os.path.join(OUT_DIR, "minute_close.parquet"))
    tickers = close.columns.tolist()

    # Build per-ticker minute panels with day ids and seq features.
    day_seq = {}
    labels_rows = []
    for t in tickers:
        c = close[t].dropna()
        if len(c) < 400:
            continue
        r1 = np.log(c / c.shift(1)).fillna(0.0)
        day_id = c.index.normalize().view("int64")
        seq = pd.DataFrame({
            "r1": r1, "absr": r1.abs(),
            "rvol5": r1.rolling(5).std().fillna(0.0),
            "rvol15": r1.rolling(15).std().fillna(0.0),
        }).fillna(0.0)
        day_seq[t] = (seq, c.index, day_id)
        d = {"time": c.index, "ticker": t}
        for h in WINDOWS_MIN:
            d[f"h{h}"] = within_day_fwd_vol(r1, day_id, h).values
        labels_rows.append(pd.DataFrame(d))

    allf = pd.concat(labels_rows, ignore_index=True)
    for h in WINDOWS_MIN:
        rank = allf.groupby("time")[f"h{h}"].rank(pct=True)
        allf[f"y{h}"] = (rank >= 0.5).astype(float)
        allf.loc[allf[f"h{h}"].isna(), f"y{h}"] = np.nan
    mean_vol = [float(allf[f"h{h}"].mean()) for h in WINDOWS_MIN]
    ascii_bars([f"{w}m" for w in WINDOWS_MIN], mean_vol,
               "[Data] Mean intraday realized volatility by horizon:",
               fmt="{:.6f}")

    # Target grid: every stride-min minute per ticker.
    steps = [min(w, T_MAX) for w in WINDOWS_MIN]
    seqs = {b: [] for b in range(len(WINDOWS_MIN))}
    Yrows = []
    times = []
    print("[Sequences] Building six minute-window branches ...", flush=True)
    # Prebuild an O(1) label lookup keyed by (ticker, time).
    ycols = [f"y{h}" for h in WINDOWS_MIN]
    ymap = {}
    for row in allf[["ticker", "time"] + ycols].itertuples(index=False):
        ymap[(row[0], row[1])] = row[2:]
    for t in tickers:
        if t not in day_seq:
            continue
        seq, tindex, day_id = day_seq[t]
        arr = seq.values.astype(np.float32)
        for pos in range(max(WINDOWS_MIN), len(tindex), args.stride_min):
            tm = tindex[pos]
            yv = ymap.get((t, tm))
            if yv is None:
                continue
            yv = np.asarray(yv, dtype=float)
            if np.isnan(yv).any():
                continue
            # Windows must stay within the same trading day.
            if day_id[pos - max(WINDOWS_MIN)] != day_id[pos]:
                continue
            hist = arr[:pos + 1]
            for b, w in enumerate(WINDOWS_MIN):
                seg = hist[-w:]
                if len(seg) < steps[b]:
                    pad = np.repeat(seg[:1], steps[b]-len(seg), axis=0)
                    seg = np.concatenate([pad, seg], axis=0)
                elif len(seg) > steps[b]:
                    sidx = np.linspace(0, len(seg)-1, steps[b]).round().astype(int)
                    seg = seg[sidx]
                seqs[b].append(seg)
            Yrows.append(yv)
            times.append(tm)

    B = len(WINDOWS_MIN)
    seq_arr = [np.asarray(seqs[b], dtype=np.float32) for b in range(B)]
    Y = np.asarray(Yrows, dtype=np.float32)
    times = np.array(times)
    print(f"  samples: {len(Y):,}   branch shapes: {[s.shape for s in seq_arr]}",
          flush=True)

    # Purged split by time.
    uniq = np.sort(np.unique(times))
    split = uniq[int(args.cv_frac * len(uniq))]
    purge_dt = uniq[max(0, int(args.cv_frac*len(uniq)) - max(WINDOWS_MIN))]
    tr = times < purge_dt
    te = times >= split
    print(f"  train={tr.sum():,}  test={te.sum():,}", flush=True)

    seq_tr, seq_te = [], []
    for s in seq_arr:
        mu = s[tr].reshape(-1, s.shape[2]).mean(0)
        sd = s[tr].reshape(-1, s.shape[2]).std(0) + 1e-9
        seq_tr.append((s[tr]-mu)/sd); seq_te.append((s[te]-mu)/sd)
    Ytr, Yte = Y[tr], Y[te]

    print("\n[Train] Intraday multi-scale coupled model ...", flush=True)
    net = MultiScaleTermStructureNet(windows=WINDOWS_MIN, hidden=24,
                                     trunk_sizes=(128, 64),
                                     smooth_lambda=args.smooth_lambda,
                                     epochs=args.epochs, patience=150, verbose=20)
    net.fit(seq_tr, Ytr)
    P = net.predict_proba(seq_te)

    print("\n=== Per-horizon intraday test AUC ===")
    aucs = [roc_auc(Yte[:, b], P[:, b]) for b in range(B)]
    for w, a in zip(WINDOWS_MIN, aucs):
        print(f"  {w:>4}m   AUC={a:.4f}")
    ascii_bars([f"{w}m" for w in WINDOWS_MIN], aucs,
               "[Graph] Intraday per-horizon test AUC (1m to 240m):")


if __name__ == "__main__":
    main()
