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

from ucn.data.ingestion import get_tickers, download_prices, download_volume
from ucn.models.multiscale import MultiScaleTermStructureNet
from ucn.training.metrics import roc_auc

OUT_DIR = os.environ.get("UCN_OUT", ROOT)
WINDOWS_MIN = [5, 15, 30, 60, 120, 240]   # horizons in minutes
T_MAX = 20


def _interval_minutes(interval):
    """Minutes per bar for a yfinance interval string like '5m' or '1h'."""
    s = str(interval).strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    raise ValueError(f"unsupported intraday interval: {interval}")


def ascii_bars(labels, values, title, width=46, fmt="{:.4f}"):
    print("\n" + title, flush=True)
    vmax = max(values) if max(values) > 0 else 1.0
    vmin = min(min(values), 0.0)
    span = (vmax - vmin) or 1.0
    for lab, v in zip(labels, values):
        n = int(round(width * (v - vmin) / span))
        print(f"  {str(lab):>6} | {'#' * n:<{width}} {fmt.format(v)}", flush=True)


def _report_bands(labels, bands, Rte, unit, quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)):
    """Report calibrated quantile band coverage and an example price cone."""
    lo = bands[:, :, 0]; hi = bands[:, :, -1]
    cover = ((Rte >= lo) & (Rte <= hi)).mean(axis=0)
    ascii_bars([f"{w}{unit}" for w in labels], [float(c) for c in cover],
               "[Bands] Calibrated 90% band coverage by horizon (target 0.90):",
               fmt="{:.3f}")
    print("\n[Bands] Example price cone for a $100 stock (last horizon), "
          "average over test:")
    avg = bands.mean(axis=0)
    b = len(labels) - 1
    for tau, qq in zip(quantiles, avg[b]):
        print(f"    {int(tau*100):>3}th pct:  ${100.0 * float(np.exp(qq)):.2f}",
              flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-tickers", type=int, default=60,
                   help="Number of liquid names to pull intraday bars for.")
    p.add_argument("--interval", default="5m",
                   help="Bar interval, e.g. 1m, 5m, 15m, 30m, 60m, 1h.")
    p.add_argument("--period", default="60d",
                   help="History window yfinance should pull for the interval.")
    p.add_argument("--stride-min", type=int, default=5,
                   help="Sample every this many minutes to reduce overlap.")
    p.add_argument("--smooth-lambda", type=float, default=0.3)
    p.add_argument("--additivity-lambda", type=float, default=0.0,
                   help="Variance additivity coupling weight on the band term "
                        "structure. 0 disables it.")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--cv-frac", type=float, default=0.80)
    p.add_argument("--warm-restarts", action="store_true",
                   help="Use cosine warm restarts with Adam moment reset.")
    p.add_argument("--restart-period", type=int, default=120,
                   help="Epochs per cosine cycle when warm restarts are on.")
    p.add_argument("--no-context", action="store_true",
                   help="Disable the cross sectional context branch.")
    p.add_argument("--no-decision", action="store_true",
                   help="Disable the decision layer and per-ticker ledger.")
    p.add_argument("--context-top-k", type=int, default=25,
                   help="Keep the top-K context features by MI with volatility.")
    p.add_argument("--emit-json", default=None,
                   help="Write the per-horizon results to this JSON path.")
    return p.parse_args()


def download_bars(tickers, cache_prefix, interval="5m", period="60d"):
    """Fetch intraday close, volume, high and low panels for richer features."""
    fields = ["Close", "Volume", "High", "Low"]
    caches = {f: f"{cache_prefix}_{f.lower()}.parquet" for f in fields}
    if all(os.path.exists(p) for p in caches.values()):
        print("Loading cached intraday bars ...", flush=True)
        return {f: pd.read_parquet(p) for f, p in caches.items()}
    print(f"Downloading {interval} bars ({period}) for {len(tickers)} "
          "tickers ...", flush=True)
    raw = yf.download(tickers, period=period, interval=interval,
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for f in fields:
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[f]
        else:
            df = raw[[f]] if f in getattr(raw, "columns", []) else raw
        df = df.dropna(axis=1, how="all")
        df.to_parquet(caches[f])
        out[f] = df
    print(f"  {out['Close'].shape[1]} tickers x {out['Close'].shape[0]} bars",
          flush=True)
    return out


def within_day_fwd_vol(r1, day_id, h):
    """Forward realized vol over the next h minutes, within the same day."""
    if h <= 1:
        # One minute realized volatility is the absolute next minute return,
        # since the standard deviation of a single observation is undefined.
        fwd = r1.shift(-1).abs()
    else:
        fwd = r1.rolling(h).std().shift(-h)
    # Invalidate windows whose end crosses into the next trading day.
    end_day = pd.Series(day_id, index=r1.index).shift(-h)
    fwd[end_day.values != day_id] = np.nan
    return fwd


def within_day_fwd_return(close_t, day_id, h):
    """Forward log return over the next h minutes, within the same day."""
    fwd = np.log(close_t.shift(-h) / close_t)
    end_day = pd.Series(day_id, index=close_t.index).shift(-h)
    fwd[end_day.values != day_id] = np.nan
    return fwd


def run(args):
    """Run the intraday model and return per-horizon results."""
    bar_min = _interval_minutes(args.interval)
    def _bars(w):
        return max(1, int(round(w / bar_min)))
    H_BARS = [_bars(w) for w in WINDOWS_MIN]
    stride_bars = max(1, int(round(args.stride_min / bar_min)))
    print("=" * 65)
    print(f" Intraday Volatility Term Structure ({WINDOWS_MIN[0]}m to "
          f"{WINDOWS_MIN[-1]}m horizons, {args.interval} bars)")
    print("=" * 65, flush=True)
    tickers = get_tickers()[:args.n_tickers]
    cache_prefix = os.path.join(OUT_DIR, f"intraday_{args.interval}")
    fields = download_bars(tickers, cache_prefix, args.interval, args.period)
    close = fields["Close"]; volume = fields["Volume"]
    high = fields["High"]; low = fields["Low"]
    tickers = close.columns.tolist()

    # Build per-ticker minute panels with day ids and seq features.
    day_seq = {}
    labels_rows = []
    eps = 1e-9
    for t in tickers:
        c = close[t].dropna()
        if len(c) < 400:
            continue
        idx = c.index
        v = volume[t].reindex(idx) if t in volume.columns else pd.Series(0.0, index=idx)
        hi = high[t].reindex(idx) if t in high.columns else c
        lo = low[t].reindex(idx) if t in low.columns else c
        r1 = np.log(c / c.shift(1)).fillna(0.0)
        day_id = c.index.normalize().view("int64")
        day_series = pd.Series(day_id, index=idx)
        # Position of each bar within its trading day, normalized to [0, 1],
        # so the model can learn the U-shaped intraday volatility curve.
        pos_in_day = day_series.groupby(day_series).cumcount().values.astype(float)
        bars_per_day = float(np.median(day_series.value_counts().values)) or 1.0
        tod = pos_in_day / bars_per_day
        # The overnight gap only lives on the first bar of each day.
        gap = np.where(pos_in_day == 0, r1.values, 0.0)
        # Parkinson high-low range is a strong single-bar volatility proxy.
        rng = np.log((hi / lo).clip(lower=1.0 + eps)).fillna(0.0)
        # Relative volume against a trailing median, robust to level shifts.
        vmed = v.rolling(20, min_periods=5).median()
        vrel = np.log1p((v / (vmed + eps)).clip(lower=0.0)).fillna(0.0)
        signed_vol = (np.sign(r1) * vrel).fillna(0.0)
        seq = pd.DataFrame({
            "r1": r1, "absr": r1.abs(),
            "rvol5": r1.rolling(5).std().fillna(0.0),
            "rvol15": r1.rolling(15).std().fillna(0.0),
            "rvol30": r1.rolling(30).std().fillna(0.0),
            "rng": rng,
            "vrel": vrel,
            "signed_vol": signed_vol,
            "mom5": r1.rolling(5).sum().fillna(0.0),
            "accel": (r1 - r1.shift(1)).fillna(0.0),
            "tod": pd.Series(tod, index=idx),
            "gap": pd.Series(gap, index=idx),
        }).fillna(0.0)
        day_seq[t] = (seq, c.index, day_id)
        d = {"time": c.index, "ticker": t}
        for w, hb in zip(WINDOWS_MIN, H_BARS):
            d[f"h{w}"] = within_day_fwd_vol(r1, day_id, hb).values
            d[f"r{w}"] = within_day_fwd_return(c, day_id, hb).values
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
    steps = [min(hb, T_MAX) for hb in H_BARS]
    seqs = {b: [] for b in range(len(WINDOWS_MIN))}
    Yrows = []
    Rrows = []
    times = []
    tk_rows = []
    print("[Sequences] Building six bar-window branches ...", flush=True)
    # Prebuild an O(1) label lookup keyed by (ticker, time) holding both the
    # volatility labels and the forward returns for the quantile bands.
    ycols = [f"y{h}" for h in WINDOWS_MIN]
    rcols = [f"r{h}" for h in WINDOWS_MIN]
    nh = len(WINDOWS_MIN)
    ymap = {}
    for row in allf[["ticker", "time"] + ycols + rcols].itertuples(index=False):
        ymap[(row[0], row[1])] = row[2:]
    for t in tickers:
        if t not in day_seq:
            continue
        seq, tindex, day_id = day_seq[t]
        arr = seq.values.astype(np.float32)
        for pos in range(max(H_BARS), len(tindex), stride_bars):
            tm = tindex[pos]
            vals = ymap.get((t, tm))
            if vals is None:
                continue
            vals = np.asarray(vals, dtype=float)
            yv = vals[:nh]; rv = vals[nh:]
            if np.isnan(yv).any() or np.isnan(rv).any():
                continue
            # The forward label already stays within the trading day. The
            # lookback window may cross the overnight gap and is padded when
            # short, so no same-day restriction is needed on the history.
            hist = arr[:pos + 1]
            for b, hb in enumerate(H_BARS):
                seg = hist[-hb:]
                if len(seg) < steps[b]:
                    pad = np.repeat(seg[:1], steps[b]-len(seg), axis=0)
                    seg = np.concatenate([pad, seg], axis=0)
                elif len(seg) > steps[b]:
                    sidx = np.linspace(0, len(seg)-1, steps[b]).round().astype(int)
                    seg = seg[sidx]
                seqs[b].append(seg)
            Yrows.append(yv)
            Rrows.append(rv)
            times.append(tm)
            tk_rows.append(t)

    B = len(WINDOWS_MIN)
    seq_arr = [np.asarray(seqs[b], dtype=np.float32) for b in range(B)]
    Y = np.asarray(Yrows, dtype=np.float32)
    R = np.asarray(Rrows, dtype=np.float32)
    times = np.array(times)
    tk_arr = np.array(tk_rows, dtype=object)
    print(f"  samples: {len(Y):,}   branch shapes: {[s.shape for s in seq_arr]}",
          flush=True)

    # Purged split by time.
    uniq = np.sort(np.unique(times))
    split = uniq[int(args.cv_frac * len(uniq))]
    purge_dt = uniq[max(0, int(args.cv_frac*len(uniq)) - max(H_BARS))]
    tr = times < purge_dt
    te = times >= split
    print(f"  train={tr.sum():,}  test={te.sum():,}", flush=True)

    # The model owns per branch standardization now, so pass raw sequences.
    seq_tr = [s[tr] for s in seq_arr]
    seq_te = [s[te] for s in seq_arr]
    Ytr, Yte = Y[tr], Y[te]
    Rtr, Rte = R[tr], R[te]

    # Cross sectional context: each intraday row gets its day's daily context
    # vector (the hierarchy and regime signal), constant within the day.
    ctx_tr = ctx_te = None
    if not getattr(args, "no_context", False):
        try:
            from multiscale_run import build_context_features
            dclose = download_prices(tickers, start="2010-01-01", cache_dir=OUT_DIR)
            dvol = download_volume(tickers, cache_dir=OUT_DIR)
            # Align each intraday timestamp to its trading day. Intraday bars are
            # exchange local and time zone aware, so we drop the zone after
            # keeping local wall time and normalize to local midnight, which
            # matches the time zone naive daily dates the context is built on.
            _tidx = pd.DatetimeIndex(pd.to_datetime(times))
            if _tidx.tz is not None:
                _tidx = _tidx.tz_localize(None)
            sdates = _tidx.normalize().values
            ctx_full, ctx_names = build_context_features(
                dclose, dvol, sdates, tk_arr, ref_label=Y[:, -1],
                top_k=getattr(args, "context_top_k", 25))
            print(f"[Context] intraday context ({len(ctx_names)} features) "
                  f"aligned by ticker and day", flush=True)
            ctx_tr = np.nan_to_num(ctx_full[tr], nan=0.5).astype(np.float32)
            ctx_te = np.nan_to_num(ctx_full[te], nan=0.5).astype(np.float32)
        except Exception as e:
            print(f"[Context] intraday context skipped: {e}", flush=True)

    print("\n[Train] Intraday multi-scale coupled model (vol + quantile bands) ...",
          flush=True)
    net = MultiScaleTermStructureNet(windows=WINDOWS_MIN, hidden=24,
                                     trunk_sizes=(128, 64),
                                     smooth_lambda=args.smooth_lambda,
                                     additivity_lambda=getattr(args, "additivity_lambda", 0.0),
                                     epochs=args.epochs, patience=150, verbose=20,
                                     warm_restarts=getattr(args, "warm_restarts", False),
                                     restart_period=getattr(args, "restart_period", 120))
    net.fit(seq_tr, Ytr, Rtr, ctx=ctx_tr)
    P = net.predict_proba(seq_te, ctx=ctx_te)
    bands = net.predict_bands(seq_te, ctx=ctx_te)

    print("\n=== Per-horizon intraday test AUC ===")
    aucs = [roc_auc(Yte[:, b], P[:, b]) for b in range(B)]
    for w, a in zip(WINDOWS_MIN, aucs):
        print(f"  {w:>4}m   AUC={a:.4f}")
    ascii_bars([f"{w}m" for w in WINDOWS_MIN], aucs,
               "[Graph] Intraday per-horizon test AUC (1m to 240m):")
    from multiscale_run import selective_accuracy
    selective_accuracy(P, Yte, WINDOWS_MIN, "m")
    _report_bands(WINDOWS_MIN, bands, Rte, "m")

    # Decision layer: score the intraday test bands into the same per-ticker
    # ledger, so both scales feed one honing loop.
    if not getattr(args, "no_decision", False):
        try:
            from ucn.models.decision import (choose_batch, score_batch,
                                              TickerLedger)
            tk_te = tk_arr[te]
            time_te = pd.to_datetime(times[te])
            p0 = np.array([close[tk].get(pd.Timestamp(tm), np.nan)
                           for tk, tm in zip(tk_te, time_te)], dtype=float)
            rows = []
            for b, w in enumerate(WINDOWS_MIN):
                ch = choose_batch(bands[:, b, :], p0)
                ch = score_batch(ch, p0 * np.exp(Rte[:, b]))
                ch["ticker"] = tk_te; ch["date"] = time_te
                ch["horizon"] = f"{w}m"; ch["source"] = "auto"
                rows.append(ch)
            batch = pd.concat(rows, ignore_index=True).dropna(
                subset=["p0", "actual"])
            ledger_path = os.path.join(OUT_DIR, "ticker_ledger.parquet")
            ledger = TickerLedger(ledger_path)
            ledger.append(batch)
            ledger.df = ledger.df.drop_duplicates(
                subset=["ticker", "date", "horizon", "source"], keep="last")
            ledger.save()
            cov = float(pd.to_numeric(batch["in_band"], errors="coerce").mean())
            mpe = float(batch["pct_error"].abs().mean())
            print(f"\n[Decision] scored {len(batch):,} intraday choices  "
                  f"in-band={cov:.3f}  mean abs pct error={mpe:.4f}  "
                  f"ledger now {len(ledger.df):,} rows", flush=True)
        except Exception as e:
            print(f"[Decision] intraday skipped: {e}", flush=True)

    save_path = os.path.join(OUT_DIR, f"multiscale_intraday_{args.interval}.npz")
    net.save(save_path)
    print(f"\n[Save] Intraday multi-scale model written to {save_path}", flush=True)

    # Horizon in trading days (390 minutes per trading day) for a common axis.
    return [{"label": f"{w}m", "days": w / 390.0, "mean_vol": mv, "auc": a}
            for w, mv, a in zip(WINDOWS_MIN, mean_vol, aucs)]


def main():
    args = parse_args()
    res = run(args)
    ej = getattr(args, "emit_json", None)
    if ej:
        import json
        with open(ej, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
