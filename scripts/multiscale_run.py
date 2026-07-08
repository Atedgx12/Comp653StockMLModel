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


def ascii_bars(labels, values, title, width=46, fmt="{:.4f}"):
    """Print a horizontal bar chart to the terminal."""
    print("\n" + title, flush=True)
    vmax = max(values) if max(values) > 0 else 1.0
    vmin = min(min(values), 0.0)
    span = (vmax - vmin) or 1.0
    for lab, v in zip(labels, values):
        n = int(round(width * (v - vmin) / span))
        bar = "#" * n
        print(f"  {str(lab):>6} | {bar:<{width}} {fmt.format(v)}", flush=True)


def _report_bands(labels, bands, Rte, unit, quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)):
    """Report calibrated quantile band coverage and an example price cone."""
    import numpy as _np
    lo = bands[:, :, 0]; hi = bands[:, :, -1]
    cover = ((Rte >= lo) & (Rte <= hi)).mean(axis=0)
    ascii_bars([f"{w}{unit}" for w in labels], [float(c) for c in cover],
               "[Bands] Calibrated 90% band coverage by horizon (target 0.90):",
               fmt="{:.3f}")
    print("\n[Bands] Example price cone for a $100 stock (last horizon), "
          "average over test:")
    avg = bands.mean(axis=0)   # (B, Q)
    b = len(labels) - 1
    for tau, qq in zip(quantiles, avg[b]):
        print(f"    {int(tau*100):>3}th pct:  ${100.0 * float(_np.exp(qq)):.2f}",
              flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--stride", type=int, default=9)
    p.add_argument("--smooth-lambda", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--min-dollar-vol", type=float, default=50_000_000.0)
    p.add_argument("--cv-frac", type=float, default=0.80)
    p.add_argument("--warm-start", default=None,
                   help="Intraday checkpoint to warm start the shared trunk from.")
    p.add_argument("--warm-start-window-days", type=int, default=60,
                   help="Length in days of the intraday window, used by the "
                        "leakage guard.")
    p.add_argument("--warm-start-force", action="store_true",
                   help="Override the leakage guard and warm start anyway.")
    p.add_argument("--stack-intraday", action="store_true",
                   help="Append intraday realized volatility as a daily feature.")
    p.add_argument("--stack-interval", default="5m")
    p.add_argument("--stack-period", default="60d")
    p.add_argument("--stack-min-train-cov", type=float, default=0.02,
                   help="Skip stacking if train coverage is below this fraction.")
    p.add_argument("--no-context", action="store_true",
                   help="Disable the static cross sectional context branch.")
    p.add_argument("--warm-restarts", action="store_true",
                   help="Use cosine warm restarts with Adam moment reset.")
    p.add_argument("--restart-period", type=int, default=120,
                   help="Epochs per cosine cycle when warm restarts are on.")
    p.add_argument("--no-decision", action="store_true",
                   help="Disable the decision layer and per-ticker ledger.")
    p.add_argument("--emit-json", default=None,
                   help="Write the per-horizon results to this JSON path.")
    return p.parse_args()


def build_context_features(close, vol, index, ticker_col):
    """Full cross sectional context vector fused into the trunk.

    This calls the real make_features to build the same rich signal the strong
    cross sectional model used: the multi lag returns, multi window volatility
    and momentum, the nomadic indicators, and the hierarchy of stock versus
    sector versus market, each ranked per date so the context is market neutral.
    The market and sector relative columns carry the hierarchy signal, and the
    model gates them. The features are aligned to the multi scale grid.
    """
    from ucn.data.features import make_features
    try:
        from ucn.data.market_context import load_or_build_sector_map
        sector_map = load_or_build_sector_map(close.columns.tolist())
    except Exception:
        sector_map = None
    X_df, _y, feat_names, _has = make_features(
        close, sent_df=None, vol_df=vol,
        min_history=300, horizon=1, stride=1,
        use_nomadic=True, use_hierarchy=True, sector_map=sector_map,
        target="quantile")
    feat_cols = [c for c in feat_names if c in X_df.columns]
    df = pd.DataFrame({"date": pd.to_datetime(X_df.index),
                       "ticker": X_df["_ticker"].values})
    for col in feat_cols:
        df[col] = X_df[col].values
    key = pd.DataFrame({"date": pd.to_datetime(index), "ticker": ticker_col})
    key["_o"] = np.arange(len(key))
    merged = key.merge(df, on=["date", "ticker"], how="left").sort_values("_o")
    ctx = merged[feat_cols].values
    cov = float(np.mean(~np.isnan(ctx).any(axis=1)))
    print(f"  context grid coverage: {cov:.3f}", flush=True)
    # Ranked and macro features both live in a zero to one band, so a missing
    # row is filled with the neutral centre rather than zero.
    ctx = np.nan_to_num(ctx, nan=0.5).astype(np.float32)
    return ctx, feat_cols


def build_intraday_stack(tickers, index, ticker_col, interval, period):
    """Per (ticker, date) realized intraday volatility aligned to the daily grid.
    This feeds the finer scale forward as one daily feature. Returns the aligned
    values with missing entries as NaN, plus the coverage fraction over the grid.
    """
    from intraday_run import download_bars
    cache_prefix = os.path.join(OUT_DIR, f"intraday_{interval}")
    fields = download_bars(list(tickers), cache_prefix, interval, period)
    close_i = fields["Close"]
    rows = []
    for t in close_i.columns:
        c = close_i[t].dropna()
        if len(c) < 50:
            continue
        r = np.log(c / c.shift(1))
        rv = r.groupby(c.index.normalize()).std()
        rows.append(pd.DataFrame({"date": rv.index, "ticker": t,
                                  "iv": rv.values}))
    if not rows:
        return None, 0.0
    long = pd.concat(rows, ignore_index=True)
    key = pd.DataFrame({"date": pd.to_datetime(index), "ticker": ticker_col})
    key["_o"] = np.arange(len(key))
    merged = key.merge(long, on=["date", "ticker"], how="left").sort_values("_o")
    vals = merged["iv"].values.astype(np.float32)
    cover = float(np.mean(~np.isnan(vals)))
    return vals, cover



def daily_seq_features(close, vol):
    """Compact daily dynamics per ticker used inside the window sequences.

    These features are shared by all six window branches, so enriching them
    enriches every horizon at once.
    """
    eps = 1e-9
    feats = {}
    for t in close.columns:
        c = close[t].dropna()
        if len(c) < 300:
            continue
        r1 = np.log(c / c.shift(1))
        rvol5 = r1.rolling(5).std()
        rvol20 = r1.rolling(20).std()
        if t in vol.columns:
            v = vol[t].reindex(c.index)
            vmed = v.rolling(20, min_periods=5).median()
            rel_vol = np.log1p((v / (vmed + eps)).clip(lower=0.0))
            dvol = np.log(v + 1.0).diff()
        else:
            rel_vol = 0.0 * r1
            dvol = 0.0 * r1
        df = pd.DataFrame({
            "r1":       r1,
            "absr":     r1.abs(),
            "rvol5":    rvol5,
            "rvol10":   r1.rolling(10).std(),
            "rvol20":   rvol20,
            "rvol60":   r1.rolling(60).std(),
            "vol_ratio": rvol5 / (rvol20 + eps),
            "mom10":    (c / c.shift(10) - 1.0),
            "mom20":    (c / c.shift(20) - 1.0),
            "accel":    (r1 - r1.shift(1)),
            "dvol":     dvol,
            "rel_vol":  rel_vol,
            "signed_vol": np.sign(r1) * rel_vol,
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
            # Forward log return over the horizon, the quantile band target.
            d[f"r{h}"] = np.log(c.shift(-h) / c).values
        rows.append(pd.DataFrame(d))
    allf = pd.concat(rows, ignore_index=True)
    for h in horizons:
        rank = allf.groupby("date")[f"h{h}"].rank(pct=True)
        allf[f"y{h}"] = (rank >= 0.5).astype(float)
        allf.loc[allf[f"h{h}"].isna(), f"y{h}"] = np.nan
    ycols = [f"y{h}" for h in horizons]
    rcols = [f"r{h}" for h in horizons]
    lab = allf[["date", "ticker"] + ycols + rcols]
    key = pd.DataFrame({"date": pd.to_datetime(index), "ticker": ticker_col})
    key["_order"] = np.arange(len(key))
    merged = key.merge(lab, on=["date", "ticker"], how="left").sort_values("_order")
    mean_vol = [float(allf[f"h{h}"].mean()) for h in horizons]
    return (merged[ycols].values, mean_vol, merged[rcols].values)



def run(args):
    """Run the daily multi-scale model and return per-horizon results."""
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
    Y, mean_vol, Yret = build_labels(close, index, ticker_col, windows)

    # Show the actual volatility term structure shape of the data.
    ascii_bars([f"{w}d" for w in windows], mean_vol,
               "[Data] Mean daily realized volatility by horizon (term structure shape):",
               fmt="{:.5f}")

    print("[Sequences] Building six day-window branches ...", flush=True)
    seqs, valid = build_multiscale_sequences(close, vol, index, ticker_col, windows)

    keep = valid & ~np.isnan(Y).any(axis=1) & ~np.isnan(Yret).any(axis=1)

    # Static cross sectional context (the rich hierarchy signal) fused into the
    # trunk, built for every grid row and gated inside the model.
    ctx_kept = None
    if not getattr(args, "no_context", False):
        print("[Context] Building cross sectional hierarchy features ...",
              flush=True)
        ctx_full, ctx_names = build_context_features(close, vol, index, ticker_col)
        ctx_kept = ctx_full[keep]
        print(f"  context features ({len(ctx_names)}): {ctx_names[:12]} ...",
              flush=True)

    # Optional stacking: feed the intraday scale forward as one daily feature.
    stack_kept = None
    if getattr(args, "stack_intraday", False):
        print("[Stack] Building intraday volatility feature ...", flush=True)
        stack_full, cover = build_intraday_stack(
            close.columns, index, ticker_col,
            getattr(args, "stack_interval", "5m"),
            getattr(args, "stack_period", "60d"))
        if stack_full is not None:
            print(f"  intraday feature coverage over the daily grid: {cover:.4f}",
                  flush=True)
            stack_kept = stack_full[keep]

    seqs = [s[keep] for s in seqs]
    Y = Y[keep]; Yret = Yret[keep]
    tk_all = np.asarray(ticker_col)[keep]
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

    # The model owns per branch standardization now, so pass raw sequences.
    seq_tr = [s[tr] for s in seqs]
    seq_te = [s[te] for s in seqs]
    Ytr, Yte = Y[tr], Y[te]
    Rtr, Rte = Yret[tr], Yret[te]
    ctx_tr = ctx_te = None
    if ctx_kept is not None:
        ctx_tr = np.nan_to_num(ctx_kept[tr], nan=0.5).astype(np.float32)
        ctx_te = np.nan_to_num(ctx_kept[te], nan=0.5).astype(np.float32)

    # Append the stacked intraday feature, guarded on train coverage so a
    # feature that is absent during training cannot shift the test distribution.
    if stack_kept is not None:
        s0 = np.nan_to_num(stack_kept, nan=0.0)
        tr_cov = float(np.mean(s0[tr] != 0.0))
        te_cov = float(np.mean(s0[te] != 0.0))
        print(f"[Stack] nonzero coverage  train={tr_cov:.4f}  test={te_cov:.4f}",
              flush=True)
        thr = getattr(args, "stack_min_train_cov", 0.02)
        if tr_cov < thr:
            print(f"[Stack] SKIPPED: train coverage {tr_cov:.4f} below "
                  f"{thr:.4f}. The intraday history is too short to train on, "
                  "so the feature is dropped to avoid a train and test mismatch.",
                  flush=True)
        else:
            def _append(seq_list, mask):
                sc = s0[mask]
                out = []
                for s in seq_list:
                    col = np.broadcast_to(sc[:, None, None],
                                          (s.shape[0], s.shape[1], 1)).astype(s.dtype)
                    out.append(np.concatenate([s, col], axis=2))
                return out
            seq_tr = _append(seq_tr, tr)
            seq_te = _append(seq_te, te)
            print(f"  appended intraday feature  new branch dim="
                  f"{seq_tr[0].shape[2]}", flush=True)

    print("\n[Train] Multi-scale coupled model (vol + quantile bands) ...",
          flush=True)
    net = MultiScaleTermStructureNet(windows=windows, hidden=24,
                                     trunk_sizes=(128, 64),
                                     smooth_lambda=args.smooth_lambda,
                                     epochs=args.epochs, patience=150, verbose=20,
                                     warm_restarts=getattr(args, "warm_restarts", False),
                                     restart_period=getattr(args, "restart_period", 120))

    # Optional guarded warm start from the intraday model. The intraday window
    # sits inside the daily test period, so a naive warm start would leak the
    # test period into training. The guard declines unless it is forced.
    ws = getattr(args, "warm_start", None)
    if ws:
        max_dt = index.max()
        window = int(getattr(args, "warm_start_window_days", 60))
        intraday_start = max_dt - np.timedelta64(window, "D")
        overlaps = (split_dt <= max_dt) and (max_dt >= intraday_start)
        force = getattr(args, "warm_start_force", False)
        print(f"[Warm start] intraday window >= {intraday_start}  "
              f"daily test starts {split_dt}", flush=True)
        if overlaps and not force:
            print("[Warm start] SKIPPED: intraday window overlaps the daily "
                  "test period, warm starting would leak. Use --warm-start-force "
                  "to override.", flush=True)
        else:
            net.warm_start_from(ws, seq_tr[0].shape[2])

    net.fit(seq_tr, Ytr, Rtr, ctx=ctx_tr)
    P = net.predict_proba(seq_te, ctx=ctx_te)
    bands = net.predict_bands(seq_te, ctx=ctx_te)

    print("\n=== Per-horizon test AUC (multi-scale) ===")
    aucs = [roc_auc(Yte[:, b], P[:, b]) for b in range(len(windows))]
    for w, a in zip(windows, aucs):
        print(f"  {w:>4}d   AUC={a:.4f}")
    ascii_bars([f"{w}d" for w in windows], aucs,
               "[Graph] Per-horizon test AUC across 1/5/10/30/90/180 days:")
    curv = float(np.mean((P[:, 2:] - 2*P[:, 1:-1] + P[:, :-2])**2))
    print(f"\n  Term-structure curvature on test: {curv:.5f}")
    _report_bands(windows, bands, Rte, "d")

    # Decision layer: turn the calibrated test bands into choices, score them
    # against the realized prices, and accumulate a combined per-ticker ledger.
    if not getattr(args, "no_decision", False):
        try:
            from ucn.models.decision import (choose_batch, score_batch,
                                              TickerLedger)
            tk_te = tk_all[te]
            date_te = pd.to_datetime(index[te])
            p0 = np.array([close[tk].get(dt, np.nan)
                           for tk, dt in zip(tk_te, date_te)], dtype=float)
            rows = []
            for b, w in enumerate(windows):
                ch = choose_batch(bands[:, b, :], p0)
                ch = score_batch(ch, p0 * np.exp(Rte[:, b]))
                ch["ticker"] = tk_te; ch["date"] = date_te
                ch["horizon"] = w; ch["source"] = "auto"
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
            mae = float(batch["abs_error"].mean())
            mpe = float(batch["pct_error"].abs().mean())
            print(f"\n[Decision] scored {len(batch):,} choices on the test set  "
                  f"in-band={cov:.3f}  mean abs error=${mae:.2f}  "
                  f"mean abs pct error={mpe:.4f}", flush=True)
            deltas = ledger.per_ticker_delta()
            nz = sum(1 for v in deltas.values() if v > 0)
            print(f"[Decision] ledger holds {len(ledger.df):,} rows across "
                  f"{ledger.df['ticker'].nunique()} tickers; per-ticker "
                  f"conformal deltas computed ({nz} nonzero) to hone future "
                  f"band widths. Saved to {ledger_path}", flush=True)
        except Exception as e:
            print(f"[Decision] skipped: {e}", flush=True)

    save_path = os.path.join(OUT_DIR, "multiscale_daily.npz")
    net.save(save_path)
    print(f"\n[Save] Daily multi-scale model written to {save_path}", flush=True)

    # One result per horizon, with the horizon expressed in trading days so it
    # can be merged with the intraday horizons on a common axis.
    return [{"label": f"{w}d", "days": float(w), "mean_vol": mv, "auc": a}
            for w, mv, a in zip(windows, mean_vol, aucs)]


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
