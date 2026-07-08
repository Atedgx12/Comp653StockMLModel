"""
COMP 653 Stock Direction Prediction — Modular Entry Point
=========================================================
This replaces pipeline_course.py as the main training script.
All logic lives in the ucn/ package; this file is orchestration only.

Usage:
    python main.py                        # full pipeline, default config
    python main.py --epochs 500           # override epochs
    python main.py --no-sent              # disable sentiment branch
    python main.py --checkpoint weights   # load pretrained, skip cold start
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
from itertools import combinations

# ── Package root on sys.path ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ucn import UnifiedCourseNetwork, UCNConfig
from ucn.information_theory import select_features_by_mi
from ucn.training.metrics import accuracy, roc_auc
from ucn.training.weights import exponential_time_weights
from ucn.data.ingestion import get_tickers, download_prices, download_volume, fetch_sentiment
from ucn.data.features import make_features


def _load_sector_map(tickers):
    """Load or build the ticker to sector map for hierarchical context."""
    try:
        from ucn.data.market_context import load_or_build_sector_map
        return load_or_build_sector_map(tickers)
    except Exception:
        return None


def _learn_cluster_map(close, insample_end, n_clusters):
    """Learn data driven sector groups by clustering return co movement."""
    try:
        from ucn.data.market_context import build_correlation_clusters
        return build_correlation_clusters(
            close, n_clusters=n_clusters, insample_end=insample_end)
    except Exception:
        return None

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

OUT_DIR = ROOT


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="UCN training pipeline")
    p.add_argument("--epochs",      type=int,   default=3000)
    p.add_argument("--patience",    type=int,   default=200)
    p.add_argument("--pgd-steps",   type=int,   default=5)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--lam",         type=float, default=3e-4,
                   help="L2 weight decay. Increase to 1e-3 for long horizons.")
    p.add_argument("--dropout",     type=float, default=0.4,
                   help="MLP branch dropout. Increase to 0.6 for long horizons.")
    p.add_argument("--noise-frac",  type=float, default=0.02)
    p.add_argument("--horizon",     type=int,   default=1,
                   help="Forward-return window in trading days. "
                        "1=next day, 20=1mo, 63=3mo, 126=6mo.")
    p.add_argument("--stride",      type=int,   default=1,
                   help="Keep every N-th date to reduce label autocorrelation. "
                        "Recommended: horizon//10 for horizon>20.")
    p.add_argument("--start",       default="2010-01-01",
                   help="Download start date. Default 2010 for more regime diversity.")
    p.add_argument("--recent-weight", type=float, default=2.0,
                   help="Exponential decay for sample weights (0=uniform, 2=moderate, 4=strong). "
                        "Upweights recent data to counter regime drift.")
    p.add_argument("--shap-select", action="store_true", dest="shap_select",
                   help="Use LightGBM gain-based importance on recent data for feature "
                        "selection instead of all-data MI. Targets current-regime features.")
    p.add_argument("--no-sent",     action="store_true")
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--cv-only",     action="store_true")
    p.add_argument("--n-cv-splits", type=int,   default=5)
    p.add_argument("--use-store",   action="store_true",
                   help="Load features from DuckDB FeatureStore instead of "
                        "recomputing them. Requires --store-path.")
    p.add_argument("--store-path",  default="D:/StockModel/features.duckdb",
                   help="Path to the DuckDB feature store.")
    p.add_argument("--use-nomadic",  action="store_true",
                   help="Add 20 extended indicators from NomadicStockBot: CCI, Williams R, "
                        "OBV, CMF, MFI, ADX, Ichimoku, VWAP deviation, Donchian breakout, "
                        "BB squeeze release, RSI/MACD temporal derivatives.")
    p.add_argument("--use-lstm",    action="store_true",
                   help="Add LSTM Branch E that processes a lookback window "
                        "of past feature vectors (professor's RNN equation).")
    p.add_argument("--lstm-lookback", type=int, default=20,
                   help="Number of past trading days fed into the LSTM.")
    p.add_argument("--lstm-hidden",   type=int, default=32,
                   help="LSTM hidden state size.")
    p.add_argument("--use-hierarchy", action="store_true",
                   help="Add hierarchical market context: stock vs sector vs "
                        "broad market relative strength plus macro regime "
                        "descriptors (broad market trend and volatility).")
    p.add_argument("--cluster-hierarchy", action="store_true",
                   help="Learn the sector groups from return co movement by "
                        "clustering, instead of using yfinance GICS labels.")
    p.add_argument("--n-clusters", type=int, default=None,
                   help="Fixed number of learned clusters. Omit to let the "
                        "silhouette score choose the count automatically.")
    p.add_argument("--use-cpcv", action="store_true",
                   help="Use combinatorial purged cross validation instead of "
                        "a single expanding walk forward, giving many out of "
                        "sample paths with purge and embargo against leakage.")
    p.add_argument("--cpcv-groups", type=int, default=6,
                   help="Number of contiguous time blocks for CPCV.")
    p.add_argument("--cpcv-test-groups", type=int, default=2,
                   help="Number of blocks held out as test in each CPCV path.")
    p.add_argument("--embargo-frac", type=float, default=0.01,
                   help="Embargo band after each test block, as a fraction of "
                        "the number of unique dates.")
    return p.parse_args()


# ── Walk-forward CV ───────────────────────────────────────────────────────

def walk_forward_cv(X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                    cfg: UCNConfig, n_splits: int = 5,
                    sample_weights: np.ndarray = None,
                    seqs: np.ndarray = None) -> tuple:
    unique_dates = np.sort(np.unique(dates))
    fold_size    = len(unique_dates) // (n_splits + 1)
    accs, aucs   = [], []

    for fold in range(n_splits):
        tr_end = unique_dates[(fold + 1) * fold_size]
        te_end = unique_dates[min((fold + 2) * fold_size, len(unique_dates) - 1)]
        purge  = getattr(cfg, 'horizon', 1)
        purge_start = unique_dates[max(0, (fold + 1) * fold_size - purge)]
        tr_m   = dates < purge_start
        te_m   = (dates >= tr_end) & (dates < te_end)
        X_tr, X_te = X[tr_m], X[te_m]
        y_tr, y_te = y[tr_m], y[te_m]
        w_tr   = sample_weights[tr_m] if sample_weights is not None else None
        s_tr   = seqs[tr_m] if seqs is not None else None
        s_te   = seqs[te_m] if seqs is not None else None
        if len(X_tr) < 500 or len(X_te) < 50:
            continue

        mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-9
        X_tr_s = (X_tr - mu) / sd; X_te_s = (X_te - mu) / sd

        t0  = time.time()
        ucn = UnifiedCourseNetwork(cfg)
        ucn.fit(X_tr_s, y_tr, sample_weights=w_tr, seqs=s_tr)
        elapsed = time.time() - t0

        pred  = ucn.predict(X_te_s, seqs=s_te)
        prob  = ucn.predict_proba(X_te_s, seqs=s_te)[:, 1]
        acc   = accuracy(y_te, pred)
        auc   = roc_auc(y_te, prob)
        accs.append(acc); aucs.append(auc)
        print(f"  fold {fold+1}/{n_splits}  train={tr_m.sum():,}  "
              f"test={te_m.sum():,}  acc={acc:.4f}  auc={auc:.4f}  "
              f"time={elapsed:.1f}s", flush=True)

    m_acc = float(np.mean(accs)) if accs else 0.0
    m_auc = float(np.mean(aucs)) if aucs else 0.0
    print(f"  CV mean  acc={m_acc:.4f}  auc={m_auc:.4f}")
    return m_acc, m_auc


# ── Combinatorial Purged Cross-Validation ─────────────────────────────────

def purged_cpcv(X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                cfg: UCNConfig, n_groups: int = 6, n_test_groups: int = 2,
                embargo_frac: float = 0.01,
                sample_weights: np.ndarray = None,
                seqs: np.ndarray = None) -> tuple:
    """
    Combinatorial purged cross validation in the style of Lopez de Prado.

    The timeline is cut into n_groups contiguous blocks.  Every combination of
    n_test_groups blocks becomes a test set and the remaining blocks form the
    training set, which gives many out of sample paths instead of the single
    held out tail that a plain walk forward provides.  This measures how the
    model behaves across several distinct market periods rather than one.

    Two safeguards stop the future from leaking into the past.  Purging drops
    any training row whose forward return label window overlaps a test block,
    which matters here because the label looks horizon days ahead and would
    otherwise reach into a test period.  An embargo drops a short band of
    training rows immediately after each test block so that serial correlation
    right after the test window cannot leak backward into training.

    The blocks stay contiguous in time and are never randomly shuffled, so the
    temporal ordering that a 90 day forecast depends on is always respected.
    """
    unique_dates = np.sort(np.unique(dates))
    n = len(unique_dates)
    horizon = getattr(cfg, "horizon", 1)
    embargo = int(np.ceil(embargo_frac * n))

    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [unique_dates[bounds[g]:bounds[g + 1]] for g in range(n_groups)]

    combos = list(combinations(range(n_groups), n_test_groups))
    accs, aucs = [], []

    print(f"[CPCV] {n_groups} blocks, {n_test_groups} test per split, "
          f"{len(combos)} paths, purge={horizon}d, embargo={embargo}d",
          flush=True)

    for ci, test_groups in enumerate(combos):
        test_dates = np.concatenate([groups[g] for g in test_groups])
        te_m = np.isin(dates, test_dates)

        # Start from every row outside the test blocks, then purge and embargo.
        tr_m = ~te_m
        for g in test_groups:
            g_start_i = np.searchsorted(unique_dates, groups[g][0])
            g_end_i   = np.searchsorted(unique_dates, groups[g][-1])
            # Purge training rows whose label window reaches into this block.
            purge_lo = unique_dates[max(0, g_start_i - horizon)]
            purge_hi = unique_dates[min(n - 1, g_end_i)]
            # Embargo a short band immediately after the block.
            emb_hi   = unique_dates[min(n - 1, g_end_i + embargo)]
            drop = (dates >= purge_lo) & (dates <= emb_hi)
            tr_m &= ~drop

        X_tr, X_te = X[tr_m], X[te_m]
        y_tr, y_te = y[tr_m], y[te_m]
        w_tr = sample_weights[tr_m] if sample_weights is not None else None
        s_tr = seqs[tr_m] if seqs is not None else None
        s_te = seqs[te_m] if seqs is not None else None
        if len(X_tr) < 500 or len(X_te) < 50:
            continue

        mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-9
        X_tr_s = (X_tr - mu) / sd; X_te_s = (X_te - mu) / sd

        t0  = time.time()
        ucn = UnifiedCourseNetwork(cfg)
        ucn.fit(X_tr_s, y_tr, sample_weights=w_tr, seqs=s_tr)
        elapsed = time.time() - t0

        pred = ucn.predict(X_te_s, seqs=s_te)
        prob = ucn.predict_proba(X_te_s, seqs=s_te)[:, 1]
        acc  = accuracy(y_te, pred)
        auc  = roc_auc(y_te, prob)
        accs.append(acc); aucs.append(auc)
        tg = "+".join(str(g) for g in test_groups)
        print(f"  path {ci+1}/{len(combos)}  test-blocks={tg}  "
              f"train={tr_m.sum():,}  test={te_m.sum():,}  "
              f"acc={acc:.4f}  auc={auc:.4f}  time={elapsed:.1f}s", flush=True)

    m_acc = float(np.mean(accs)) if accs else 0.0
    m_auc = float(np.mean(aucs)) if aucs else 0.0
    s_auc = float(np.std(aucs)) if aucs else 0.0
    print(f"  CPCV mean  acc={m_acc:.4f}  auc={m_auc:.4f}  "
          f"auc_std={s_auc:.4f}  ({len(aucs)} paths)")
    return m_acc, m_auc


# ── LightGBM baseline ─────────────────────────────────────────────────────

def lgbm_baseline(X: np.ndarray, y: np.ndarray,
                  dates: np.ndarray) -> tuple:
    if not HAS_LGB:
        print("LightGBM not installed — skipping baseline.")
        return None
    unique_dates = np.sort(np.unique(dates))
    split_dt     = unique_dates[int(0.80 * len(unique_dates))]
    tr_m = dates < split_dt; te_m = dates >= split_dt
    X_tr, X_te = X[tr_m], X[te_m]; y_tr, y_te = y[tr_m], y[te_m]
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-9
    X_tr_s = (X_tr-mu)/sd; X_te_s = (X_te-mu)/sd

    dtrain = lgb.Dataset(X_tr_s, label=y_tr)
    dvalid = lgb.Dataset(X_te_s, label=y_te, reference=dtrain)
    params = dict(objective="binary", metric="binary_logloss",
                  num_leaves=63, max_depth=7, learning_rate=0.02,
                  min_child_samples=100, subsample=0.7, subsample_freq=1,
                  colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                  is_unbalance=True, device="gpu", seed=42, verbosity=-1)
    booster = lgb.train(params, dtrain, num_boost_round=1000,
                        valid_sets=[dtrain, dvalid],
                        valid_names=["train", "valid"],
                        callbacks=[lgb.early_stopping(75, verbose=False),
                                   lgb.log_evaluation(25)])
    prob = booster.predict(X_te_s)
    pred = (prob > 0.5).astype(int)
    acc  = accuracy(y_te, pred)
    auc  = roc_auc(y_te, prob)
    print(f"\n[LightGBM-GPU]  acc={acc:.4f}  auc={auc:.4f}")
    booster.save_model(os.path.join(OUT_DIR, "lgbm_full.txt"))
    return acc, auc


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("=" * 65)
    print(" COMP 653 — UCN Stock Direction Prediction (Modular)")
    print("=" * 65, flush=True)

    # 1. Data
    tickers = get_tickers()
    close   = download_prices(tickers, start=args.start, cache_dir=OUT_DIR)
    sent_df = None if args.no_sent else fetch_sentiment(
                  tickers, close.index, cache_dir=OUT_DIR)
    vol_df  = download_volume(tickers, cache_dir=OUT_DIR)

    # 2. Features + labels — load from FeatureStore or recompute
    if args.use_store:
        from ucn.data.store import FeatureStore
        print(f"[Features] Loading from FeatureStore: {args.store_path}",
              flush=True)
        store  = FeatureStore(args.store_path)
        X_all, y_all, date_vals = store.load_all(horizon=args.horizon)
        dates      = date_vals.astype("datetime64[D]")
        feat_names = [f"f{i}" for i in range(X_all.shape[1] - 1)] + ["sent_rank"]
        has_sent   = True
        store.summary()
        store.close()
    else:
        # Choose the sector grouping source for the hierarchical context.
        # Learned clusters are estimated only from the in sample window so
        # future co movement never leaks into the features.
        sector_map = None
        if args.use_hierarchy:
            if args.cluster_hierarchy:
                insample_end = str(pd.Timestamp(
                    close.index[len(close.index) // 2]).date())
                sector_map = _learn_cluster_map(
                    close, insample_end, args.n_clusters)
            else:
                sector_map = _load_sector_map(close.columns.tolist())

        X_df, y_df, feat_names, has_sent = make_features(
            close, sent_df, vol_df,
            horizon=args.horizon,
            stride=args.stride,
            use_nomadic=args.use_nomadic,
            use_hierarchy=args.use_hierarchy,
            sector_map=sector_map)
        dates = X_df.index.values
        # Extract ticker column for LSTM (not a training feature)
        ticker_ids = X_df.pop("_ticker").values if "_ticker" in X_df.columns else None
        X_all = X_df.values.astype(np.float64)
        y_all = y_df.values.astype(int)

    # 3. Feature selection
    price_names = [f for f in feat_names if f != "sent_rank"]
    price_idx   = [feat_names.index(f) for f in price_names]

    if args.shap_select and HAS_LGB:
        # Use LightGBM gain-based importance on the most recent 30% of data
        # to identify features predictive in the CURRENT regime.
        print("\n[Feature Selection] LightGBM gain-based (recent-regime) ...",
              flush=True)
        recent_cut = np.sort(np.unique(dates))[int(0.70 * len(np.unique(dates)))]
        recent_m   = dates >= recent_cut
        X_rec = X_all[recent_m][:, price_idx]
        y_rec = y_all[recent_m]
        if len(X_rec) > 500:
            dtmp  = lgb.Dataset(X_rec, label=y_rec)
            bst   = lgb.train(
                dict(objective="binary", num_leaves=31, learning_rate=0.05,
                     verbosity=-1, seed=42),
                dtmp, num_boost_round=100)
            imp    = dict(zip(price_names, bst.feature_importance("gain")))
            selected = sorted(imp, key=imp.get, reverse=True)
            print(f"  Top 10 by recent-regime LightGBM gain:")
            for f in selected[:10]:
                print(f"    {f:30s}  gain={imp[f]:.1f}")
        else:
            selected, _ = select_features_by_mi(
                X_all[:, price_idx], y_all, price_names, k=len(price_names))
    else:
        selected, mi_scores = select_features_by_mi(
            X_all[:, price_idx], y_all, price_names, k=len(price_names))

    sel_idx     = [price_names.index(f) for f in selected]
    X_sel_price = X_all[:, sel_idx]
    if has_sent:
        sent_col = X_all[:, feat_names.index("sent_rank")].reshape(-1, 1)
        X_sel    = np.hstack([X_sel_price, sent_col])
    else:
        X_sel = X_sel_price

    print(f"  Using {len(selected)} features"
          + (" + VADER sentiment" if has_sent else ""), flush=True)

    # 4. Build config
    cfg_cv = UCNConfig(
        hidden_sizes=(64, 32),
        use_sent=has_sent,
        lr=args.lr,
        lam=args.lam,
        epochs=60,
        patience=60,
        pgd_steps=args.pgd_steps,
        noise_frac=args.noise_frac,
        dropout_rate=args.dropout,
        meta_dropout=min(args.dropout * 0.5, 0.3),
        verbose=0,
        use_fgsm=True,
        use_lstm=args.use_lstm,
        lstm_lookback=args.lstm_lookback,
        lstm_hidden=args.lstm_hidden,
    )
    cfg_full = UCNConfig(
        hidden_sizes=(256, 128, 64),
        use_sent=has_sent,
        lr=args.lr,
        lam=args.lam,
        epochs=args.epochs,
        patience=args.patience,   # default 200
        pgd_steps=args.pgd_steps,
        noise_frac=args.noise_frac,
        dropout_rate=args.dropout,
        meta_dropout=min(args.dropout * 0.5, 0.3),
        verbose=20,
        use_fgsm=True,
        use_lstm=args.use_lstm,
        lstm_lookback=args.lstm_lookback,
        lstm_hidden=args.lstm_hidden,
    )

    # 5a. Compute sample weights early — needed by LSTM alignment and CV
    unique_dates   = np.sort(np.unique(dates))
    sample_weights = (exponential_time_weights(dates, decay=args.recent_weight)
                      if args.recent_weight > 0 else None)

    # 5b. Build LSTM sequence tensor if requested
    seqs_all = None
    if args.use_lstm:
        from ucn.models.lstm import build_sequences
        print(f"\n[LSTM] Building sequence tensor "
              f"(lookback={args.lstm_lookback}) ...", flush=True)
        # Build per-ticker sequences using the ticker identity column
        X_tmp = pd.DataFrame(X_sel, index=dates)
        if ticker_ids is not None:
            # Align ticker_ids to current X_sel (after MI selection dropped columns)
            if len(ticker_ids) == len(X_sel):
                X_tmp["_ticker"] = ticker_ids
            else:
                X_tmp["_ticker"] = ticker_ids[seq_mask] if 'seq_mask' in dir() else ticker_ids
        seqs_all, seq_mask = build_sequences(X_tmp, lookback=args.lstm_lookback)
        # Align X_sel, y_all, dates, sample_weights to valid sequence rows
        X_sel          = X_sel[seq_mask]
        y_all          = y_all[seq_mask]
        dates          = dates[seq_mask]
        if ticker_ids is not None:
            ticker_ids = ticker_ids[seq_mask]
        if sample_weights is not None:
            sample_weights = sample_weights[seq_mask]
        print(f"  Sequence tensor: {seqs_all.shape}  "
              f"(dropped {(~seq_mask).sum()} rows without full history)",
              flush=True)

    # 5c. Walk-forward CV — recompute unique_dates after any LSTM alignment
    unique_dates = np.sort(np.unique(dates))   # re-derived from current dates

    cut    = unique_dates[int(0.50 * len(unique_dates))]
    sub_m  = dates <= cut
    w_sub  = sample_weights[sub_m] if sample_weights is not None else None
    seqs_sub = seqs_all[sub_m] if seqs_all is not None else None

    if args.use_cpcv:
        print(f"\n[CV] Combinatorial purged CV on {sub_m.sum():,} rows "
              f"(first 50% of dates) ...", flush=True)
        cv_acc, cv_auc = purged_cpcv(
            X_sel[sub_m], y_all[sub_m], dates[sub_m],
            cfg_cv, n_groups=args.cpcv_groups,
            n_test_groups=args.cpcv_test_groups,
            embargo_frac=args.embargo_frac,
            sample_weights=w_sub,
            seqs=seqs_sub)
    else:
        print(f"\n[CV] Walk-forward on {sub_m.sum():,} rows "
              f"(first 50% of dates) ...", flush=True)
        cv_acc, cv_auc = walk_forward_cv(
            X_sel[sub_m], y_all[sub_m], dates[sub_m],
            cfg_cv, n_splits=args.n_cv_splits,
            sample_weights=w_sub,
            seqs=seqs_sub)

    pd.DataFrame([{"acc": cv_acc, "auc": cv_auc}],
                 index=["UCN"]).to_csv(
        os.path.join(OUT_DIR, "cv_results_unified.csv"))

    if args.cv_only:
        print("--cv-only flag set. Stopping after CV.")
        return

    # 6. Full retrain — 80/20 temporal split
    print("\n[Train] Full retrain on entire dataset ...", flush=True)
    split_dt = unique_dates[int(0.80 * len(unique_dates))]
    tr_m     = dates < split_dt; te_m = dates >= split_dt
    X_tr, X_te = X_sel[tr_m], X_sel[te_m]
    y_tr, y_te = y_all[tr_m], y_all[te_m]
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-9
    X_tr_s = (X_tr-mu)/sd; X_te_s = (X_te-mu)/sd

    ucn = (UnifiedCourseNetwork.from_checkpoint(args.checkpoint, cfg_full)
           if args.checkpoint and os.path.exists(args.checkpoint + ".npz")
           else UnifiedCourseNetwork(cfg_full))

    print(f"  train={tr_m.sum():,}  test={te_m.sum():,}", flush=True)

    # Slice the already-computed sample weights to the training rows
    w_tr   = sample_weights[tr_m] if sample_weights is not None else None
    seqs_tr = seqs_all[tr_m] if seqs_all is not None else None
    seqs_te = seqs_all[te_m] if seqs_all is not None else None
    if w_tr is not None:
        print(f"  Time weights: decay={args.recent_weight}  "
              f"recent/old ratio={w_tr.max()/w_tr.min():.1f}x", flush=True)
    if seqs_tr is not None:
        print(f"  LSTM sequences: train={seqs_tr.shape}  test={seqs_te.shape}",
              flush=True)
    ucn.fit(X_tr_s, y_tr, sample_weights=w_tr, seqs=seqs_tr)

    pred  = ucn.predict(X_te_s, seqs=seqs_te)
    prob  = ucn.predict_proba(X_te_s, seqs=seqs_te)[:, 1]
    acc   = accuracy(y_te, pred)
    auc   = roc_auc(y_te, prob)
    print(f"  Test  acc={acc:.4f}  auc={auc:.4f}")

    # Save checkpoint for fine-tuning
    ucn.save_checkpoint(os.path.join(OUT_DIR, "ucn_weights.npz"))
    ucn.branch_summary()

    # 7. LightGBM baseline
    print("\n[Baseline] LightGBM-GPU ...", flush=True)
    lgb_res = lgbm_baseline(X_sel, y_all, dates)

    # 8. Summary
    rows = {"UnifiedCourseNetwork (PGD)": {"acc": acc, "auc": auc}}
    if lgb_res:
        rows["LightGBM-GPU"] = {"acc": lgb_res[0], "auc": lgb_res[1]}
    df = pd.DataFrame(rows).T
    print("\n=== Final Results ===")
    print(df.to_string())
    df.to_csv(os.path.join(OUT_DIR, "final_results_unified.csv"))
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
