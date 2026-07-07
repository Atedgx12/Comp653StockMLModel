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

# ── Package root on sys.path ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ucn import UnifiedCourseNetwork, UCNConfig
from ucn.information_theory import select_features_by_mi
from ucn.training.metrics import accuracy, roc_auc
from ucn.data.ingestion import get_tickers, download_prices, download_volume, fetch_sentiment
from ucn.data.features import make_features

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
    p.add_argument("--patience",    type=int,   default=150)
    p.add_argument("--pgd-steps",   type=int,   default=5)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--noise-frac",  type=float, default=0.02)
    p.add_argument("--no-sent",     action="store_true")
    p.add_argument("--checkpoint",  default=None,
                   help="Load pretrained weights (.npz) before training")
    p.add_argument("--cv-only",     action="store_true",
                   help="Run walk-forward CV only, skip full retrain")
    p.add_argument("--n-cv-splits", type=int,   default=5)
    return p.parse_args()


# ── Walk-forward CV ───────────────────────────────────────────────────────

def walk_forward_cv(X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                    cfg: UCNConfig, n_splits: int = 5) -> tuple:
    unique_dates = np.sort(np.unique(dates))
    fold_size    = len(unique_dates) // (n_splits + 1)
    accs, aucs   = [], []

    for fold in range(n_splits):
        tr_end = unique_dates[(fold + 1) * fold_size]
        te_end = unique_dates[min((fold + 2) * fold_size, len(unique_dates) - 1)]
        tr_m   = dates < tr_end
        te_m   = (dates >= tr_end) & (dates < te_end)
        X_tr, X_te = X[tr_m], X[te_m]
        y_tr, y_te = y[tr_m], y[te_m]
        if len(X_tr) < 500 or len(X_te) < 50:
            continue

        mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-9
        X_tr_s = (X_tr - mu) / sd; X_te_s = (X_te - mu) / sd

        t0  = time.time()
        ucn = UnifiedCourseNetwork(cfg)
        ucn.fit(X_tr_s, y_tr)
        elapsed = time.time() - t0

        pred  = ucn.predict(X_te_s)
        prob  = ucn.predict_proba(X_te_s)[:, 1]
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


# ── LightGBM baseline ─────────────────────────────────────────────────────

def lgbm_baseline(X: np.ndarray, y: np.ndarray,
                  dates: np.ndarray) -> tuple:
    if not HAS_LGB:
        print("LightGBM not installed — skipping baseline.")
        return None
    unique_dates = np.sort(np.unique(dates))
    split_dt     = unique_dates[int(0.85 * len(unique_dates))]
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
    close   = download_prices(tickers, cache_dir=OUT_DIR)
    sent_df = None if args.no_sent else fetch_sentiment(
                  tickers, close.index, cache_dir=OUT_DIR)
    vol_df  = download_volume(tickers, cache_dir=OUT_DIR)

    # 2. Features + labels
    X_df, y_df, feat_names, has_sent = make_features(
        close, sent_df, vol_df)
    dates = X_df.index.values
    X_all = X_df.values.astype(np.float64)
    y_all = y_df.values.astype(int)

    # 3. MI feature selection (keep all; MI ranking for interpretability)
    price_names = [f for f in feat_names if f != "sent_rank"]
    price_idx   = [feat_names.index(f) for f in price_names]
    selected, mi_scores = select_features_by_mi(
        X_all[:, price_idx], y_all, price_names, k=len(price_names))
    sel_idx     = [price_names.index(f) for f in selected]
    X_sel_price = X_all[:, sel_idx]
    if has_sent:
        sent_col = X_all[:, feat_names.index("sent_rank")].reshape(-1, 1)
        X_sel    = np.hstack([X_sel_price, sent_col])
    else:
        X_sel = X_sel_price

    print(f"  Using {len(selected)} MI-selected features"
          + (" + VADER sentiment" if has_sent else ""), flush=True)

    # 4. Build config
    cfg_cv = UCNConfig(
        hidden_sizes=(64, 32),
        use_sent=has_sent,
        lr=args.lr,
        epochs=60,
        patience=60,
        pgd_steps=args.pgd_steps,
        noise_frac=args.noise_frac,
        verbose=0,
        use_fgsm=True,
    )
    cfg_full = UCNConfig(
        hidden_sizes=(256, 128, 64),
        use_sent=has_sent,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        pgd_steps=args.pgd_steps,
        noise_frac=args.noise_frac,
        verbose=20,
        use_fgsm=True,
    )

    # 5. Walk-forward CV (on first 25% of dates for speed)
    unique_dates = np.sort(np.unique(dates))
    cut    = unique_dates[int(0.25 * len(unique_dates))]
    sub_m  = dates <= cut
    print(f"\n[CV] Walk-forward on {sub_m.sum():,} rows ...", flush=True)
    cv_acc, cv_auc = walk_forward_cv(
        X_sel[sub_m], y_all[sub_m], dates[sub_m],
        cfg_cv, n_splits=args.n_cv_splits)

    pd.DataFrame([{"acc": cv_acc, "auc": cv_auc}],
                 index=["UCN"]).to_csv(
        os.path.join(OUT_DIR, "cv_results_unified.csv"))

    if args.cv_only:
        print("--cv-only flag set. Stopping after CV.")
        return

    # 6. Full retrain
    print("\n[Train] Full retrain on entire dataset ...", flush=True)
    split_dt = unique_dates[int(0.85 * len(unique_dates))]
    tr_m     = dates < split_dt; te_m = dates >= split_dt
    X_tr, X_te = X_sel[tr_m], X_sel[te_m]
    y_tr, y_te = y_all[tr_m], y_all[te_m]
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-9
    X_tr_s = (X_tr-mu)/sd; X_te_s = (X_te-mu)/sd

    ucn = (UnifiedCourseNetwork.from_checkpoint(args.checkpoint, cfg_full)
           if args.checkpoint and os.path.exists(args.checkpoint + ".npz")
           else UnifiedCourseNetwork(cfg_full))

    print(f"  train={tr_m.sum():,}  test={te_m.sum():,}", flush=True)
    ucn.fit(X_tr_s, y_tr)

    pred  = ucn.predict(X_te_s)
    prob  = ucn.predict_proba(X_te_s)[:, 1]
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
