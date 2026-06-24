"""
Stock Direction Prediction Pipeline - Full S&P 500 Universe
COMP 653 Statistical Machine Learning, Summer 2026
Zachary Powell  zp21@rice.edu

Cross-sectional approach: every (date, ticker) pair is one sample.
Target: next-day return > 0  (1 = up, 0 = down)
~500 tickers x 2800 trading days = ~1.4M samples
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START    = "2015-01-01"
END      = datetime.today().strftime("%Y-%m-%d")
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
SEED     = 42
N_SPLITS = 5

# ---------------------------------------------------------------------------
# 1. S&P 500 ticker list
# ---------------------------------------------------------------------------

def get_sp500_tickers():
    """Pull current S&P 500 constituents from Wikipedia."""
    print("[1] Fetching S&P 500 constituent list from Wikipedia ...")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"    Found {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"    Wikipedia fetch failed ({e}), using fallback 100-ticker list.")
        # Fallback: large-cap subset
        return [
            "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","LLY","JPM","AVGO",
            "XOM","TSLA","UNH","V","JNJ","MA","PG","COST","HD","MRK","CVX","ABBV",
            "WMT","BAC","CRM","NFLX","AMD","ACN","MCD","PEP","TMO","ADBE","LIN","TXN",
            "QCOM","DIS","GE","VZ","CSCO","INTU","IBM","CAT","AXP","SPGI","RTX","PLD",
            "HON","AMGN","GILD","ISRG","BKNG","SYK","VRTX","REGN","KO","PFE","CI",
            "MDT","USB","CME","BLK","DE","MMM","BSX","GS","MS","MO","ELV","NOW",
            "INTC","PYPL","SBUX","LRCX","ADI","KLAC","MCHP","AMAT","SNPS","CDNS",
            "ZTS","MDLZ","HCA","EOG","SLB","WM","FDX","ITW","APH","ADP","ECL",
            "ORLY","CTAS","NKE","LOW","SPY","QQQ","IWM","DIA","GLD","TLT","HYG","LQD",
        ]

# ---------------------------------------------------------------------------
# 2. Download price data in batches
# ---------------------------------------------------------------------------

def download_prices(tickers, start=START, end=END, batch_size=50):
    cache = os.path.join(OUT_DIR, "close_cache_full.parquet")
    if os.path.exists(cache):
        print("[2] Loading cached price data ...")
        close = pd.read_parquet(cache)
        print(f"    Loaded {close.shape[1]} tickers x {close.shape[0]} days.")
        return close

    print(f"[2] Downloading {len(tickers)} tickers in batches of {batch_size} ...")
    frames = []
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    for bi, batch in enumerate(batches):
        t0 = time.time()
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                c = raw["Close"]
            else:
                c = raw[["Close"]] if "Close" in raw.columns else raw
            frames.append(c)
        except Exception as e:
            print(f"    Batch {bi+1}/{len(batches)} failed: {e}")
            continue
        elapsed = time.time() - t0
        print(f"    Batch {bi+1}/{len(batches)} done ({len(batch)} tickers, {elapsed:.1f}s)")

    close = pd.concat(frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    close = close.dropna(axis=1, thresh=int(0.7 * len(close)))   # drop tickers with >30% missing
    close = close.fillna(method="ffill").dropna()
    close.to_parquet(cache)
    print(f"    Saved {close.shape[1]} tickers x {close.shape[0]} days to cache.")
    return close

# ---------------------------------------------------------------------------
# 3. Feature engineering — cross-sectional (one row per ticker per day)
# ---------------------------------------------------------------------------

def make_features(close: pd.DataFrame):
    """
    For each ticker build time-series features.
    Stack all tickers into a single flat (date x ticker, features) matrix.
    """
    print("[3] Engineering features (cross-sectional) ...")
    all_X, all_y = [], []
    tickers = close.columns.tolist()
    n = len(tickers)

    for i, ticker in enumerate(tickers):
        if (i+1) % 50 == 0 or i == n-1:
            print(f"    {i+1}/{n} tickers processed ...", flush=True)

        c = close[ticker].dropna()
        if len(c) < 100:
            continue

        feat = {}
        for lag in [1, 2, 3, 5, 10, 20]:
            feat[f"ret{lag}"] = np.log(c / c.shift(lag))
        for w in [5, 10, 20, 60]:
            feat[f"vol{w}"] = np.log(c / c.shift(1)).rolling(w).std()
        for m in [5, 10, 20, 60]:
            feat[f"mom{m}"] = (c - c.shift(m)) / c.shift(m)
        # RSI-14
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        feat["rsi14"] = 100 - 100 / (1 + rs)
        # Distance from 52-week high/low
        feat["dist52h"] = c / c.rolling(252).max() - 1
        feat["dist52l"] = c / c.rolling(252).min() - 1
        # Volume ratio (if available — skip if not)
        feat["ticker_id"] = float(i)   # cross-sectional identifier

        df = pd.DataFrame(feat, index=c.index)
        fwd = np.log(c / c.shift(1)).shift(-1)
        y   = (fwd > 0).astype(int)

        df, y = df.align(y, join="inner")
        df    = df.dropna()
        y     = y.loc[df.index].dropna()
        df    = df.loc[y.index]

        all_X.append(df)
        all_y.append(y)

    X = pd.concat(all_X, axis=0).sort_index()
    y = pd.concat(all_y, axis=0).sort_index()
    X, y = X.align(y, join="inner", axis=0)

    print(f"    Feature matrix: {X.shape}  |  base rate: {y.mean():.4f}")
    return X, y

# ---------------------------------------------------------------------------
# 4. Walk-forward evaluation — date-based splits
# ---------------------------------------------------------------------------

def walk_forward_eval_date(X, y, model_fn, model_name, n_splits=N_SPLITS):
    """
    Split by calendar time so all observations on the same date stay together.
    """
    unique_dates = np.sort(X.index.unique())
    fold_size    = len(unique_dates) // (n_splits + 1)
    accs, aucs   = [], []

    print(f"\n  [{model_name}] Walk-forward CV ({n_splits} folds) ...")
    for fold in range(n_splits):
        train_end   = unique_dates[(fold + 1) * fold_size]
        test_start  = train_end
        test_end    = unique_dates[min((fold + 2) * fold_size, len(unique_dates)-1)]

        tr_mask = X.index < train_end
        te_mask = (X.index >= test_start) & (X.index < test_end)

        X_tr, X_te = X[tr_mask], X[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]

        if len(X_tr) < 1000 or len(X_te) < 100:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        t0    = time.time()
        model = model_fn()
        model.fit(X_tr_s, y_tr)
        elapsed = time.time() - t0

        pred = model.predict(X_te_s)
        prob = model.predict_proba(X_te_s)[:, 1]
        acc  = accuracy_score(y_te, pred)
        auc  = roc_auc_score(y_te, prob)
        accs.append(acc);  aucs.append(auc)
        print(f"    fold {fold+1}/{n_splits}  "
              f"train={len(X_tr):,}  test={len(X_te):,}  "
              f"acc={acc:.4f}  auc={auc:.4f}  "
              f"time={elapsed:.1f}s", flush=True)

    mean_acc = np.mean(accs) if accs else 0
    mean_auc = np.mean(aucs) if aucs else 0
    print(f"  [{model_name}] mean acc={mean_acc:.4f}  mean auc={mean_auc:.4f}\n", flush=True)
    return mean_acc, mean_auc

# ---------------------------------------------------------------------------
# 5. Model factories
# ---------------------------------------------------------------------------

def lr_factory():
    return LogisticRegression(max_iter=1000, C=0.05, solver="saga",
                               n_jobs=-1, random_state=SEED,
                               class_weight="balanced")

def rf_factory():
    return RandomForestClassifier(n_estimators=300, max_depth=7,
                                   min_samples_leaf=50, n_jobs=-1,
                                   random_state=SEED, class_weight="balanced",
                                   verbose=0)

def lgbm_factory(verbose_eval=50):
    return lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=7,
        min_child_samples=100,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        class_weight="balanced",
        device="gpu",
        verbose=-1,
        # print progress every verbose_eval rounds during fit via callbacks
    )

# ---------------------------------------------------------------------------
# 6. Final retrain on full data with verbose epoch output
# ---------------------------------------------------------------------------

def retrain_final(X, y):
    print("[5] Retraining final LightGBM on full dataset with verbose training ...")
    split    = int(len(X.index.unique()) * 0.85)
    split_dt = np.sort(X.index.unique())[split]

    X_tr = X[X.index < split_dt]
    X_te = X[X.index >= split_dt]
    y_tr = y[y.index < split_dt]
    y_te = y[y.index >= split_dt]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"    Train: {len(X_tr):,} samples  |  Test: {len(X_te):,} samples")
    print(f"    Training LightGBM (GPU) — printing every 25 rounds ...\n", flush=True)

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=7,
        min_child_samples=100,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        class_weight="balanced",
        device="gpu",
        verbose=-1,
    )

    dtrain = lgb.Dataset(X_tr_s, label=y_tr.values)
    dvalid = lgb.Dataset(X_te_s, label=y_te.values, reference=dtrain)

    params = {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "num_leaves":       63,
        "max_depth":        7,
        "learning_rate":    0.02,
        "min_child_samples":100,
        "subsample":        0.7,
        "subsample_freq":   1,
        "colsample_bytree": 0.7,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "is_unbalance":     True,
        "device":           "gpu",
        "seed":             SEED,
        "verbosity":        -1,
    }

    t0     = time.time()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(75, verbose=False),
            lgb.log_evaluation(25),   # print every 25 rounds to stdout
        ],
    )
    elapsed = time.time() - t0
    print(f"\n    Training complete in {elapsed:.1f}s  |  best iteration: {booster.best_iteration}")

    # Predict with raw booster
    prob = booster.predict(X_te_s)
    pred = (prob > 0.5).astype(int)

    print("\n=== Final Model Test-Set Results ===")
    print(f"  Accuracy : {accuracy_score(y_te, pred):.4f}")
    print(f"  AUC      : {roc_auc_score(y_te, prob):.4f}")
    print(classification_report(y_te, pred, target_names=["Down", "Up"]))

    # Save
    booster.save_model(os.path.join(OUT_DIR, "lgbm_booster_full.txt"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler_full.pkl"))

    # Feature importance
    fi = pd.Series(booster.feature_importance(importance_type="gain"),
                   index=X.columns)
    top20 = fi.nlargest(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    top20.sort_values().plot.barh(ax=ax)
    ax.set_title("LightGBM Feature Importance — Gain (Top 20)")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_importance_full.png"), dpi=150)

    # Cumulative return (SPY only for comparison)
    ret_spy = np.log(X_te.groupby(X_te.index).first()
                     .assign(spy_ret=lambda df: df["ret1"])["spy_ret"])
    ret_spy = ret_spy.dropna()
    strat   = ret_spy * (2 * pd.Series(
                  pred[:len(ret_spy)], index=ret_spy.index) - 1)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    (ret_spy.cumsum().apply(np.exp) - 1).plot(ax=ax2, label="SPY ret (proxy)")
    (strat.cumsum().apply(np.exp) - 1).plot(ax=ax2, label="Model long/short")
    ax2.set_title("Cumulative Return: Test Period")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "cumulative_return_full.png"), dpi=150)
    print("  Saved plots and model artifacts to", OUT_DIR)

    return booster, scaler

# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print(" Stock Direction Prediction — Full S&P 500 Pipeline")
    print("=" * 60, flush=True)

    tickers = get_sp500_tickers()
    close   = download_prices(tickers)
    X, y    = make_features(close)

    # Baseline models on a 20% sample to save time
    sample_dates = np.sort(X.index.unique())
    sample_cut   = sample_dates[int(0.2 * len(sample_dates))]
    X_sample     = X[X.index <= sample_cut]
    y_sample     = y[y.index <= sample_cut]
    print(f"\n[4] Running baseline CV on {len(X_sample):,} sample rows ...")

    results = {}
    for name, fn in [("LogisticRegression", lr_factory),
                     ("RandomForest",       rf_factory),
                     ("LightGBM-GPU",       lgbm_factory)]:
        acc, auc = walk_forward_eval_date(X_sample, y_sample, fn, name, n_splits=3)
        results[name] = {"acc": acc, "auc": auc}

    print("\n=== Baseline CV Summary ===")
    df_res = pd.DataFrame(results).T
    print(df_res.to_string())
    df_res.to_csv(os.path.join(OUT_DIR, "cv_results_full.csv"))

    # Full retrain with verbose epoch output
    booster, scaler = retrain_final(X, y)
    print("\nAll done.")
