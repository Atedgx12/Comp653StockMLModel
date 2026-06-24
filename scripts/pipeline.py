"""
Stock Direction Prediction Pipeline
COMP 653 Statistical Machine Learning, Summer 2026
Zachary Powell  zp21@rice.edu

Predicts next-day directional movement (up/down) for a basket of equities
using technical features, logistic regression, random forest, and LightGBM.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKERS   = ["SPY", "AAPL", "MSFT", "AMD", "JPM"]
START     = "2015-01-01"
END       = datetime.today().strftime("%Y-%m-%d")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
SEED      = 42
N_SPLITS  = 5          # walk-forward CV folds

# ---------------------------------------------------------------------------
# 1. Data download
# ---------------------------------------------------------------------------

def download_data(tickers=TICKERS, start=START, end=END):
    print(f"\n[1] Downloading {tickers} from {start} to {end} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"].dropna(how="all")
    return close


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def make_features(close: pd.DataFrame, target_ticker="SPY"):
    """
    Builds a feature matrix from closing prices.
    Features for each ticker:
        ret_1, ret_5, ret_10, ret_20        log returns over N days
        vol_10, vol_20                      rolling volatility
        mom_5, mom_20                       signed momentum
        rsi_14                              relative strength index
    Target: next-day sign of SPY return  (1 = up, 0 = down)
    """
    frames = []
    for ticker in close.columns:
        c = close[ticker]
        for lag in [1, 5, 10, 20]:
            frames.append(np.log(c / c.shift(lag)).rename(f"{ticker}_ret{lag}"))
        for w in [10, 20]:
            frames.append(np.log(c / c.shift(1)).rolling(w).std()
                          .rename(f"{ticker}_vol{w}"))
        for m in [5, 20]:
            frames.append(((c - c.shift(m)) / c.shift(m))
                          .rename(f"{ticker}_mom{m}"))
        # RSI-14
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        frames.append((100 - 100 / (1 + rs)).rename(f"{ticker}_rsi14"))

    X = pd.concat(frames, axis=1).dropna()

    # Target: next-day return of the target ticker
    fwd = np.log(close[target_ticker] / close[target_ticker].shift(1)).shift(-1)
    y   = (fwd > 0).astype(int)

    X, y = X.align(y, join="inner", axis=0)
    y    = y.dropna()
    X    = X.loc[y.index]

    print(f"[2] Feature matrix: {X.shape}  |  base rate: {y.mean():.4f}")
    return X, y


# ---------------------------------------------------------------------------
# 3. Walk-forward train / test split
# ---------------------------------------------------------------------------

def walk_forward_eval(X, y, model_fn, model_name, n_splits=N_SPLITS):
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    accs, aucs = [], []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = model_fn()
        model.fit(X_tr_s, y_tr)

        pred  = model.predict(X_te_s)
        prob  = model.predict_proba(X_te_s)[:, 1]
        acc   = accuracy_score(y_te, pred)
        auc   = roc_auc_score(y_te, prob)
        accs.append(acc);  aucs.append(auc)
        print(f"  [{model_name}] fold {fold+1}/{n_splits}  acc={acc:.4f}  auc={auc:.4f}")

    print(f"  [{model_name}] mean acc={np.mean(accs):.4f}  mean auc={np.mean(aucs):.4f}\n")
    return np.mean(accs), np.mean(aucs)


# ---------------------------------------------------------------------------
# 4. Model factories
# ---------------------------------------------------------------------------

def lr_factory():
    return LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs",
                               random_state=SEED, class_weight="balanced")

def rf_factory():
    return RandomForestClassifier(n_estimators=400, max_depth=6,
                                   min_samples_leaf=20,
                                   n_jobs=-1, random_state=SEED,
                                   class_weight="balanced")

def lgbm_factory():
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        class_weight="balanced",
        device="gpu",          # use the RTX 5080
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# 5. Final retrain on full dataset
# ---------------------------------------------------------------------------

def retrain_final(X, y):
    print("[4] Retraining final LightGBM model on full dataset ...")
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = lgbm_factory()
    model.fit(
        X_tr_s, y_tr,
        eval_set=[(X_te_s, y_te)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )

    pred = model.predict(X_te_s)
    prob = model.predict_proba(X_te_s)[:, 1]
    print("\n=== Final Model Test-Set Results ===")
    print(f"  Accuracy : {accuracy_score(y_te, pred):.4f}")
    print(f"  AUC      : {roc_auc_score(y_te, prob):.4f}")
    print(classification_report(y_te, pred, target_names=["Down","Up"]))

    # Save artifacts
    joblib.dump(model,  os.path.join(OUT_DIR, "lgbm_final.pkl"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler_final.pkl"))
    print(f"  Saved model and scaler to {OUT_DIR}")

    # Feature importance plot
    feat_imp = pd.Series(model.feature_importances_, index=X.columns)
    top20    = feat_imp.nlargest(20)
    fig, ax  = plt.subplots(figsize=(10, 6))
    top20.sort_values().plot.barh(ax=ax)
    ax.set_title("LightGBM Feature Importances (Top 20)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    print("  Saved feature_importance.png")

    # Cumulative return plot
    returns = np.log(
        pd.read_parquet(os.path.join(OUT_DIR, "close_cache.parquet"))["SPY"]
        / pd.read_parquet(os.path.join(OUT_DIR, "close_cache.parquet"))["SPY"].shift(1)
    ).dropna()
    test_dates = X_te.index
    strat_ret  = returns.loc[test_dates] * (2 * pd.Series(pred, index=test_dates) - 1)
    cum_strat  = strat_ret.cumsum().apply(np.exp) - 1
    cum_bh     = returns.loc[test_dates].cumsum().apply(np.exp) - 1

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    cum_bh.plot(ax=ax2, label="Buy & Hold SPY")
    cum_strat.plot(ax=ax2, label="Model Long/Short")
    ax2.set_title("Cumulative Return: Model vs Buy-and-Hold (Test Set)")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "cumulative_return.png"), dpi=150)
    print("  Saved cumulative_return.png")

    return model, scaler


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Download & cache
    cache_path = os.path.join(OUT_DIR, "close_cache.parquet")
    if os.path.exists(cache_path):
        print("[1] Loading cached price data ...")
        close = pd.read_parquet(cache_path)
    else:
        close = download_data()
        close.to_parquet(cache_path)

    # Features
    X, y = make_features(close)

    # Walk-forward evaluation
    print("\n[3] Walk-forward cross-validation ...")
    results = {}
    for name, fn in [("LogisticRegression", lr_factory),
                     ("RandomForest",       rf_factory),
                     ("LightGBM-GPU",       lgbm_factory)]:
        acc, auc = walk_forward_eval(X, y, fn, name)
        results[name] = {"acc": acc, "auc": auc}

    print("\n=== Cross-Validation Summary ===")
    df_res = pd.DataFrame(results).T
    print(df_res.to_string())
    df_res.to_csv(os.path.join(OUT_DIR, "cv_results.csv"))

    # Final retrain
    model, scaler = retrain_final(X, y)
    print("\nDone.")
