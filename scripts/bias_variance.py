import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_course import download_prices, get_sp500_tickers, make_features

B = 20          
EPOCHS = 300
LR = 0.1


def fit_logistic(X, y):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(EPOCHS):
        p = 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g = (p - y) / len(y)
        w -= LR * (X.T @ g); b -= LR * g.sum()
    return w, b


def proba(X, w, b):
    return 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))


def main():
    close = download_prices(get_sp500_tickers())
    X_df, y_df, _, _ = make_features(close, sent_df=None)
    X = X_df.values.astype(float); y = y_df.values.astype(float)
    dates = X_df.index.values

 
    uniq = np.sort(np.unique(dates)); cut = uniq[int(0.7 * len(uniq))]
    tr, te = dates < cut, dates >= cut
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xtr, ytr = (X[tr] - mu) / sd, y[tr]
    Xte, yte = (X[te] - mu) / sd, y[te]

 
    rng = np.random.default_rng(0); n = len(ytr)
    preds = np.empty((B, len(yte)))
    for b in range(B):
        idx = rng.integers(0, n, n)
        w, b0 = fit_logistic(Xtr[idx], ytr[idx])
        preds[b] = proba(Xte, w, b0)


    fbar = preds.mean(0)
    variance = np.mean((preds - fbar) ** 2)
    systematic = np.mean((fbar - yte) ** 2)       
    total = np.mean((preds - yte) ** 2)          
    noise = yte.mean() * (1 - yte.mean())          
    bias2 = max(systematic - noise, 0.0)

    print(f"\ntest rows: {len(yte):,}   base rate: {yte.mean():.4f}\n")
    print(f"  Variance   = {variance:.4f}")
    print(f"  Bias^2     = {bias2:.4f}")
    print(f"  Noise      = {noise:.4f}")
    print(f"  ---------------------")
    print(f"  Total      = {total:.4f}   (= Variance + Bias^2 + Noise)")


if __name__ == "__main__":
    main()
