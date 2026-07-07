import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_course import mutual_information

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
         "SOL-USD", "DOGE-USD", "LTC-USD"]
HORIZON = 1                  
EPOCHS, LR = 3000, 1e-6


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-9))


def features(c):
    df = pd.DataFrame({
        "ret1":    c.pct_change(1),
        "ret5":    c.pct_change(5),
        "ret10":   c.pct_change(10),
        "ret20":   c.pct_change(20),
        "mom20":   c / c.shift(20) - 1,
        "vol20":   c.pct_change().rolling(20).std(),
        "rsi14":   rsi(c, 14),
        "dist_hi": c / c.rolling(60).max() - 1,
        "dist_lo": c / c.rolling(60).min() - 1,
    })
    y = (c.shift(-HORIZON) > c).astype(float)
    return df, y


def fit_logistic(X, y):                     
    Xi = np.column_stack((X, np.ones(len(X))))
    beta = np.zeros(Xi.shape[1])
    for _ in range(EPOCHS):
        p = 1 / (1 + np.exp(-np.clip(Xi @ beta, -30, 30)))
        beta = beta - LR * (Xi.T @ (p - y))
    return lambda Z: 1 / (1 + np.exp(
        -np.clip(np.column_stack((Z, np.ones(len(Z)))) @ beta, -30, 30)))


def load_prices():
    import yfinance as yf
    px = yf.download(COINS, start="2018-01-01", auto_adjust=True,
                     progress=False)["Close"]
    return px.dropna(how="all")


def main():
    px = load_prices()
    print(f"\ncoins: {px.shape[1]}   days: {px.shape[0]}   horizon: {HORIZON}")

    Xs, ys, dts = [], [], []
    for coin in px.columns:
        f, y = features(px[coin].dropna())
        df = f.copy(); df["_y"] = y; df = df.dropna()
        Xs.append(df.drop(columns="_y").values)
        ys.append(df["_y"].values)
        dts.append(df.index.values)
    names = list(f.columns)
    X = np.vstack(Xs); y = np.concatenate(ys); dates = np.concatenate(dts)
    print(f"rows: {len(y):,}   base rate (up): {y.mean():.4f}  "
          f"(always-up baseline = {max(y.mean(),1-y.mean()):.4f})")

    u = np.sort(np.unique(dates)); cut = u[int(0.7 * len(u))]
    tr, te = dates < cut, dates >= cut
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    p = fit_logistic(Xtr, y[tr])(Xte)

    acc = np.mean((p >= 0.5) == y[te])
    best_in = max(mutual_information(Xte[:, j], y[te]) for j in range(Xte.shape[1]))
    out_mi = mutual_information(p, y[te])
    print(f"\n  logistic accuracy     = {acc:.4f}")
    print(f"  always-up baseline    = {max(y[te].mean(),1-y[te].mean()):.4f}")
    print(f"  best input feature MI = {best_in:.4f} bits")
    print(f"  model output MI       = {out_mi:.4f} bits")


if __name__ == "__main__":
    main()
