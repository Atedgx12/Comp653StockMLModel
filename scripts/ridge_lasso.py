import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_course import (download_prices, get_sp500_tickers, make_features,
                             select_features_by_mi)

LR, EPOCHS = 1e-7, 3000


def ridge(X, y, lam):
    b = np.zeros(X.shape[1])
    for _ in range(EPOCHS):
        e = y - X @ b
        b = b + LR * (X.T @ e) - LR * lam * b
    return b


def lasso(X, y, lam):
    b = np.zeros(X.shape[1])
    for _ in range(EPOCHS):
        e = y - X @ b
        b = b + LR * (X.T @ e) - LR * lam * np.sign(b)
    return b


def main():
    close = download_prices(get_sp500_tickers())
    X_df, y_df, names, _ = make_features(close, sent_df=None)
    X = X_df.values.astype(float);
    y = y_df.values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.column_stack((X, np.ones(len(X))))
    print(f"\nrows: {len(y):,}   features: {len(names)}")


    br = ridge(X, y, 5000)
    bl = lasso(X, y, 5000)
    print(f"\nat lambda = 5000:")
    print(f"  ridge nonzero = {np.sum(np.abs(br[:-1]) > 1e-3):>2}/{len(names)}  (L2 shrinks, keeps all)")
    print(f"  lasso nonzero = {np.sum(np.abs(bl[:-1]) > 1e-3):>2}/{len(names)}  (L1 zeros some)")


    k = 6
    lasso_top = [names[i] for i in np.argsort(-np.abs(bl[:-1]))[:k]]
    mi_top = select_features_by_mi(X[:, :-1], y, names, k=k)
    print(f"\n  lasso top-{k}: {lasso_top}")
    print(f"  MI    top-{k}: {mi_top}")
    print(f"  in both     : {sorted(set(lasso_top) & set(mi_top))}")


if __name__ == "__main__":
    main()
