"""
Feature engineering and cross-sectional label construction.
Extracted from pipeline_course.py make_features().
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


def make_features(
    close: pd.DataFrame,
    sent_df: Optional[pd.DataFrame] = None,
    vol_df: Optional[pd.DataFrame] = None,
    top_pct: float = 0.20,
    bottom_pct: float = 0.20,
    min_history: int = 300,
    horizon: int = 1,
    stride: int = 1,
) -> Tuple[pd.DataFrame, pd.Series, List[str], bool]:
    """
    Build the cross-sectional feature matrix and labels.

    Parameters
    ----------
    horizon : int
        Forward-return window in trading days.
        1 = next day, 20 = 1 month, 63 = 3 months, 126 = 6 months.
    stride : int
        Keep only every stride-th row to reduce label autocorrelation.
        stride=1 keeps all rows; stride=horizon gives non-overlapping labels.
        For horizon 20-63, stride=5 to 10 is a good compromise.
    """
    print("[Features] Engineering features ...", flush=True)
    all_X  = []
    tickers = close.columns.tolist()

    for i, ticker in enumerate(tickers):
        c = close[ticker].dropna()
        if len(c) < min_history:
            continue

        r1   = np.log(c / c.shift(1))
        feat = {}

        for lag in [1, 2, 3, 5, 10, 20, 60, 120, 252, 756]:
            feat[f"ret{lag}"]   = np.log(c / c.shift(lag))
        for w in [5, 10, 20, 60, 120, 252]:
            feat[f"vol{w}"]     = r1.rolling(w).std()
        for m in [5, 10, 20, 60, 120, 252]:
            feat[f"mom{m}"]     = (c - c.shift(m)) / c.shift(m)

        feat["vol_ratio"]       = (r1.rolling(5).std() /
                                   (r1.rolling(20).std() + 1e-9))
        feat["ma50_ratio"]      = c / c.rolling(50).mean() - 1
        feat["ma200_ratio"]     = c / c.rolling(200).mean() - 1
        feat["ma50_200_cross"]  = (c.rolling(50).mean() /
                                   (c.rolling(200).mean() + 1e-9) - 1)
        feat["ret_accel"]       = (np.log(c / c.shift(5)) -
                                   np.log(c.shift(5) / c.shift(20)))

        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        feat["rsi14"]           = 100 - 100 / (1 + gain / (loss + 1e-9))

        feat["dist52h"]         = c / c.rolling(252).max() - 1
        feat["dist52l"]         = c / c.rolling(252).min() - 1
        feat["dist3yh"]         = c / c.rolling(756).max() - 1
        feat["dist3yl"]         = c / c.rolling(756).min() - 1

        for w in [5, 20, 60, 252]:
            feat[f"sharpe{w}"]  = feat[f"ret{w}"] / (feat[f"vol{w}"] + 1e-9)

        if vol_df is not None and ticker in vol_df.columns:
            v      = vol_df[ticker].reindex(c.index).fillna(0.0)
            v_ma5  = v.rolling(5,  min_periods=5).mean()  + 1e-9
            v_ma20 = v.rolling(20, min_periods=20).mean() + 1e-9
            v_ma60 = v.rolling(60, min_periods=60).mean() + 1e-9
            feat["rel_vol5"]   = v / v_ma20
            feat["rel_vol20"]  = v / v_ma60
            feat["vol_accel"]  = (v / v_ma5) / (v_ma5 / v_ma20 + 1e-9)
        else:
            feat["rel_vol5"]   = 1.0
            feat["rel_vol20"]  = 1.0
            feat["vol_accel"]  = 1.0

        feat["_sent"]  = (sent_df[ticker].reindex(c.index).fillna(0.0)
                          if sent_df is not None and ticker in sent_df.columns
                          else 0.0)
        feat["_fwd"]   = np.log(c.shift(-horizon) / c)

        df = pd.DataFrame(feat, index=c.index).dropna()
        all_X.append(df)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)} tickers ...", flush=True)

    X_full    = pd.concat(all_X).sort_index()
    fwd_raw   = X_full.pop("_fwd")
    sent_raw  = X_full.pop("_sent")
    feat_names = X_full.columns.tolist()

    print("  Computing cross-sectional ranks ...", flush=True)
    X_ranked = X_full.groupby(X_full.index).rank(pct=True)

    has_sent = sent_df is not None
    if has_sent:
        X_ranked["sent_rank"] = sent_raw.groupby(sent_raw.index).rank(pct=True)
        feat_names_out = feat_names + ["sent_rank"]
    else:
        feat_names_out = feat_names

    print(f"  Building labels (top {int(top_pct*100)}% vs "
          f"bottom {int(bottom_pct*100)}%) ...", flush=True)
    fwd_rank = fwd_raw.groupby(fwd_raw.index).rank(pct=True)
    y = pd.Series(np.nan, index=fwd_raw.index)
    y[fwd_rank >= (1.0 - top_pct)]  = 1
    y[fwd_rank <= bottom_pct]        = 0
    keep     = y.notna()
    X_final  = X_ranked[keep]
    y_final  = y[keep].astype(int)

    # Stride subsampling: keep only every stride-th unique date to reduce
    # label autocorrelation caused by overlapping forward-return windows.
    if stride > 1:
        all_dates = np.sort(X_final.index.unique())
        keep_dates = set(all_dates[::stride])
        mask = X_final.index.isin(keep_dates)
        X_final = X_final[mask]
        y_final = y_final[mask]
        print(f"  Stride={stride}: kept {len(all_dates[::stride])} / "
              f"{len(all_dates)} unique dates "
              f"({len(X_final):,} rows)", flush=True)

    print(f"  Feature matrix: {X_final.shape}  base rate: {y_final.mean():.4f}  "
          f"(kept {len(X_final):,} rows)", flush=True)
    return X_final, y_final, feat_names_out, has_sent
