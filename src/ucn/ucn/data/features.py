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
    use_nomadic: bool = False,
    use_hierarchy: bool = False,
    sector_map: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], bool]:
    """
    Build the cross-sectional feature matrix and labels.

    Parameters
    ----------
    horizon : int
        Forward-return window in trading days.
    stride : int
        Keep only every stride-th row to reduce label autocorrelation.
    use_nomadic : bool
        Add ~20 extended indicators from NomadicStockBot methodology:
        CCI, Williams %R, OBV, CMF, MFI, ADX, Ichimoku, VWAP deviation,
        Donchian breakout, BB squeeze release, RSI/MACD temporal derivatives.
        For horizon 20-63, stride=5 to 10 is a good compromise.
    """
    print("[Features] Engineering features ...", flush=True)
    all_X  = []
    tickers = close.columns.tolist()

    # Hierarchical market context: build the broad market index and one index
    # per sector once, before the ticker loop, so every ticker is conditioned
    # on the same market and sector levels.
    market_index = None
    sector_indices = None
    if use_hierarchy:
        from .market_context import build_equal_weight_index, build_sector_indices
        market_index = build_equal_weight_index(close)
        sector_indices = build_sector_indices(close, sector_map)
        n_sectors = len(sector_indices)
        print(f"  Hierarchical context: 1 market index + {n_sectors} "
              f"sector index(es)", flush=True)

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
        feat["_ticker"] = ticker   # preserve identity for LSTM sequence building

        # Optional: NomadicStockBot extended indicators
        if use_nomadic:
            try:
                from .nomadic_features import add_nomadic_features
                # Use OHLCV data: approximate high/low from close if not available
                if vol_df is not None and ticker in vol_df.columns:
                    vol_s = vol_df[ticker].reindex(c.index).fillna(0.0)
                else:
                    vol_s = pd.Series(1.0, index=c.index)
                # Daily OHLC approximation from close only (conservative)
                high_s = c.rolling(1).max()
                low_s  = c.rolling(1).min()
                feat   = add_nomadic_features(c, feat, high_s, low_s, vol_s)
            except Exception:
                pass   # silently skip if nomadic features fail for any ticker

        # Optional: hierarchical market context (stock vs sector vs market
        # relative strength plus macro regime descriptors).
        if use_hierarchy and market_index is not None:
            from .market_context import add_hierarchical_context
            sector_label = (sector_map.get(ticker, "MARKET")
                            if sector_map else "MARKET")
            sector_idx = sector_indices.get(sector_label,
                                            sector_indices.get("MARKET"))
            feat = add_hierarchical_context(ticker, c, feat,
                                            market_index, sector_idx)

        df = pd.DataFrame(feat, index=c.index).dropna()
        all_X.append(df)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)} tickers ...", flush=True)

    X_full    = pd.concat(all_X).sort_index()
    fwd_raw   = X_full.pop("_fwd")
    sent_raw  = X_full.pop("_sent")
    ticker_col = X_full.pop("_ticker")   # preserve for LSTM — not a feature

    print("  Computing cross-sectional ranks ...", flush=True)
    # Macro regime columns are identical across the cross section on any date,
    # so cross sectional percentile ranking would collapse them to a constant.
    # I separate them out, z score them across time to expose regime shifts,
    # and rank only the columns that actually vary across the cross section.
    macro_cols = [c for c in X_full.columns if c.startswith("macro_")]
    rank_cols  = [c for c in X_full.columns if not c.startswith("macro_")]

    X_ranked = X_full[rank_cols].groupby(X_full.index).rank(pct=True)

    if macro_cols:
        macro_raw = X_full[macro_cols]
        macro_z = (macro_raw - macro_raw.mean()) / (macro_raw.std() + 1e-9)
        # Map z scores into the same zero to one band as the ranked features
        # so every input to the network shares a comparable scale.
        macro_scaled = 0.5 + 0.15 * macro_z.clip(-3, 3)
        for col in macro_cols:
            X_ranked[col] = macro_scaled[col]
        print(f"  Macro regime features (time z scored): {len(macro_cols)}",
              flush=True)

    # Feature name order must match the column order of X_ranked, which places
    # the ranked columns first and the macro columns after them.
    feat_names = rank_cols + macro_cols

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
    X_final  = X_ranked[keep].copy()
    y_final  = y[keep].astype(int)
    # Attach ticker column for LSTM sequence building (not used in training directly)
    X_final["_ticker"] = ticker_col[keep]

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
