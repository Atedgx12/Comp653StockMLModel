"""
Hierarchical market context features.

The idea is to condition each stock on the layers above it in the market
hierarchy.  A single stock sits inside a sector, the sector sits inside a
broad market, and the broad market sits inside a volatility regime.  A move
of plus two percent means something very different in a calm bull market
than in the middle of a crash, so I give the model the context of the
layers above each name.

Three layers of context are built here.

Stock relative to sector.  I subtract the equal weighted sector return from
the stock return at several horizons.  This isolates idiosyncratic strength
from sector rotation.  These features vary across the cross section on any
given date, so they survive the later percentile ranking and behave like
ordinary features.

Stock relative to market.  I subtract the equal weighted market return from
the stock return.  This captures how a name trades against the whole S&P
universe.

Macro regime.  Broad market trend and volatility at several horizons.  These
values are identical for every ticker on a given date, so cross sectional
ranking would flatten them to a constant.  To keep them useful I mark them
with a macro prefix and z score them across time instead of across the cross
section.  These are the features that let the model notice it is in a
different regime than the one it trained on, which is the core problem on
long horizons where the training window and the test window sit in different
market environments.

The market and sector indices are synthetic equal weighted averages built
straight from the ticker universe, so this module needs no extra downloads.
A sector map may be supplied to group tickers.  When none is given every
ticker falls into a single MARKET group, in which case the sector layer and
the market layer coincide and only the market relative features are informative.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# Horizons used for every context comparison, in trading days.
CONTEXT_HORIZONS: List[int] = [5, 20, 60, 120, 252]

# Horizons used for the macro volatility and trend descriptors.
MACRO_HORIZONS: List[int] = [20, 60, 120, 252]


def build_equal_weight_index(close: pd.DataFrame,
                             members: Optional[List[str]] = None) -> pd.Series:
    """
    Build an equal weighted price index from a set of member tickers.

    I average the daily log returns across members and cumulate them, which
    gives a smooth index level that is robust to individual tickers starting
    or ending at different dates.  Averaging returns rather than prices avoids
    letting a single high priced name dominate the level.
    """
    cols = members if members is not None else close.columns.tolist()
    cols = [c for c in cols if c in close.columns]
    if not cols:
        # No members means no index, return a flat series so callers stay safe.
        return pd.Series(1.0, index=close.index)

    sub = close[cols]
    daily_ret = np.log(sub / sub.shift(1))
    mean_ret = daily_ret.mean(axis=1)
    level = mean_ret.cumsum().apply(np.exp)
    return level


def build_sector_indices(close: pd.DataFrame,
                         sector_map: Optional[Dict[str, str]]) -> Dict[str, pd.Series]:
    """
    Build one equal weighted index per sector.

    A sector map assigns each ticker to a sector label.  When no map is given
    every ticker is placed in a single MARKET sector so the pipeline still runs
    and the sector layer simply mirrors the market layer.
    """
    if not sector_map:
        return {"MARKET": build_equal_weight_index(close)}

    groups: Dict[str, List[str]] = {}
    for ticker in close.columns:
        sector = sector_map.get(ticker, "MARKET")
        groups.setdefault(sector, []).append(ticker)

    return {sector: build_equal_weight_index(close, members)
            for sector, members in groups.items()}


def _horizon_return(level: pd.Series, h: int) -> pd.Series:
    """Log return of an index level over a horizon of h trading days."""
    return np.log(level / level.shift(h))


def add_hierarchical_context(
    ticker: str,
    close_ticker: pd.Series,
    feat: dict,
    market_index: pd.Series,
    sector_index: pd.Series,
) -> dict:
    """
    Append hierarchical context features for one ticker to its feature dict.

    Parameters
    ----------
    ticker : str
        Ticker symbol, kept for clarity and possible per name behaviour.
    close_ticker : pd.Series
        Close price series for this ticker, already dropna aligned.
    feat : dict
        The feature dictionary being assembled inside make_features.
    market_index : pd.Series
        Equal weighted broad market index level.
    sector_index : pd.Series
        Equal weighted index level for this ticker's sector.

    Returns
    -------
    dict
        The same feature dictionary with context columns added.
    """
    idx = close_ticker.index
    stock_level = close_ticker

    market = market_index.reindex(idx)
    sector = sector_index.reindex(idx)

    # Stock relative to sector and stock relative to market at each horizon.
    for h in CONTEXT_HORIZONS:
        stock_r  = np.log(stock_level / stock_level.shift(h))
        sector_r = _horizon_return(sector, h)
        market_r = _horizon_return(market, h)

        feat[f"rs_sector{h}"] = stock_r - sector_r
        feat[f"rs_market{h}"] = stock_r - market_r
        feat[f"sector_vs_market{h}"] = sector_r - market_r

    # Macro regime descriptors.  These are identical across the cross section
    # on any date, so they carry a macro prefix and skip cross sectional
    # ranking downstream.  Broad market trend and realized volatility describe
    # what kind of environment the prediction is being made in.
    market_daily = np.log(market / market.shift(1))
    for h in MACRO_HORIZONS:
        feat[f"macro_trend{h}"] = _horizon_return(market, h)
        feat[f"macro_vol{h}"]   = market_daily.rolling(h).std()

    return feat


def macro_feature_names(feat_names: List[str]) -> List[str]:
    """Return the subset of feature names that describe the macro regime."""
    return [f for f in feat_names if f.startswith("macro_")]


def load_or_build_sector_map(
    tickers: List[str],
    cache_path: str = "D:/StockModel/sector_map.json",
) -> Optional[Dict[str, str]]:
    """
    Load a ticker to sector map, building it from yfinance on first use.

    The map is cached to a JSON file so the network is queried only once.
    Each ticker is assigned its GICS sector.  When a sector cannot be found
    the ticker is placed in a MARKET catch all group.  When yfinance is not
    available the function returns None so the caller falls back to a single
    market group and the pipeline still runs.
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if all(t in cached for t in tickers):
                return {t: cached.get(t, "MARKET") for t in tickers}
        except Exception:
            pass  # fall through and rebuild

    try:
        import yfinance as yf
    except Exception:
        return None

    sector_map: Dict[str, str] = {}
    for i, t in enumerate(tickers):
        sector = "MARKET"
        try:
            info = yf.Ticker(t).info
            sector = info.get("sector") or "MARKET"
        except Exception:
            sector = "MARKET"
        sector_map[t] = sector
        if (i + 1) % 25 == 0:
            print(f"  Sector lookup {i+1}/{len(tickers)} ...", flush=True)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(sector_map, f, indent=2)
    except Exception:
        pass

    return sector_map


def build_correlation_clusters(
    close: pd.DataFrame,
    n_clusters: Optional[int] = None,
    insample_end: Optional[str] = None,
    k_range: tuple = (8, 20),
    min_obs: int = 250,
    market_neutral: bool = True,
) -> Dict[str, str]:
    """
    Group tickers into data driven sectors by clustering return co movement.

    Rather than trust an external sector label, I let the data decide which
    names belong together.  Two stocks belong in the same group when their
    returns move together, which is the statistical meaning of a sector.  I
    build the pairwise correlation of daily returns, turn it into a distance
    where highly correlated names sit close together, and cluster that distance.

    Raw return correlation is dominated by the broad market factor.  Almost
    every stock correlates strongly with the market, so clustering raw returns
    finds one giant blob and splits it into a few coarse pieces that are useless
    for a sector relative feature.  When market_neutral is set I first remove
    the common market factor by subtracting the cross sectional mean return on
    each date, then cluster the residuals.  What remains after the whole market
    tide is stripped out is the idiosyncratic co movement that actually defines
    a peer group, so the clustering finds real sector structure.

    To avoid letting future co movement leak into the features, correlations
    are estimated only from data up to insample_end.  The resulting cluster
    labels are then held fixed and applied to the whole sample.

    When n_clusters is not given, the number of groups is chosen by sweeping a
    range of candidate counts and keeping the one with the best silhouette
    score, so the data decides how many sectors exist.

    Returns a map from ticker to a cluster label such as CL03, in the same form
    the rest of the pipeline expects from a sector map.  Tickers with too little
    history to cluster fall into a MARKET catch all group.
    """
    sub = close.loc[:insample_end] if insample_end is not None else close
    returns = np.log(sub / sub.shift(1))

    valid = [c for c in returns.columns if returns[c].notna().sum() >= min_obs]
    dropped = [c for c in close.columns if c not in valid]
    if len(valid) < k_range[0] + 1:
        return {t: "MARKET" for t in close.columns}

    ret_valid = returns[valid]
    if market_neutral:
        # Subtract the equal weighted market return on each date so the shared
        # market factor no longer dominates the correlations.  What is left is
        # each stock's idiosyncratic move relative to the whole universe.
        market_ret = ret_valid.mean(axis=1)
        ret_valid = ret_valid.sub(market_ret, axis=0)

    corr = ret_valid.corr().fillna(0.0)
    # Distance in zero to two: identical movers sit at zero, opposite movers at two.
    dist = (1.0 - corr).clip(lower=0.0).values

    labels = _cluster_distance_matrix(dist, n_clusters, k_range)

    cluster_map = {t: f"CL{int(lab):02d}" for t, lab in zip(valid, labels)}
    for t in dropped:
        cluster_map[t] = "MARKET"

    n_found = len(set(labels))
    kind = "market neutral" if market_neutral else "raw"
    print(f"  Learned clusters ({kind}): {n_found} groups from return co "
          f"movement ({len(valid)} tickers clustered, {len(dropped)} in MARKET)",
          flush=True)
    return cluster_map


def _cluster_distance_matrix(dist, n_clusters, k_range):
    """
    Cluster a precomputed distance matrix, selecting k by silhouette when
    n_clusters is not supplied.  Uses scikit learn when available and falls
    back to a compact numpy KMeans on the distance rows otherwise.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        if n_clusters is not None:
            model = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage="average")
            return model.fit_predict(dist)

        best_k, best_score, best_labels = k_range[0], -1.0, None
        upper = min(k_range[1], dist.shape[0] - 1)
        for k in range(k_range[0], upper + 1):
            model = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average")
            labels = model.fit_predict(dist)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(dist, labels, metric="precomputed")
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels
        print(f"  Silhouette selected k={best_k} (score={best_score:.4f})",
              flush=True)
        return best_labels

    except Exception:
        k = n_clusters or (k_range[0] + k_range[1]) // 2
        return _numpy_kmeans(dist, k)


def _numpy_kmeans(X, k, iters=50, seed=0):
    """A compact KMeans used only when scikit learn is unavailable."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(X.shape[0], dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels
