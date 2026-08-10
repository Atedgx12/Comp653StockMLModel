"""
DuckDB Feature Store
====================
Stores pre-computed feature vectors and labels in a columnar database so
that walk-forward folds and mini-batch sampling are instant SQL queries
rather than numpy boolean mask operations on full in-memory arrays.

Schema
------
features (
    date     DATE      NOT NULL,
    ticker   VARCHAR   NOT NULL,
    horizon  SMALLINT  NOT NULL,   -- prediction horizon in trading days
    vec      DOUBLE[]  NOT NULL,   -- full feature vector (all 39 + sent)
    label    TINYINT,              -- 0 / 1  (NULL = middle 60% discarded)
    PRIMARY KEY (date, ticker, horizon)
)

Usage
-----
from ucn.data.store import FeatureStore

# Build once from make_features output
store = FeatureStore("D:/StockModel/features.duckdb")
store.upsert(X_df, y_df, horizon=90)

# Load a walk-forward fold in milliseconds
X_tr, y_tr, dates_tr = store.load_range(end="2020-01-01", horizon=90)
X_te, y_te, dates_te = store.load_range(start="2020-01-01", end="2022-01-01", horizon=90)

# Load everything for a given horizon
X, y, dates = store.load_all(horizon=90)
"""
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class FeatureStore:
    """
    Persistent DuckDB-backed store for cross-sectional feature vectors.
    Falls back gracefully if duckdb is not installed.
    """

    def __init__(self, path: str):
        if not HAS_DUCKDB:
            raise ImportError(
                "duckdb is required for FeatureStore. "
                "Install with: pip install duckdb")
        self.path = path
        self.con  = duckdb.connect(path)
        self._init_schema()

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS features (
                date    DATE     NOT NULL,
                ticker  VARCHAR  NOT NULL,
                horizon SMALLINT NOT NULL,
                vec     DOUBLE[] NOT NULL,
                label   TINYINT,
                PRIMARY KEY (date, ticker, horizon)
            )
        """)
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_dh "
            "ON features (date, horizon)")

    # ── Write ──────────────────────────────────────────────────────────────

    def upsert(
        self,
        X_df: pd.DataFrame,
        y_df: pd.Series,
        horizon: int,
        batch_size: int = 50_000,
    ) -> int:
        """
        Insert or replace features from a make_features() output.

        Parameters
        ----------
        X_df    : cross-sectionally ranked feature DataFrame
                  (index = date, columns = feature names)
        y_df    : binary labels (1 = outperformer, 0 = underperformer)
                  index matches X_df; missing dates/tickers are already dropped
        horizon : prediction horizon used when computing y_df
        batch_size: rows per INSERT batch

        Returns
        -------
        Number of rows inserted.
        """
        # Build a flat record dataframe
        df = X_df.copy()
        df["_label"]   = y_df.reindex(df.index)
        df["_horizon"] = horizon

        # Reconstruct ticker from the DataFrame structure.
        # After make_features, the index is the DATE and each row corresponds
        # to one (date, ticker) pair, but the ticker identity was lost during
        # pd.concat.  We carry the ticker as an extra column if present.
        if "ticker" in df.columns:
            df["_ticker"] = df["ticker"]
        else:
            # Fallback: use a synthetic ID based on row order within each date
            df["_ticker"] = (df.groupby(df.index).cumcount()
                               .astype(str).radd("t"))

        feat_cols = [c for c in X_df.columns if c != "ticker"]
        n_feat    = len(feat_cols)

        rows_inserted = 0
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start:start + batch_size]
            records = []
            for date, row in chunk.iterrows():
                vec   = row[feat_cols].values.tolist()
                label = None if pd.isna(row["_label"]) else int(row["_label"])
                records.append((
                    str(date)[:10],   # DATE string
                    str(row["_ticker"]),
                    horizon,
                    vec,
                    label,
                ))
            self.con.executemany(
                "INSERT OR REPLACE INTO features "
                "(date, ticker, horizon, vec, label) VALUES (?, ?, ?, ?, ?)",
                records)
            rows_inserted += len(records)

        self.con.commit()
        print(f"  FeatureStore: {rows_inserted:,} rows upserted "
              f"(horizon={horizon}, db={self.path})", flush=True)
        return rows_inserted

    # ── Read ───────────────────────────────────────────────────────────────

    def load_range(
        self,
        horizon: int,
        start: Optional[str] = None,
        end: Optional[str] = None,
        label_only: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load feature vectors for a date range.

        Parameters
        ----------
        horizon    : prediction horizon filter
        start      : inclusive lower bound (ISO date string), or None
        end        : exclusive upper bound (ISO date string), or None
        label_only : if True, only return rows where label IS NOT NULL

        Returns
        -------
        X      : float64 array (N, d)
        y      : int array    (N,)
        dates  : datetime64   (N,)
        """
        conditions = [f"horizon = {horizon}"]
        if label_only:
            conditions.append("label IS NOT NULL")
        if start:
            conditions.append(f"date >= '{start}'")
        if end:
            conditions.append(f"date < '{end}'")

        where = " AND ".join(conditions)
        sql   = f"SELECT vec, label, date FROM features WHERE {where} ORDER BY date"
        rows  = self.con.execute(sql).fetchall()

        if not rows:
            return (np.empty((0, 0)), np.empty(0, dtype=int),
                    np.empty(0, dtype="datetime64[D]"))

        X      = np.array([r[0] for r in rows], dtype=np.float64)
        y      = np.array([r[1] for r in rows], dtype=int)
        dates  = np.array([r[2] for r in rows], dtype="datetime64[D]")
        return X, y, dates

    def load_all(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.load_range(horizon=horizon)

    # ── Metadata ───────────────────────────────────────────────────────────

    def date_range(self, horizon: int):
        """Return (min_date, max_date) for a given horizon."""
        row = self.con.execute(
            "SELECT MIN(date), MAX(date) FROM features "
            f"WHERE horizon = {horizon} AND label IS NOT NULL"
        ).fetchone()
        return row[0], row[1]

    def row_count(self, horizon: int) -> int:
        return self.con.execute(
            "SELECT COUNT(*) FROM features "
            f"WHERE horizon = {horizon} AND label IS NOT NULL"
        ).fetchone()[0]

    def horizons(self):
        """Return list of all horizons stored."""
        return [r[0] for r in
                self.con.execute(
                    "SELECT DISTINCT horizon FROM features ORDER BY horizon"
                ).fetchall()]

    def summary(self):
        """Print a summary of what is stored."""
        print(f"\nFeatureStore: {self.path}")
        for h in self.horizons():
            lo, hi = self.date_range(h)
            n = self.row_count(h)
            print(f"  horizon={h:4d}  rows={n:>8,}  "
                  f"dates {lo} .. {hi}")

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Convenience: build from make_features output ──────────────────────────

def build_store(
    close: "pd.DataFrame",
    sent_df,
    vol_df,
    horizons: list,
    db_path: str,
    start: str = "2010-01-01",
) -> FeatureStore:
    """
    Build or update a FeatureStore for multiple horizons in one call.
    Re-uses the make_features pipeline for each horizon.

    Example
    -------
    store = build_store(close, sent_df, vol_df,
                        horizons=[1, 20, 63, 90, 126],
                        db_path="D:/StockModel/features.duckdb")
    store.summary()
    """
    from .features import make_features

    store = FeatureStore(db_path)
    for h in horizons:
        print(f"\nBuilding horizon={h} ...", flush=True)
        X_df, y_df, feat_names, _ = make_features(
            close, sent_df, vol_df, horizon=h, stride=1)
        # Attach ticker column from the DataFrame's construction order
        store.upsert(X_df, y_df, horizon=h)
    store.summary()
    return store
