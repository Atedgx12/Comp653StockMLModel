"""
Decision layer and per-ticker calibration ledger.

After the model predicts a return band per horizon, this module turns the band
into actionable choices and scores them against the realized value.

For each row it produces:
  - a point price target (the band median),
  - a plausible price range (the calibrated band edges),
  - a buy, sell, or hold signal from the median predicted move,
  - call and put option strike suggestions from the current price.

Each choice is scored by its distance to the realized value and by whether the
actual fell inside the band. The outcomes accumulate in a combined per-ticker
ledger, which drives a per-ticker conformal widening so each ticker's future
band matches its own realized coverage. The ledger also accepts manual trader
choices, so a ticker gains its own history from the decisions made on it.
"""
import os
import numpy as np
import pandas as pd

TAUS = [0.05, 0.25, 0.50, 0.75, 0.95]

LEDGER_COLS = [
    "ticker", "date", "horizon", "p0", "target", "range_lo", "range_hi",
    "signal", "call_strike", "put_strike", "band_lo_ret", "band_hi_ret",
    "actual", "error", "abs_error", "pct_error", "in_band", "source",
]


def choose_batch(q, p0, hold_eps=0.005):
    """Turn return quantile bands into per row choices.

    q : (N, Q) forward return quantiles, ascending. p0 : (N,) current prices.
    Returns a DataFrame with the target, range, signal, and option strikes.
    """
    q = np.asarray(q, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    Q = q.shape[1]
    mid = Q // 2
    med = q[:, mid]
    lo = q[:, 0]; hi = q[:, -1]
    q25 = q[:, 1] if Q >= 3 else q[:, 0]
    q75 = q[:, -2] if Q >= 3 else q[:, -1]
    signal = np.where(med > hold_eps, "buy",
                      np.where(med < -hold_eps, "sell", "hold"))
    return pd.DataFrame({
        "p0": p0,
        "target": p0 * np.exp(med),
        "range_lo": p0 * np.exp(lo),
        "range_hi": p0 * np.exp(hi),
        "signal": signal,
        "call_strike": p0 * np.exp(q75),
        "put_strike": p0 * np.exp(q25),
        "band_lo_ret": lo,
        "band_hi_ret": hi,
    })


def score_batch(choices, actual_price):
    """Score the point target and band membership against the realized price."""
    out = choices.copy()
    ap = np.asarray(actual_price, dtype=float)
    tgt = choices["target"].values
    out["actual"] = ap
    out["error"] = ap - tgt
    out["abs_error"] = np.abs(ap - tgt)
    out["pct_error"] = (ap - tgt) / np.maximum(tgt, 1e-9)
    out["in_band"] = ((ap >= choices["range_lo"].values)
                      & (ap <= choices["range_hi"].values))
    return out


class TickerLedger:
    """Combined per-ticker store of predictions, choices, actuals, and errors."""

    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.df = pd.read_parquet(path)
        else:
            self.df = pd.DataFrame(columns=LEDGER_COLS)

    def append(self, rows):
        add = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        for c in LEDGER_COLS:
            if c not in add.columns:
                add[c] = np.nan
        add = add[LEDGER_COLS]
        self.df = add.copy() if self.df.empty else pd.concat(
            [self.df, add], ignore_index=True)
        return self

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.df.to_parquet(self.path, index=False)
        return self

    def coverage_by_ticker(self):
        """Realized band coverage per ticker."""
        cov = self.df.copy()
        cov["in_band"] = pd.to_numeric(cov["in_band"], errors="coerce")
        return cov.groupby("ticker")["in_band"].mean()

    def per_ticker_delta(self, alpha=0.10, min_rows=20):
        """Per-ticker conformal widening from the accumulated band history.

        The conformity score in return space is max(lo - y, y - hi). The
        per-ticker widening is the empirical (1 - alpha) quantile of that score,
        so applying it to a ticker's future band restores its nominal coverage.
        """
        deltas = {}
        for tk, g in self.df.groupby("ticker"):
            p0 = g["p0"].values.astype(float)
            actual = g["actual"].values.astype(float)
            ok = (p0 > 0) & np.isfinite(actual) & (actual > 0)
            if ok.sum() < min_rows:
                deltas[tk] = 0.0
                continue
            y = np.log(actual[ok] / p0[ok])
            lo = g["band_lo_ret"].values.astype(float)[ok]
            hi = g["band_hi_ret"].values.astype(float)[ok]
            s = np.maximum(lo - y, y - hi)
            n = len(s)
            k = int(np.ceil((1 - alpha) * (n + 1)))
            deltas[tk] = float(max(np.sort(s)[min(k, n) - 1], 0.0))
        return deltas


def log_manual_choice(ledger, ticker, date, horizon, p0, target,
                      range_lo, range_hi, signal="hold", actual=np.nan):
    """Append a trader's manual choice on one ticker to the ledger."""
    row = {
        "ticker": ticker, "date": pd.to_datetime(date), "horizon": horizon,
        "p0": float(p0), "target": float(target),
        "range_lo": float(range_lo), "range_hi": float(range_hi),
        "signal": signal, "call_strike": np.nan, "put_strike": np.nan,
        "band_lo_ret": float(np.log(range_lo / p0)),
        "band_hi_ret": float(np.log(range_hi / p0)),
        "actual": float(actual) if np.isfinite(actual) else np.nan,
        "source": "manual",
    }
    if np.isfinite(row["actual"]):
        row["error"] = row["actual"] - row["target"]
        row["abs_error"] = abs(row["error"])
        row["pct_error"] = row["error"] / max(row["target"], 1e-9)
        row["in_band"] = range_lo <= row["actual"] <= range_hi
    ledger.append([row])
    return ledger
