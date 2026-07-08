"""
Unified volatility term structure: intraday and daily in one run.

This runs the intraday horizons (1, 5, 15, 30, 60, 240 minutes) and the daily
horizons (1, 5, 10, 30, 90, 180 days) in a single execution and stitches them
into one term structure spanning from one minute to one hundred eighty days.

Intraday and daily come from different data sources with different histories,
so they cannot share the same sample rows.  Each scale is modeled on its own
data, and the two are joined into one reported curve on a shared axis measured
in trading days.

Usage:
    set UCN_GPU=1& python term_structure_all.py --epochs 800
"""
import os
import sys
import argparse
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import multiscale_run
import intraday_run


def ascii_bars(labels, values, title, width=52, fmt="{:.4f}"):
    print("\n" + title, flush=True)
    vmax = max(values) if max(values) > 0 else 1.0
    vmin = min(min(values), 0.0)
    span = (vmax - vmin) or 1.0
    for lab, v in zip(labels, values):
        n = int(round(width * (v - vmin) / span))
        print(f"  {str(lab):>6} | {'#' * n:<{width}} {fmt.format(v)}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--stride", type=int, default=9)
    p.add_argument("--stride-min", type=int, default=5)
    p.add_argument("--n-tickers", type=int, default=60)
    p.add_argument("--smooth-lambda", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--min-dollar-vol", type=float, default=50_000_000.0)
    p.add_argument("--cv-frac", type=float, default=0.80)
    return p.parse_args()


def main():
    a = parse_args()

    intra_args = SimpleNamespace(
        n_tickers=a.n_tickers, stride_min=a.stride_min,
        smooth_lambda=a.smooth_lambda, epochs=a.epochs, cv_frac=a.cv_frac)
    daily_args = SimpleNamespace(
        start=a.start, stride=a.stride, ref_horizon=30,
        smooth_lambda=a.smooth_lambda, epochs=a.epochs,
        min_dollar_vol=a.min_dollar_vol, cv_frac=a.cv_frac)

    print("\n########## INTRADAY SCALE ##########", flush=True)
    intra = intraday_run.run(intra_args)

    print("\n########## DAILY SCALE ##########", flush=True)
    daily = multiscale_run.run(daily_args)

    combined = sorted(intra + daily, key=lambda r: r["days"])

    print("\n" + "=" * 65)
    print(" UNIFIED VOLATILITY TERM STRUCTURE: 1 minute to 180 days")
    print("=" * 65, flush=True)
    labels = [r["label"] for r in combined]
    ascii_bars(labels, [r["mean_vol"] for r in combined],
               "Mean realized volatility by horizon (per bar):", fmt="{:.6f}")
    ascii_bars(labels, [r["auc"] for r in combined],
               "Predictability by horizon (held-out AUC):")
    print("\nHorizon        days       mean_vol     AUC")
    print("-" * 46)
    for r in combined:
        print(f"  {r['label']:>6}  {r['days']:>10.4f}  {r['mean_vol']:>11.6f}  "
              f"{r['auc']:>7.4f}")


if __name__ == "__main__":
    main()
