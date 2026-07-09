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
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="60d")
    p.add_argument("--n-tickers", type=int, default=60)
    p.add_argument("--smooth-lambda", type=float, default=0.3)
    p.add_argument("--additivity-lambda", type=float, default=0.0,
                   help="Variance additivity coupling weight, passed to both "
                        "scales. 0 disables it.")
    p.add_argument("--label-pct", type=float, default=0.5,
                   help="Extreme label split fraction passed to both scales. "
                        "Below 0.5 drops the ambiguous middle. 0.5 is median.")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--min-dollar-vol", type=float, default=50_000_000.0)
    p.add_argument("--cv-frac", type=float, default=0.80)
    p.add_argument("--parallel-scales", action="store_true",
                   help="Run the intraday and daily scales as concurrent "
                        "processes, one per compute engine.")
    p.add_argument("--daily-device", choices=["gpu", "cpu"], default="cpu",
                   help="Device for the daily scale when running in parallel. "
                        "The intraday scale always takes the GPU.")
    p.add_argument("--warm-start", default=None,
                   help="Intraday checkpoint to warm start the daily trunk.")
    p.add_argument("--warm-start-force", action="store_true")
    p.add_argument("--stack-intraday", action="store_true",
                   help="Append intraday realized volatility as a daily feature.")
    p.add_argument("--warm-restarts", action="store_true",
                   help="Use cosine warm restarts with Adam moment reset.")
    p.add_argument("--restart-period", type=int, default=120)
    p.add_argument("--context-top-k", type=int, default=25)
    return p.parse_args()


def _run_parallel(a):
    """Run the two scales as separate processes so both engines are busy.

    The intraday scale takes the GPU and the daily scale takes the chosen
    device. Each process writes its results to JSON, which the parent merges.
    Note that the GPU is far faster here, so routing the daily scale to the CPU
    only helps when its slice is small enough not to become the long pole.
    """
    import subprocess, json
    py = sys.executable
    intra_json = os.path.join(ROOT, "_intra_results.json")
    daily_json = os.path.join(ROOT, "_daily_results.json")
    intra_cmd = [py, "-u", os.path.join(ROOT, "intraday_run.py"),
                 "--n-tickers", str(a.n_tickers), "--stride-min", str(a.stride_min),
                 "--interval", a.interval, "--period", a.period,
                 "--smooth-lambda", str(a.smooth_lambda), "--epochs", str(a.epochs),
                 "--additivity-lambda", str(a.additivity_lambda),
                 "--label-pct", str(a.label_pct),
                 "--cv-frac", str(a.cv_frac), "--emit-json", intra_json]
    daily_cmd = [py, "-u", os.path.join(ROOT, "multiscale_run.py"),
                 "--start", a.start, "--stride", str(a.stride),
                 "--smooth-lambda", str(a.smooth_lambda), "--epochs", str(a.epochs),
                 "--additivity-lambda", str(a.additivity_lambda),
                 "--label-pct", str(a.label_pct),
                 "--min-dollar-vol", str(a.min_dollar_vol),
                 "--cv-frac", str(a.cv_frac), "--emit-json", daily_json]
    if a.warm_start:
        daily_cmd += ["--warm-start", a.warm_start]
    if a.warm_start_force:
        daily_cmd += ["--warm-start-force"]
    if a.stack_intraday:
        daily_cmd += ["--stack-intraday"]
    env_gpu = dict(os.environ); env_gpu["UCN_GPU"] = "1"
    env_cpu = dict(os.environ); env_cpu.pop("UCN_GPU", None)
    daily_env = env_gpu if a.daily_device == "gpu" else env_cpu
    print(f"[Parallel] intraday on GPU, daily on {a.daily_device.upper()}",
          flush=True)
    p1 = subprocess.Popen(intra_cmd, env=env_gpu)
    p2 = subprocess.Popen(daily_cmd, env=daily_env)
    p1.wait(); p2.wait()
    with open(intra_json) as f:
        intra = json.load(f)
    with open(daily_json) as f:
        daily = json.load(f)
    return intra, daily


def main():
    a = parse_args()

    if a.parallel_scales:
        intra, daily = _run_parallel(a)
    else:
        intra_args = SimpleNamespace(
            n_tickers=a.n_tickers, stride_min=a.stride_min,
            interval=a.interval, period=a.period,
            smooth_lambda=a.smooth_lambda, epochs=a.epochs, cv_frac=a.cv_frac,
            warm_restarts=a.warm_restarts, restart_period=a.restart_period,
            additivity_lambda=a.additivity_lambda,
            label_pct=a.label_pct,
            context_top_k=a.context_top_k)
        daily_args = SimpleNamespace(
            start=a.start, stride=a.stride, ref_horizon=30,
            smooth_lambda=a.smooth_lambda, epochs=a.epochs,
            min_dollar_vol=a.min_dollar_vol, cv_frac=a.cv_frac,
            warm_start=a.warm_start, warm_start_force=a.warm_start_force,
            stack_intraday=a.stack_intraday,
            warm_restarts=a.warm_restarts, restart_period=a.restart_period,
            additivity_lambda=a.additivity_lambda,
            label_pct=a.label_pct,
            context_top_k=a.context_top_k)

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
