"""
Fine-tuning demo for UnifiedCourseNetwork.

Usage examples
--------------
1. Fine-tune ONLY the MLP branch on the last 2 years of data:
       python fine_tune.py --branch mlp --years 2

2. Fine-tune the meta layer and sentiment branch (e.g. after adding new sentiment source):
       python fine_tune.py --branch meta sent --years 1

3. Full fine-tune with higher PGD steps:
       python fine_tune.py --pgd-steps 7 --epochs 500

Run from B:\\Rice\\Comp653(Summer2026)\\Module3\\homework\\stock_model\\
"""
import os, sys, argparse
import numpy as np

# Allow running from the stock_model directory
sys.path.insert(0, os.path.dirname(__file__))

from ucn import UnifiedCourseNetwork, UCNConfig
from ucn.training.metrics import accuracy, roc_auc

CHECKPOINT = os.path.join(os.path.dirname(__file__), "ucn_weights.npz")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune UnifiedCourseNetwork branches")
    p.add_argument("--branch", nargs="+", default=[],
                   choices=["lr", "nb", "mlp", "sent", "meta"],
                   help="Branches to fine-tune. All others are frozen. "
                        "Leave empty to fine-tune everything.")
    p.add_argument("--years",  type=float, default=2.0,
                   help="Number of most recent years to use for fine-tuning.")
    p.add_argument("--epochs", type=int,   default=200)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr",     type=float, default=5e-4,
                   help="Learning rate for fine-tuned branches.")
    p.add_argument("--pgd-steps", type=int, default=5)
    p.add_argument("--checkpoint", default=CHECKPOINT,
                   help="Path to .npz weights file to load.")
    p.add_argument("--save-as", default=None,
                   help="Path to save fine-tuned weights (defaults to overwrite checkpoint).")
    return p.parse_args()


def load_data_from_pipeline():
    """Load cached features using the monolithic pipeline helpers."""
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "pipeline_course",
        os.path.join(os.path.dirname(__file__), "pipeline_course.py"))
    pipe = types.ModuleType("pipeline_course")
    spec.loader.exec_module(pipe)

    tickers = pipe.get_sp500_tickers()
    close   = pipe.download_prices(tickers)
    sent_df = pipe.fetch_sentiment(tickers, close.index)
    vol_df  = pipe.download_volume(tickers)
    X_df, y_df, feat_names, has_sent = pipe.make_features(close, sent_df, vol_df)

    dates = X_df.index.values
    X_all = X_df.values.astype(np.float64)
    y_all = y_df.values.astype(int)
    return X_all, y_all, dates, feat_names, has_sent


def main():
    args = parse_args()

    print("Loading data from pipeline cache ...")
    X_all, y_all, dates, feat_names, has_sent = load_data_from_pipeline()

    # Subset to the most recent N years
    unique_dates = np.sort(np.unique(dates))
    cutoff_idx   = max(0, int(len(unique_dates) - args.years * 252))
    cutoff_date  = unique_dates[cutoff_idx]
    mask         = dates >= cutoff_date
    X, y         = X_all[mask], y_all[mask]
    print(f"Fine-tune subset: {mask.sum():,} rows  (last {args.years} years)")

    # Determine which branches to freeze
    all_branches  = {"lr", "nb", "mlp", "sent", "meta"}
    tune_branches = set(args.branch) if args.branch else all_branches
    freeze        = all_branches - tune_branches
    print(f"Tuning : {sorted(tune_branches)}")
    print(f"Frozen : {sorted(freeze)}")

    # Build config
    cfg = UCNConfig(
        use_sent       = has_sent,
        lr             = args.lr,
        epochs         = args.epochs,
        patience       = args.patience,
        pgd_steps      = args.pgd_steps,
        frozen_branches= tuple(freeze),
        branch_lrs     = {b: args.lr for b in tune_branches},
        verbose        = 20,
    )

    # Load pretrained checkpoint and fine-tune
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ucn = UnifiedCourseNetwork.from_checkpoint(args.checkpoint, cfg)
    else:
        print("No checkpoint found — training from scratch.")
        ucn = UnifiedCourseNetwork(cfg)

    # Z-score normalise using this subset's stats
    mu = X.mean(0); sd = X.std(0) + 1e-9
    X_norm = (X - mu) / sd

    ucn.fit(X_norm, y)

    # Quick evaluation on last 15%
    n_test  = max(int(len(X_norm) * 0.15), 50)
    X_test  = X_norm[len(X_norm)-n_test:]
    y_test  = y[len(y)-n_test:]
    pred    = ucn.predict(X_test)
    prob    = ucn.predict_proba(X_test)[:, 1]
    print(f"\nFine-tune test  acc={accuracy(y_test, pred):.4f}  "
          f"auc={roc_auc(y_test, prob):.4f}")

    save_path = args.save_as or args.checkpoint
    ucn.save_checkpoint(save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
