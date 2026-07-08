"""
Branch-level fine-tuning tests for UnifiedCourseNetwork.

Each test:
  1. Trains a small UCN on synthetic data to get a baseline
  2. Saves a checkpoint
  3. Reloads it and fine-tunes with only one branch unfrozen
  4. Asserts that ONLY that branch's weights changed
  5. Reports accuracy before and after

Run:
    python -m ucn.testing.branch_tests
  or:
    python branch_tests.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import copy, tempfile
import numpy as np
from ucn import UnifiedCourseNetwork, UCNConfig
from ucn.training.metrics import accuracy, roc_auc

RNG  = np.random.default_rng(42)
N    = 2000
D    = 20          # 19 price features + 1 sentiment
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def make_dataset(n=N, d=D, seed=0):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d))
    # Weak learnable signal in first 5 features
    y   = (X[:, :5].mean(axis=1) + rng.normal(0, 0.8, n) > 0).astype(int)
    return X.astype(np.float64), y


def base_cfg(**kwargs):
    defaults = dict(
        hidden_sizes=(32, 16),
        use_sent=True,
        epochs=30,
        patience=30,
        verbose=0,
        pgd_steps=1,
        noise_frac=0.0,
        val_frac=0.15,
    )
    defaults.update(kwargs)   # kwargs override defaults, no duplicates
    return UCNConfig(**defaults)


def weights_changed(before: dict, after: dict) -> set:
    """Return set of param keys whose values changed between two weight dicts."""
    return {k for k in before if not np.allclose(before[k], after[k], atol=1e-10)}


def weights_unchanged(before: dict, after: dict) -> set:
    return {k for k in before if np.allclose(before[k], after[k], atol=1e-10)}


def run_test(branch_name: str, X_train, y_train, X_test, y_test,
             pretrained_params: dict, tmpdir: str) -> bool:
    all_branches = {"lr", "nb", "mlp", "sent", "meta"}
    freeze       = all_branches - {branch_name}

    cfg = base_cfg(
        epochs=20,
        frozen_branches=tuple(freeze),
        branch_lrs={branch_name: 1e-3},
    )

    ckpt = os.path.join(tmpdir, "pretrained.npz")
    np.savez_compressed(ckpt, **pretrained_params)

    ucn = UnifiedCourseNetwork.from_checkpoint(ckpt, cfg)
    acc_before = accuracy(y_test, ucn.predict(X_test))

    w_before = {k: v.copy() for k, v in ucn.params.items()}
    ucn.fit(X_train, y_train)
    w_after  = {k: v.copy() for k, v in ucn.params.items()}

    acc_after = accuracy(y_test, ucn.predict(X_test))

    changed   = weights_changed(w_before, w_after)
    unchanged = weights_unchanged(w_before, w_after)

    # Classify each changed param by branch
    changed_branches = set()
    for k in changed:
        for b in all_branches:
            if cfg.param_belongs_to(k, b):
                changed_branches.add(b)

    frozen_changed = changed_branches - {branch_name}
    ok = len(frozen_changed) == 0

    status = PASS if ok else FAIL
    print(f"  {status}  branch='{branch_name:6s}'  "
          f"acc {acc_before:.4f} -> {acc_after:.4f}  "
          f"changed={sorted(changed_branches)}  "
          f"frozen_violated={sorted(frozen_changed)}")
    if not ok:
        print(f"         Frozen params that changed: "
              f"{[k for k in changed if any(cfg.param_belongs_to(k, b) for b in frozen_changed)]}")
    return ok


def test_checkpoint_roundtrip(ucn: UnifiedCourseNetwork, tmpdir: str) -> bool:
    """Test that save -> load -> predict gives identical results."""
    path = os.path.join(tmpdir, "roundtrip.npz")
    X, y = make_dataset(200)
    pred_before = ucn.predict(X)
    ucn.save_checkpoint(path)

    ucn2 = UnifiedCourseNetwork(base_cfg())
    ucn2.load_checkpoint(path)
    pred_after = ucn2.predict(X)

    ok = np.array_equal(pred_before, pred_after)
    print(f"  {PASS if ok else FAIL}  checkpoint roundtrip  "
          f"predictions_identical={ok}")
    return ok


def test_branch_summary(ucn: UnifiedCourseNetwork) -> bool:
    """Test branch_summary runs without error."""
    try:
        ucn.branch_summary()
        print(f"  {PASS}  branch_summary()")
        return True
    except Exception as e:
        print(f"  {FAIL}  branch_summary() raised: {e}")
        return False


def main():
    print("=" * 60)
    print("UCN Branch Fine-Tuning Test Suite")
    print("=" * 60)

    X, y = make_dataset()
    n_split  = int(len(X) * 0.8)
    X_train, X_test = X[:n_split], X[n_split:]
    y_train, y_test = y[:n_split], y[n_split:]

    # Train a baseline model
    print("\n[1] Training baseline model ...")
    ucn_base = UnifiedCourseNetwork(base_cfg())
    ucn_base.fit(X_train, y_train)
    pretrained = {k: v.copy() for k, v in ucn_base.params.items()}
    base_acc = accuracy(y_test, ucn_base.predict(X_test))
    base_auc = roc_auc(y_test, ucn_base.predict_proba(X_test)[:, 1])
    print(f"  Baseline  acc={base_acc:.4f}  auc={base_auc:.4f}")

    # Per-branch fine-tuning tests
    print("\n[2] Per-branch isolation tests ...")
    all_pass = True
    with tempfile.TemporaryDirectory() as tmpdir:
        for branch in ["lr", "nb", "mlp", "sent", "meta"]:
            ok = run_test(branch, X_train, y_train, X_test, y_test,
                          pretrained, tmpdir)
            all_pass = all_pass and ok

        # Checkpoint roundtrip
        print("\n[3] Checkpoint roundtrip ...")
        ok = test_checkpoint_roundtrip(ucn_base, tmpdir)
        all_pass = all_pass and ok

    # Branch summary
    print("\n[4] Branch summary ...")
    ok = test_branch_summary(ucn_base)
    all_pass = all_pass and ok

    print("\n" + "=" * 60)
    if all_pass:
        print(f"{PASS}  All tests passed.")
    else:
        print(f"{FAIL}  Some tests failed. See above.")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
