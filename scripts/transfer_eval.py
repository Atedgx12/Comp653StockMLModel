"""Sample transfer learning evaluation script.

Demonstrates the protocol described in docs/proposal_revised.md: pretrain on
one panel, evaluate zero shot on a second panel, fine tune on part of the
second panel, evaluate again on the held out tail.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stockml.data.preprocessing import drop_zero_volume, winsorize_returns
from stockml.evaluation.transfer import transfer_evaluate
from stockml.features.pipeline import build_features
from stockml.labels.returns import multi_horizon_return_labels
from stockml.models import build_model
from stockml.utils.io import project_root, read_yaml
from stockml.utils.logging import configure_logging
from stockml.utils.seeding import set_global_seed


def _synth_panel(seed: int, drift: float, vol: float, tickers: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2024-12-31")
    panels = []
    for ticker in tickers:
        rets = rng.normal(drift, vol, len(dates))
        close = 50.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.001, len(dates))),
                "high": close * (1 + np.abs(rng.normal(0, 0.005, len(dates)))),
                "low": close * (1 - np.abs(rng.normal(0, 0.005, len(dates)))),
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, len(dates)),
                "ticker": ticker,
            },
            index=dates,
        )
        df.index.name = "date"
        panels.append(df)
    return pd.concat(panels)


def main() -> None:
    configure_logging("INFO")
    set_global_seed(42)

    cfg_root: Path = project_root() / "configs"
    feature_cfg = read_yaml(cfg_root / "features" / "standard_technicals.yaml")
    model_cfg = read_yaml(cfg_root / "model" / "linear.yaml")

    equities = _synth_panel(seed=0, drift=0.0005, vol=0.012, tickers=["EQ_A", "EQ_B", "EQ_C"])
    crypto = _synth_panel(seed=1, drift=0.0008, vol=0.030, tickers=["CR_A", "CR_B"])

    def prepare(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        panel = drop_zero_volume(panel)
        f = build_features(panel, feature_cfg, market=None)
        f = multi_horizon_return_labels(f, horizons=[5])
        f = winsorize_returns(f, return_col="log_return_1")
        f = f.dropna(subset=["y_logret_h5"])
        drop = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ticker",
            "y_logret_h5",
        }
        cols = [c for c in f.columns if c not in drop and not c.startswith("log_return_")]
        return f, cols

    eq_feats, eq_cols = prepare(equities)
    cr_feats, cr_cols = prepare(crypto)

    common = [c for c in eq_cols if c in cr_cols]

    model = build_model(model_cfg, task="regression")
    model.fit(
        eq_feats[common].to_numpy(),
        eq_feats["y_logret_h5"].to_numpy(),
        feature_names=common,
    )

    report = transfer_evaluate(
        pretrained_model=model,
        target_panel=cr_feats,
        feature_columns=common,
        label_column="y_logret_h5",
        task="regression",
        finetune_fraction=0.5,
    )
    print("Zero shot:", report.zero_shot)
    print("Fine tuned:", report.fine_tuned)
    print(f"Target train rows: {report.n_target_train}, target test rows: {report.n_target_test}")


if __name__ == "__main__":
    main()
