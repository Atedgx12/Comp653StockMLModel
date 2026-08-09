# Evaluation and Reproduction Notes

The repository includes the checkpoints and reported metrics from the final
course run, but it does not yet provide a one-command scientific reproduction
of those values. The automated suite checks imports, tensor shapes, prediction
bounds, checkpoint round trips, baseline utilities, and both supported Python
versions. It intentionally avoids downloading a changing live market dataset.

Before using the reported AUC and coverage values as research or production
estimates, a reproducibility run should address these items:

1. **Fit mutual-information selection on training rows only.** The current
   final runners construct context features before the outer temporal split and
   pass all available labels to the selector. The selected feature names should
   instead be learned on the training interval and then applied unchanged to
   validation and test rows.
2. **Use chronological validation and calibration sets.** The network currently
   takes the final 15% of the supplied training arrays for both early stopping
   and conformal calibration. The runners build rows in ticker groups, so the
   inner split should be made explicitly by timestamp. A separate calibration
   interval would also avoid reusing the early-stopping set for conformal
   widening.
3. **Compare like-for-like baselines.** Every baseline should use the same
   volatility-rank label, horizon, universe, purge, and held-out timestamps as
   the multi-scale model. The report's LightGBM mean and the single held-out
   multi-scale AUC are useful historical references but are not a paired test at
   all twelve horizons.
4. **Freeze the data manifest.** Record provider versions, download timestamps,
   ticker membership, date bounds, exclusions, and hashes of cached inputs.
   This is especially important for `yfinance` intraday retention windows and
   changing index constituents.
5. **Emit machine-readable run metadata.** Save the command-line arguments,
   random seed, dependency versions, selected feature names, split dates,
   checkpoint hash, and per-horizon metrics next to every checkpoint.

These are evaluation hardening tasks; they do not change the current model
architecture, retained checkpoints, or the values quoted from the course
report.
