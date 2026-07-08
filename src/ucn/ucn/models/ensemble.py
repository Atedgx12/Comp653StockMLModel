"""
VolatilityEnsemble — combine the cross-sectional UCN volatility model with the
multi-scale term-structure model into a single calibrated probability.

Both models predict the same event, P(next-horizon realized volatility is above
the cross-sectional median). They consume different representations, so the
ensemble must be handed aligned inputs for the same (ticker, date) rows:

  - X_raw     : the UCN feature matrix, already reduced to the saved selected
                columns in the saved order (the price features plus the optional
                sentiment column). The saved scaler is applied internally.
  - ucn_seqs  : optional LSTM sequences for the UCN branch, or None.
  - ms_seqs   : the list of six window sequences the multi-scale model expects.

The combined probability is a convex blend a * P_ucn + (1 - a) * P_ms. The blend
weight can be fixed or fit on a validation set with fit_weights.
"""
import os
import numpy as np

from .unified_network import UnifiedCourseNetwork
from .multiscale import MultiScaleTermStructureNet
from ..training.metrics import roc_auc


class VolatilityEnsemble:
    def __init__(self, ucn=None, ucn_scaler=None, multiscale=None,
                 horizon=None, weights=(0.5, 0.5)):
        self.ucn = ucn
        self.ucn_scaler = ucn_scaler   # dict with mu, sd, selected, has_sent
        self.ms = multiscale
        self.horizon = horizon
        self.weights = tuple(weights)
        self.val_auc = None

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def load(cls, out_dir, cfg=None, horizon=None, weights=(0.5, 0.5),
             ucn_weights="ucn_weights.npz", ucn_scaler="ucn_scaler.npz",
             ms_ckpt=None):
        """Load whichever saved models are present in out_dir.

        cfg is the UCNConfig used to build the UCN skeleton before loading its
        weights. ms_ckpt is the multi-scale checkpoint filename, e.g.
        multiscale_daily.npz or multiscale_intraday_5m.npz.
        """
        ucn = scaler = ms = None
        wpath = os.path.join(out_dir, ucn_weights)
        spath = os.path.join(out_dir, ucn_scaler)
        if cfg is not None and os.path.exists(wpath):
            ucn = UnifiedCourseNetwork.from_checkpoint(wpath, cfg)
        if os.path.exists(spath):
            sc = np.load(spath, allow_pickle=True)
            scaler = {"mu": sc["mu"], "sd": sc["sd"],
                      "selected": sc["selected"].tolist(),
                      "has_sent": bool(sc["has_sent"])}
            if horizon is None and "horizon" in sc.files:
                horizon = int(sc["horizon"])
        if ms_ckpt is not None:
            mpath = os.path.join(out_dir, ms_ckpt)
            if os.path.exists(mpath):
                ms = MultiScaleTermStructureNet.load(mpath)
        return cls(ucn, scaler, ms, horizon, weights)

    # ── Per-model probabilities ──────────────────────────────────────────

    def _ucn_prob(self, X_raw, seqs=None):
        mu = self.ucn_scaler["mu"]; sd = self.ucn_scaler["sd"]
        Xs = (np.asarray(X_raw, dtype=float) - mu) / sd
        return np.asarray(self.ucn.predict_proba(Xs, seqs=seqs))[:, 1]

    def _branch_for_horizon(self):
        w = list(self.ms.windows)
        if self.horizon in w:
            return w.index(self.horizon)
        return int(np.argmin([abs(x - self.horizon) for x in w]))

    def _ms_prob(self, ms_seqs):
        P = np.asarray(self.ms.predict_proba(ms_seqs))   # (N, B)
        return P[:, self._branch_for_horizon()]

    # ── Combined prediction ──────────────────────────────────────────────

    def predict_proba(self, X_raw=None, ucn_seqs=None, ms_seqs=None):
        parts = []; wts = []
        if self.ucn is not None and X_raw is not None:
            parts.append(self._ucn_prob(X_raw, ucn_seqs)); wts.append(self.weights[0])
        if self.ms is not None and ms_seqs is not None:
            parts.append(self._ms_prob(ms_seqs)); wts.append(self.weights[1])
        if not parts:
            raise ValueError("no model inputs supplied to the ensemble")
        wts = np.asarray(wts, dtype=float)
        wts = wts / wts.sum()
        return sum(w * p for w, p in zip(wts, parts))

    def predict(self, **kw):
        return (self.predict_proba(**kw) >= 0.5).astype(int)

    # ── Weight optimization ──────────────────────────────────────────────

    def fit_weights(self, y_true, X_raw=None, ucn_seqs=None, ms_seqs=None,
                    grid=None):
        """Pick the convex blend weight that maximizes validation AUC.

        Falls back to whichever single model is available when only one set of
        inputs is provided.
        """
        pu = self._ucn_prob(X_raw, ucn_seqs) if (self.ucn is not None
                                                 and X_raw is not None) else None
        pm = self._ms_prob(ms_seqs) if (self.ms is not None
                                        and ms_seqs is not None) else None
        if pu is None and pm is None:
            raise ValueError("no model inputs supplied to fit_weights")
        if pu is None:
            self.weights = (0.0, 1.0); self.val_auc = roc_auc(y_true, pm); return self
        if pm is None:
            self.weights = (1.0, 0.0); self.val_auc = roc_auc(y_true, pu); return self
        grid = np.linspace(0.0, 1.0, 21) if grid is None else np.asarray(grid)
        best_a, best_auc = 0.5, -1.0
        for a in grid:
            auc = roc_auc(y_true, a * pu + (1.0 - a) * pm)
            if auc > best_auc:
                best_auc, best_a = auc, float(a)
        self.weights = (best_a, 1.0 - best_a)
        self.val_auc = best_auc
        return self
