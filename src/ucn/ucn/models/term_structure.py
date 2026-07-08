"""
Volatility term structure network.

A single shared trunk feeds several horizon heads at once, one for each of
1, 5, 10, 30, 90, and 180 trading days.  Each head predicts whether a name
will sit in the high or low forward volatility group at that horizon.  The
heads are linked in two ways.

The first link is the shared trunk.  Every horizon is predicted from the same
hidden representation, so evidence that helps one horizon helps the others and
the model cannot learn six unrelated stories.

The second link is a term structure coupling penalty.  Volatility at nearby
horizons moves together, so the six predicted probabilities should trace a
smooth curve across horizon rather than jump around.  I penalize the second
difference of the curve, which is its curvature, and that penalty backpropagates
into every head.  A kink at the 30 day point pulls on the 10 day and 90 day
heads through the shared trunk.  This is the drift coupling: the model is trained
to keep the whole volatility term structure internally consistent, the same way
the VIX term structure stays smooth in options markets.

Everything is written against the array backend so it runs on the GPU with CuPy
when UCN_GPU is set and on NumPy otherwise.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as _np
from ..backend import xp as np, to_device, to_cpu, new_rng
from ..utils import sigmoid


# Default horizons in trading days that make up the volatility term structure.
DEFAULT_HORIZONS: List[int] = [1, 5, 10, 30, 90, 180]


class VolTermStructureNet:
    """Shared trunk with one head per horizon and a curvature coupling loss.

    Parameters
    ----------
    horizons : list of int
        Trading day horizons the heads predict, in increasing order.
    hidden_sizes : tuple of int
        Sizes of the shared trunk layers.
    lr, beta1, beta2, lam : float
        Adam learning rate, moments, and L2 weight decay.
    dropout_rate : float
        Inverted dropout on each trunk layer.
    smooth_lambda : float
        Strength of the term structure curvature penalty that couples heads.
    epochs, batch_size, patience : int
        Training schedule and early stopping patience.
    seed : int
        Random seed.
    """

    def __init__(self, horizons: Optional[List[int]] = None,
                 hidden_sizes=(128, 64), lr=1e-3, beta1=0.9, beta2=0.999,
                 lam=1e-3, dropout_rate=0.3, smooth_lambda=0.1,
                 epochs=300, batch_size=2048, patience=30, seed=42,
                 verbose=20):
        self.horizons = horizons or DEFAULT_HORIZONS
        self.H = len(self.horizons)
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.lam = lam
        self.dropout_rate = dropout_rate
        self.smooth_lambda = smooth_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.verbose = verbose

        self.params: dict = {}
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0
        self._rng = new_rng(seed)
        self._idx_rng = _np.random.default_rng(seed)
        # Log spaced horizon coordinates used to weight the curvature penalty,
        # because the gaps between horizons are uneven.
        self._hz = _np.log(_np.asarray(self.horizons, dtype=float))

    # ── Weight initialization ────────────────────────────────────────────

    def _init_weights(self, d: int):
        rng = new_rng(self.seed)
        sizes = [d, *self.hidden_sizes]
        for i in range(len(sizes) - 1):
            s = math.sqrt(2.0 / sizes[i])
            self.params[f"W{i+1}"] = rng.standard_normal((sizes[i], sizes[i+1])) * s
            self.params[f"b{i+1}"] = np.zeros(sizes[i+1])
        # One linear head per horizon, each producing a single logit.
        h_last = self.hidden_sizes[-1]
        self.params["W_head"] = rng.standard_normal((h_last, self.H)) * math.sqrt(1.0/h_last)
        self.params["b_head"] = np.zeros(self.H)
        for k in self.params:
            self.m[k] = np.zeros_like(self.params[k])
            self.v[k] = np.zeros_like(self.params[k])

    # ── Forward ──────────────────────────────────────────────────────────

    def _forward(self, X, training: bool = True) -> dict:
        c = {"A0": X}
        A = X
        p = self.dropout_rate
        for i in range(len(self.hidden_sizes)):
            Z = A @ self.params[f"W{i+1}"] + self.params[f"b{i+1}"]
            A = np.maximum(0.0, Z)
            if training and p > 0:
                mask = (self._rng.random(A.shape) >= p).astype(A.dtype) / (1.0 - p)
                A = A * mask
                c[f"drop{i+1}"] = mask
            c[f"Z{i+1}"] = Z
            c[f"A{i+1}"] = A
        # Heads produce one logit per horizon, sigmoid to per horizon probability.
        logits = A @ self.params["W_head"] + self.params["b_head"]
        P = sigmoid(logits)
        c["logits"] = logits
        c["P"] = P
        c["trunk"] = A
        return c

    # ── Curvature coupling ───────────────────────────────────────────────

    def _curvature_grad(self, P):
        """Second difference of the horizon curve and its gradient.

        For each sample the six probabilities are treated as a curve over the
        log horizon axis.  The penalty is the sum of squared second differences,
        which is large when the curve bends sharply.  I return both the mean
        penalty and its gradient with respect to P so it can join the head
        gradients.
        """
        # Second difference across the horizon axis (axis 1).
        d2 = P[:, 2:] - 2.0 * P[:, 1:-1] + P[:, :-2]        # (N, H-2)
        pen = float((d2 ** 2).mean())

        gP = np.zeros_like(P)
        # d(pen)/dP via the transpose of the second difference stencil.
        n = P.shape[0]
        scale = 2.0 / (n * max(P.shape[1] - 2, 1))
        gP[:, 2:]   += scale * d2
        gP[:, 1:-1] += scale * (-2.0 * d2)
        gP[:, :-2]  += scale * d2
        return pen, gP

    # ── Backward ─────────────────────────────────────────────────────────

    def _backward(self, c, Y):
        """Y : (N, H) binary labels per horizon. Returns gradient dict."""
        g = {}
        N = Y.shape[0]
        P = c["P"]

        # Binary cross entropy gradient per head: (P - Y) / N.
        dlogits = (P - Y) / N

        # Add the curvature coupling gradient, routed through the sigmoid.
        if self.smooth_lambda > 0:
            _, gP = self._curvature_grad(P)
            dlogits = dlogits + self.smooth_lambda * gP * P * (1.0 - P)

        A = c["trunk"]
        g["W_head"] = A.T @ dlogits
        g["b_head"] = dlogits.sum(0)

        dA = dlogits @ self.params["W_head"].T
        for i in range(len(self.hidden_sizes), 0, -1):
            if f"drop{i}" in c:
                dA = dA * c[f"drop{i}"]
            dA = dA * (c[f"Z{i}"] > 0).astype(dA.dtype)
            g[f"W{i}"] = c[f"A{i-1}"].T @ dA
            g[f"b{i}"] = dA.sum(0)
            if i > 1:
                dA = dA @ self.params[f"W{i}"].T
        return g

    # ── Adam update ──────────────────────────────────────────────────────

    def _update(self, g, lr):
        self.t += 1
        eps = 1e-8
        b1c = 1.0 - self.beta1 ** self.t
        b2c = 1.0 - self.beta2 ** self.t
        for k, val in self.params.items():
            grad = g[k]
            if k.startswith("W"):
                grad = grad + self.lam * val
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[k] / b1c
            v_hat = self.v[k] / b2c
            self.params[k] = val - lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── Fit ──────────────────────────────────────────────────────────────

    def fit(self, X, Y):
        """X : (N, d) features.  Y : (N, H) binary labels per horizon."""
        X = to_device(X)
        Y = to_device(Y)
        if not self.params:
            self._init_weights(X.shape[1])

        n_val = max(int(len(X) * 0.15), 1)
        X_tr, Y_tr = X[:len(X)-n_val], Y[:len(Y)-n_val]
        X_val, Y_val = X[len(X)-n_val:], Y[len(Y)-n_val:]
        idx = _np.arange(len(X_tr))

        best_val = 1e18
        best_params = None
        no_improve = 0

        for epoch in range(self.epochs):
            self._idx_rng.shuffle(idx)
            ep_bce = 0.0; n_b = 0
            for s in range(0, len(X_tr), self.batch_size):
                b = to_device(idx[s:s + self.batch_size])
                c = self._forward(X_tr[b], training=True)
                g = self._backward(c, Y_tr[b])
                self._update(g, self.lr)
                Pb = c["P"]; eps = 1e-12
                ep_bce += float(to_cpu(-(Y_tr[b]*np.log(Pb+eps)
                                         + (1-Y_tr[b])*np.log(1-Pb+eps)).mean()))
                n_b += 1

            c_val = self._forward(X_val, training=False)
            P = c_val["P"]
            eps = 1e-12
            bce = float(to_cpu(-(Y_val * np.log(P + eps)
                                 + (1 - Y_val) * np.log(1 - P + eps)).mean()))
            # Accuracy on the CPU to avoid a CuPy boolean reduction kernel that
            # fails to compile against the mismatched CUDA headers on this host.
            Pc = to_cpu(P); Yc = to_cpu(Y_val)
            val_acc = float(((Pc >= 0.5) == (Yc >= 0.5)).mean())
            tr_bce = ep_bce / max(n_b, 1)
            if bce < best_val - 1e-6:
                best_val = bce
                best_params = {k: v.copy() for k, v in self.params.items()}
                no_improve = 0
            else:
                no_improve += 1
            if self.verbose and (epoch + 1) % self.verbose == 0:
                marker = " *" if no_improve == 0 else ""
                print(f"  Epoch {epoch+1:4d}/{self.epochs}  "
                      f"train_BCE={tr_bce:.5f}  val_BCE={bce:.5f}  "
                      f"val_acc={val_acc:.4f}{marker}", flush=True)
            if self.patience and no_improve >= self.patience:
                if self.verbose:
                    print(f"  Early stop epoch {epoch+1}", flush=True)
                break

        if best_params is not None:
            self.params = best_params
        return self

    # ── Predict ──────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """Return per horizon probabilities, shape (N, H)."""
        out = self._forward(to_device(X), training=False)["P"]
        return to_cpu(out)
