"""
Quantile term structure network: predict price ranges, not point prices.

Instead of a single number, each horizon head outputs several quantiles of the
forward log return, the 5th, 25th, 50th, 75th, and 95th percentiles.  The band
between the low and high quantiles is the predicted price range: multiply
today's price by the exponential of the quantile returns.

The quantiles are trained with the pinball loss, the standard objective for
quantile regression.  To keep the band valid I construct the quantiles as a
base level plus positive increments, so they never cross, a low quantile is
always below a high quantile.

Because the drift of returns is close to unpredictable, the median quantile
tends toward zero, while the outer quantiles widen in step with the predicted
volatility.  The result is an honest prediction cone: it does not claim to know
the price, it states the range the price should occupy and how wide that range
is at each horizon.

Runs on the array backend, GPU under CuPy when UCN_GPU is set.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as _np
from ..backend import xp as np, to_device, to_cpu, new_rng


DEFAULT_QUANTILES: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95]


def _softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class QuantileTermStructureNet:
    """Shared trunk with one quantile fan per horizon, trained by pinball loss."""

    def __init__(self, horizons: List[int], quantiles: Optional[List[float]] = None,
                 hidden_sizes=(128, 64), lr=1e-3, beta1=0.9, beta2=0.999,
                 lam=1e-4, dropout_rate=0.2, epochs=400, batch_size=2048,
                 patience=40, seed=42, verbose=20):
        self.horizons = horizons
        self.H = len(horizons)
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.Q = len(self.quantiles)
        self.hidden_sizes = hidden_sizes
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.lam = lam
        self.dropout_rate = dropout_rate
        self.epochs = epochs; self.batch_size = batch_size
        self.patience = patience; self.seed = seed; self.verbose = verbose
        self.params: dict = {}
        self.m: dict = {}; self.v: dict = {}; self.t = 0
        self._rng = new_rng(seed)
        self._idx_rng = _np.random.default_rng(seed)
        self._tau = _np.asarray(self.quantiles).reshape(1, 1, self.Q)

    def _init_weights(self, d):
        rng = new_rng(self.seed)
        sizes = [d, *self.hidden_sizes]
        for i in range(len(sizes) - 1):
            s = math.sqrt(2.0 / sizes[i])
            self.params[f"W{i+1}"] = rng.standard_normal((sizes[i], sizes[i+1])) * s
            self.params[f"b{i+1}"] = np.zeros(sizes[i+1])
        h_last = self.hidden_sizes[-1]
        out = self.H * self.Q
        self.params["W_head"] = rng.standard_normal((h_last, out)) * math.sqrt(1.0/h_last)
        self.params["b_head"] = np.zeros(out)
        for k in self.params:
            self.m[k] = np.zeros_like(self.params[k])
            self.v[k] = np.zeros_like(self.params[k])

    def _forward(self, X, training=True):
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
            c[f"Z{i+1}"] = Z; c[f"A{i+1}"] = A
        raw = A @ self.params["W_head"] + self.params["b_head"]
        N = raw.shape[0]
        raw3 = raw.reshape(N, self.H, self.Q)
        # Monotone quantiles: base plus positive increments so they never cross.
        q0 = raw3[:, :, 0:1]
        sp = _softplus(raw3[:, :, 1:])
        cum = np.cumsum(sp, axis=2)
        q = np.concatenate([q0, q0 + cum], axis=2)
        c["trunk"] = A; c["raw3"] = raw3; c["q"] = q
        return c

    def _backward(self, c, Y):
        # Y : (N, H) forward returns.  Pinball gradient per quantile.
        q = c["q"]
        N = Y.shape[0]
        tau = np.asarray(self._tau)
        diff = Y[:, :, None] - q                      # (N, H, Q)
        under = (diff < 0).astype(q.dtype)
        grad_q = (-tau + under) / (N * self.H * self.Q)

        # Backprop through the monotone construction into raw3.
        raw3 = c["raw3"]
        grad_raw = np.zeros_like(raw3)
        grad_raw[:, :, 0] = grad_q.sum(axis=2)        # base affects all quantiles
        for r in range(1, self.Q):
            dsp = grad_q[:, :, r:].sum(axis=2)         # increment r affects q_k, k>=r
            grad_raw[:, :, r] = dsp * _sigmoid(raw3[:, :, r])
        d_head = grad_raw.reshape(N, self.H * self.Q)

        g = {}
        A = c["trunk"]
        g["W_head"] = A.T @ d_head
        g["b_head"] = d_head.sum(0)
        dA = d_head @ self.params["W_head"].T
        for i in range(len(self.hidden_sizes), 0, -1):
            if f"drop{i}" in c:
                dA = dA * c[f"drop{i}"]
            dA = dA * (c[f"Z{i}"] > 0).astype(dA.dtype)
            g[f"W{i}"] = c[f"A{i-1}"].T @ dA
            g[f"b{i}"] = dA.sum(0)
            if i > 1:
                dA = dA @ self.params[f"W{i}"].T
        return g

    def _update(self, g, lr):
        self.t += 1
        eps = 1e-8
        b1c = 1 - self.beta1 ** self.t
        b2c = 1 - self.beta2 ** self.t
        for k, val in self.params.items():
            grad = g[k]
            if k.startswith("W"):
                grad = grad + self.lam * val
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grad ** 2
            self.params[k] = val - lr * (self.m[k]/b1c) / (np.sqrt(self.v[k]/b2c) + eps)

    def _pinball(self, q, Y):
        tau = np.asarray(self._tau)
        diff = Y[:, :, None] - q
        return float(to_cpu(np.maximum(tau * diff, (tau - 1) * diff).mean()))

    def fit(self, X, Y):
        X = to_device(X); Y = to_device(Y)
        if not self.params:
            self._init_weights(X.shape[1])
        n_val = max(int(len(X) * 0.15), 1)
        X_tr, Y_tr = X[:len(X)-n_val], Y[:len(Y)-n_val]
        X_val, Y_val = X[len(X)-n_val:], Y[len(Y)-n_val:]
        idx = _np.arange(len(X_tr))
        best = 1e18; best_p = None; bad = 0
        for epoch in range(self.epochs):
            self._idx_rng.shuffle(idx)
            ep = 0.0; n_b = 0
            for s in range(0, len(X_tr), self.batch_size):
                b = to_device(idx[s:s + self.batch_size])
                c = self._forward(X_tr[b], training=True)
                g = self._backward(c, Y_tr[b])
                self._update(g, self.lr)
                ep += self._pinball(c["q"], Y_tr[b]); n_b += 1
            cval = self._forward(X_val, training=False)
            vloss = self._pinball(cval["q"], Y_val)
            # Coverage of the outer band on validation.
            qv = to_cpu(cval["q"]); Yv = to_cpu(Y_val)
            cover = float(((Yv >= qv[:, :, 0]) & (Yv <= qv[:, :, -1])).mean())
            if vloss < best - 1e-7:
                best = vloss; best_p = {k: v.copy() for k, v in self.params.items()}; bad = 0
            else:
                bad += 1
            if self.verbose and (epoch + 1) % self.verbose == 0:
                marker = " *" if bad == 0 else ""
                print(f"  Epoch {epoch+1:4d}/{self.epochs}  "
                      f"train_pinball={ep/max(n_b,1):.6f}  "
                      f"val_pinball={vloss:.6f}  "
                      f"val_coverage={cover:.3f}{marker}", flush=True)
            if self.patience and bad >= self.patience:
                if self.verbose:
                    print(f"  Early stop epoch {epoch+1}", flush=True)
                break
        if best_p is not None:
            self.params = best_p
        return self

    def predict_quantiles(self, X):
        """Return predicted return quantiles, shape (N, H, Q)."""
        return to_cpu(self._forward(to_device(X), training=False)["q"])
