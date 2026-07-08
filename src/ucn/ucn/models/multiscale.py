"""
Multi scale temporal term structure network.

Each horizon is a context branch that looks back over its own time window.
Six LSTM branches read the last 1, 5, 10, 30, 90, and 180 days of the feature
sequence, each at its own temporal resolution, and each produces an embedding
of what that scale sees.  The branches are linked by their drift, the difference
between adjacent scale embeddings, which measures how the short view diverges
from the long view.  The embeddings and the drift are fused in a shared trunk,
and six horizon heads predict forward realized volatility at each horizon.  A
curvature coupling penalty keeps the six predictions on a smooth term structure.

The drift is a first class part of the forward pass, so its gradient flows back
into every branch during training.  That is the sense in which the model drift
across time windows is backpropagated as context.

The whole thing runs on the array backend, so it uses the GPU under CuPy when
UCN_GPU is set and NumPy otherwise.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as _np
from ..backend import xp as np, to_device, to_cpu, new_rng
from ..utils import sigmoid


DEFAULT_WINDOWS: List[int] = [1, 5, 10, 30, 90, 180]


def _lstm_forward(seqs, W, U, b, H):
    """Run one LSTM branch over a sequence tensor and cache for BPTT.

    seqs : (N, T, d).  Returns h_T (N, H) and a cache dict.
    """
    N, T = seqs.shape[0], seqs.shape[1]
    h = np.zeros((N, T + 1, H))
    c = np.zeros((N, T + 1, H))
    gates = np.zeros((N, T, 4 * H))
    for t in range(T):
        z = seqs[:, t, :] @ W + h[:, t, :] @ U + b
        gates[:, t, :] = z
        f = sigmoid(z[:,      :H])
        i = sigmoid(z[:,     H:2*H])
        g = np.tanh(z[:, 2*H:3*H])
        o = sigmoid(z[:, 3*H:])
        c[:, t+1, :] = f * c[:, t, :] + i * g
        h[:, t+1, :] = o * np.tanh(c[:, t+1, :])
    cache = {"seqs": seqs, "h": h, "c": c, "gates": gates, "T": T, "H": H}
    return h[:, T, :], cache


def _lstm_backward(d_hT, cache, W, U):
    """BPTT for one LSTM branch. Returns gradients for W, U, b."""
    seqs = cache["seqs"]; h = cache["h"]; c = cache["c"]
    gates = cache["gates"]; T = cache["T"]; H = cache["H"]
    dW = np.zeros_like(W); dU = np.zeros_like(U); db = np.zeros(4 * H)
    d_h_next = d_hT.copy()
    d_c_next = np.zeros_like(d_hT)
    for t in reversed(range(T)):
        z = gates[:, t, :]
        f = sigmoid(z[:,      :H])
        i = sigmoid(z[:,     H:2*H])
        g = np.tanh(z[:, 2*H:3*H])
        o = sigmoid(z[:, 3*H:])
        tanh_c = np.tanh(c[:, t+1, :])
        d_o = d_h_next * tanh_c
        d_c = d_h_next * o * (1.0 - tanh_c**2) + d_c_next
        d_f = d_c * c[:, t, :]
        d_i = d_c * g
        d_g = d_c * i
        d_cp = d_c * f
        dz_f = d_f * f * (1 - f)
        dz_i = d_i * i * (1 - i)
        dz_g = d_g * (1 - g**2)
        dz_o = d_o * o * (1 - o)
        dz = np.concatenate([dz_f, dz_i, dz_g, dz_o], axis=1)
        dW += seqs[:, t, :].T @ dz
        dU += h[:, t, :].T @ dz
        db += dz.sum(0)
        d_h_next = dz @ U.T
        d_c_next = d_cp
    return dW, dU, db


class MultiScaleTermStructureNet:
    """Six window LSTM branches fused with drift, six horizon heads, coupling."""

    def __init__(self, windows: Optional[List[int]] = None, hidden=24,
                 trunk_sizes=(128, 64), lr=1e-3, beta1=0.9, beta2=0.999,
                 lam=1e-3, dropout_rate=0.3, smooth_lambda=0.3,
                 epochs=200, batch_size=1024, patience=25, seed=42, verbose=20):
        self.windows = windows or DEFAULT_WINDOWS
        self.B = len(self.windows)
        self.H = hidden
        self.trunk_sizes = trunk_sizes
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.lam = lam
        self.dropout_rate = dropout_rate
        self.smooth_lambda = smooth_lambda
        self.epochs = epochs; self.batch_size = batch_size
        self.patience = patience; self.seed = seed; self.verbose = verbose
        self.params: dict = {}
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0
        self._rng = new_rng(seed)
        self._idx_rng = _np.random.default_rng(seed)

    def _init_weights(self, d: int):
        rng = new_rng(self.seed)
        H = self.H
        for bnc in range(self.B):
            s = math.sqrt(2.0 / (d + H))
            self.params[f"lstmW{bnc}"] = rng.standard_normal((d, 4*H)) * s
            self.params[f"lstmU{bnc}"] = rng.standard_normal((H, 4*H)) * s
            self.params[f"lstmb{bnc}"] = np.zeros(4 * H)
        # Fusion input: B embeddings plus B-1 drift vectors, each of size H.
        fuse_in = (self.B + (self.B - 1)) * H
        sizes = [fuse_in, *self.trunk_sizes]
        for i in range(len(sizes) - 1):
            sc = math.sqrt(2.0 / sizes[i])
            self.params[f"W{i+1}"] = rng.standard_normal((sizes[i], sizes[i+1])) * sc
            self.params[f"b{i+1}"] = np.zeros(sizes[i+1])
        h_last = self.trunk_sizes[-1]
        self.params["W_head"] = rng.standard_normal((h_last, self.B)) * math.sqrt(1.0/h_last)
        self.params["b_head"] = np.zeros(self.B)
        for k in self.params:
            self.m[k] = np.zeros_like(self.params[k])
            self.v[k] = np.zeros_like(self.params[k])

    def _forward(self, seq_list, training=True):
        c = {}
        embs = []
        caches = []
        for bnc in range(self.B):
            hT, cache = _lstm_forward(seq_list[bnc],
                                      self.params[f"lstmW{bnc}"],
                                      self.params[f"lstmU{bnc}"],
                                      self.params[f"lstmb{bnc}"], self.H)
            embs.append(hT)
            caches.append(cache)
        # Drift between adjacent scale embeddings.
        drifts = [embs[k+1] - embs[k] for k in range(self.B - 1)]
        fuse = np.concatenate(embs + drifts, axis=1)
        c["embs"] = embs; c["caches"] = caches; c["fuse"] = fuse

        A = fuse
        p = self.dropout_rate
        for i in range(len(self.trunk_sizes)):
            Z = A @ self.params[f"W{i+1}"] + self.params[f"b{i+1}"]
            A = np.maximum(0.0, Z)
            if training and p > 0:
                mask = (self._rng.random(A.shape) >= p).astype(A.dtype) / (1.0 - p)
                A = A * mask
                c[f"drop{i+1}"] = mask
            c[f"Z{i+1}"] = Z; c[f"A{i+1}"] = A
        logits = A @ self.params["W_head"] + self.params["b_head"]
        c["trunk"] = A
        c["P"] = sigmoid(logits)
        return c

    def _curvature_grad(self, P):
        d2 = P[:, 2:] - 2.0 * P[:, 1:-1] + P[:, :-2]
        gP = np.zeros_like(P)
        n = P.shape[0]
        scale = 2.0 / (n * max(P.shape[1] - 2, 1))
        gP[:, 2:]   += scale * d2
        gP[:, 1:-1] += scale * (-2.0 * d2)
        gP[:, :-2]  += scale * d2
        return gP

    def _backward(self, c, Y):
        g = {}
        N = Y.shape[0]
        P = c["P"]
        dlogits = (P - Y) / N
        if self.smooth_lambda > 0:
            dlogits = dlogits + self.smooth_lambda * self._curvature_grad(P) * P * (1 - P)

        A = c["trunk"]
        g["W_head"] = A.T @ dlogits
        g["b_head"] = dlogits.sum(0)
        dA = dlogits @ self.params["W_head"].T
        for i in range(len(self.trunk_sizes), 0, -1):
            if f"drop{i}" in c:
                dA = dA * c[f"drop{i}"]
            dA = dA * (c[f"Z{i}"] > 0).astype(dA.dtype)
            g[f"W{i}"] = c[f"A{i-1}" if i > 1 else "fuse"].T @ dA if i > 1 else c["fuse"].T @ dA
            g[f"b{i}"] = dA.sum(0)
            if i > 1:
                dA = dA @ self.params[f"W{i}"].T
        d_fuse = dA @ self.params["W1"].T

        # Split the fused gradient into embedding and drift parts.
        H = self.H; B = self.B
        d_embs = [d_fuse[:, k*H:(k+1)*H].copy() for k in range(B)]
        base = B * H
        for k in range(B - 1):
            d_drift = d_fuse[:, base + k*H: base + (k+1)*H]
            d_embs[k+1] += d_drift
            d_embs[k]   -= d_drift

        # BPTT each branch with its embedding gradient.
        for bnc in range(B):
            dW, dU, db = _lstm_backward(d_embs[bnc], c["caches"][bnc],
                                        self.params[f"lstmW{bnc}"],
                                        self.params[f"lstmU{bnc}"])
            g[f"lstmW{bnc}"] = dW
            g[f"lstmU{bnc}"] = dU
            g[f"lstmb{bnc}"] = db
        return g

    def _update(self, g, lr):
        self.t += 1
        eps = 1e-8
        b1c = 1 - self.beta1 ** self.t
        b2c = 1 - self.beta2 ** self.t
        for k, val in self.params.items():
            grad = g[k]
            if k.startswith("W") or k.startswith("lstmW") or k.startswith("lstmU"):
                grad = grad + self.lam * val
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grad ** 2
            self.params[k] = val - lr * (self.m[k]/b1c) / (np.sqrt(self.v[k]/b2c) + eps)

    def fit(self, seq_list, Y):
        """seq_list : list of B arrays (N, T_b, d).  Y : (N, B) labels."""
        seq_list = [to_device(s) for s in seq_list]
        Y = to_device(Y)
        d = seq_list[0].shape[2]
        if not self.params:
            self._init_weights(d)
        N = seq_list[0].shape[0]
        n_val = max(int(N * 0.15), 1)
        tr = slice(0, N - n_val); va = slice(N - n_val, N)
        idx = _np.arange(N - n_val)
        best = 1e18; best_p = None; bad = 0
        for epoch in range(self.epochs):
            self._idx_rng.shuffle(idx)
            for s in range(0, len(idx), self.batch_size):
                b = to_device(idx[s:s + self.batch_size])
                sl = [seq_list[k][tr][b] for k in range(self.B)]
                c = self._forward(sl, training=True)
                g = self._backward(c, Y[tr][b])
                self._update(g, self.lr)
            cval = self._forward([seq_list[k][va] for k in range(self.B)],
                                 training=False)
            P = cval["P"]; eps = 1e-12
            bce = float(to_cpu(-(Y[va]*np.log(P+eps)
                                 + (1-Y[va])*np.log(1-P+eps)).mean()))
            if bce < best - 1e-6:
                best = bce; best_p = {k: v.copy() for k, v in self.params.items()}; bad = 0
            else:
                bad += 1
            if self.verbose and (epoch + 1) % self.verbose == 0:
                print(f"  Epoch {epoch+1:4d}/{self.epochs}  val_BCE={bce:.5f}",
                      flush=True)
            if self.patience and bad >= self.patience:
                if self.verbose:
                    print(f"  Early stop epoch {epoch+1}", flush=True)
                break
        if best_p is not None:
            self.params = best_p
        return self

    def predict_proba(self, seq_list):
        seq_list = [to_device(s) for s in seq_list]
        return to_cpu(self._forward(seq_list, training=False)["P"])
