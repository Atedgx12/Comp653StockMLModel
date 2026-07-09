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
from ..training.metrics import roc_auc


DEFAULT_WINDOWS: List[int] = [1, 5, 10, 30, 90, 180]


def _nan_auc_mean(Yc, Pc):
    """Mean per column AUC that skips NaN labels left by an extreme split."""
    aucs = []
    for b in range(Yc.shape[1]):
        m = ~_np.isnan(Yc[:, b])
        if int(m.sum()) > 10 and _np.unique(Yc[m, b]).size > 1:
            aucs.append(roc_auc(Yc[m, b], Pc[m, b]))
    return float(_np.mean(aucs)) if aucs else 0.0


def _softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _lstm_forward(seqs, W, U, b, H):
    """Run one LSTM branch over a sequence tensor and cache for BPTT.

    seqs : (N, T, d).  Returns h_T (N, H) and a cache dict.
    """
    N, T = seqs.shape[0], seqs.shape[1]
    h = np.zeros((N, T + 1, H), dtype=seqs.dtype)
    c = np.zeros((N, T + 1, H), dtype=seqs.dtype)
    gates = np.zeros((N, T, 4 * H), dtype=seqs.dtype)
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
                 additivity_lambda=0.0,
                 epochs=200, batch_size=1024, patience=25, seed=42, verbose=20,
                 quantiles=None, quantile_weight=1.0, alpha=0.10,
                 lr_decay=0.5, lr_patience=40, min_lr=1e-5, standardize=True,
                 dtype="float32", warm_restarts=False, restart_period=120,
                 restart_mult=1.0):
        self.windows = windows or DEFAULT_WINDOWS
        self.B = len(self.windows)
        self.H = hidden
        self.trunk_sizes = trunk_sizes
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.lam = lam
        self.dropout_rate = dropout_rate
        self.smooth_lambda = smooth_lambda
        # Variance additivity coupling. The squared 90 percent band width is an
        # integrated variance proxy, and dividing it by the horizon length gives
        # a variance rate that additivity holds roughly constant across the term
        # structure. Penalizing the curvature of that rate lets the reliable
        # long horizon band regularize the noisy short horizon ones.
        self.additivity_lambda = additivity_lambda
        self._Ht = None
        self.epochs = epochs; self.batch_size = batch_size
        self.patience = patience; self.seed = seed; self.verbose = verbose
        # Reduce-on-plateau schedule: when validation loss stops improving for
        # lr_patience epochs the rate is multiplied by lr_decay down to min_lr.
        self.lr_decay = lr_decay; self.lr_patience = lr_patience
        self.min_lr = min_lr
        # Cosine warm restarts. When enabled the rate follows a cosine from the
        # base rate down to min_lr over restart_period epochs, then jumps back to
        # the base rate and the Adam moments are reset, which excites the model
        # out of a local minimum. restart_mult lengthens each successive cycle.
        self.warm_restarts = warm_restarts
        self.restart_period = restart_period
        self.restart_mult = restart_mult
        # Quantile price band heads, trained jointly with the volatility heads.
        self.quantiles = quantiles or [0.05, 0.25, 0.50, 0.75, 0.95]
        self.Q = len(self.quantiles)
        self.quantile_weight = quantile_weight
        self.alpha = alpha           # 1 - alpha is the target band coverage
        self.conformal_delta = None  # per horizon widening set after fit
        self._taus = None
        # Per branch feature scaler, learned in fit and reused at predict time
        # so a saved model is self contained and reapplies to raw sequences.
        self.standardize = standardize
        # Single precision by default. float32 halves memory traffic, which
        # doubles effective GPU L2 and CPU L3 residency and bandwidth, with no
        # accuracy cost here because inputs are standardized and labels are ranks.
        self.dtype = _np.float32 if str(dtype) == "float32" else _np.float64
        self.scalers = None          # list of (mu, sd) per branch
        self.d_ctx = 0               # static context feature dimension
        self.ctx_scaler = None       # (mu, sd) for the static context vector
        self.params: dict = {}
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0
        self._rng = new_rng(seed)
        self._idx_rng = _np.random.default_rng(seed)

    def _init_weights(self, d: int, d_ctx: int = 0):
        rng = new_rng(self.seed)
        H = self.H
        self.d_ctx = d_ctx
        for bnc in range(self.B):
            s = math.sqrt(2.0 / (d + H))
            self.params[f"lstmW{bnc}"] = rng.standard_normal((d, 4*H)) * s
            self.params[f"lstmU{bnc}"] = rng.standard_normal((H, 4*H)) * s
            self.params[f"lstmb{bnc}"] = np.zeros(4 * H)
        # Fusion input: B embeddings, B-1 drift vectors, and the static context
        # vector of cross sectional features that carries the hierarchy signal.
        fuse_in = (self.B + (self.B - 1)) * H + d_ctx
        sizes = [fuse_in, *self.trunk_sizes]
        for i in range(len(sizes) - 1):
            sc = math.sqrt(2.0 / sizes[i])
            self.params[f"W{i+1}"] = rng.standard_normal((sizes[i], sizes[i+1])) * sc
            self.params[f"b{i+1}"] = np.zeros(sizes[i+1])
        h_last = self.trunk_sizes[-1]
        self.params["W_head"] = rng.standard_normal((h_last, self.B)) * math.sqrt(1.0/h_last)
        self.params["b_head"] = np.zeros(self.B)
        # Quantile price band heads: B horizons times Q quantiles.
        self.params["W_q"] = rng.standard_normal((h_last, self.B * self.Q)) * math.sqrt(1.0/h_last)
        self.params["b_q"] = np.zeros(self.B * self.Q)
        # Hierarchy gate on the static context. 2*sigmoid keeps each per feature
        # scale in (0, 2), and the zero initialization starts the gate at one so
        # it is neutral before training learns which feature groups matter.
        if d_ctx > 0:
            self.params["w_ctx"] = np.zeros(d_ctx)
        for k in self.params:
            self.params[k] = self.params[k].astype(self.dtype)
        for k in self.params:
            self.m[k] = np.zeros_like(self.params[k])
            self.v[k] = np.zeros_like(self.params[k])

    def _forward(self, seq_list, ctx=None, training=True):
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
        parts = embs + drifts
        # Gate and append the static context vector when present.
        if self.d_ctx > 0 and ctx is not None:
            gate = 2.0 * sigmoid(self.params["w_ctx"])
            ctx_g = ctx * gate
            parts = parts + [ctx_g]
            c["ctx"] = ctx; c["gate"] = gate
        fuse = np.concatenate(parts, axis=1)
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

        # Quantile band heads with a non crossing construction: the lowest
        # quantile is a free level and each higher quantile adds a positive
        # softplus increment, so the band edges can never cross.
        raw = (A @ self.params["W_q"] + self.params["b_q"]).reshape(-1, self.B, self.Q)
        inc = _softplus(raw[:, :, 1:])
        q0 = raw[:, :, :1]
        q = np.concatenate([q0, q0 + np.cumsum(inc, axis=2)], axis=2)
        c["raw"] = raw
        c["q"] = q
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

    def _additivity_grad(self, q, raw):
        # Integrated variance additivity in scale free form. The band width
        # w = q_high - q_low is a forward return spread, and under variance
        # additivity for a random walk it grows as w proportional to sqrt(H), so
        # 2*log(w) tracks log(H) up to a constant and the curvature of log(w)
        # along the horizon axis must match half the curvature of log(H). The
        # loss is the squared residual of that match, which lets the reliable
        # long horizon band regularize the noisy short horizon ones. Working in
        # log space keeps the gradient well conditioned regardless of the return
        # units, and the width depends only on the positive increments so the
        # gradient flows through softplus into the increment logits.
        N, B, Q = q.shape
        if B < 3 or self.additivity_lambda <= 0:
            return np.zeros_like(raw)
        eps = 1e-6
        w = q[:, :, -1] - q[:, :, 0]
        Lw = np.log(w + eps)
        C = Lw[:, 2:] - 2.0 * Lw[:, 1:-1] + Lw[:, :-2]
        if self._Ht is None:
            H = _np.asarray(self.windows, dtype=self.dtype)
            lH = _np.log(H)
            t = lH[2:] - 2.0 * lH[1:-1] + lH[:-2]
            self._Ht = to_device(t.reshape(1, -1))
        res = 2.0 * C - self._Ht
        dC = (4.0 * self.additivity_lambda / (N * max(B - 2, 1))) * res
        gL = np.zeros_like(Lw)
        gL[:, 2:]   += dC
        gL[:, 1:-1] += -2.0 * dC
        gL[:, :-2]  += dC
        dLdw = gL / (w + eps)
        draw = np.zeros_like(raw)
        draw[:, :, 1:] = dLdw[:, :, None] * _sigmoid(raw[:, :, 1:])
        return draw

    def _backward(self, c, Y, Yret=None):
        g = {}
        N = Y.shape[0]
        P = c["P"]
        # Masked cross entropy. Under an extreme label split some horizons carry
        # NaN for the dropped middle, so those targets are set equal to P to give
        # them exactly zero gradient while the labeled horizons still train.
        Yf = np.where(np.isnan(Y), P, Y)
        dlogits = (P - Yf) / N
        if self.smooth_lambda > 0:
            dlogits = dlogits + self.smooth_lambda * self._curvature_grad(P) * P * (1 - P)

        A = c["trunk"]
        g["W_head"] = A.T @ dlogits
        g["b_head"] = dlogits.sum(0)
        dA = dlogits @ self.params["W_head"].T

        # Quantile head gradient via the pinball loss. Yret is the forward
        # return per horizon.  The gradient flows through the non crossing
        # construction back into the shared trunk, so the price bands train the
        # same representation as the volatility heads. The additivity coupling
        # adds into the same increment logits when enabled.
        need_q = (Yret is not None) or (self.additivity_lambda > 0)
        if need_q:
            q = c["q"]; raw = c["raw"]
            Q, B = self.Q, self.B
            draw = np.zeros_like(raw)
            if Yret is not None:
                if self._taus is None:
                    self._taus = to_device(_np.asarray(self.quantiles,
                                           dtype=self.dtype).reshape(1, 1, Q))
                e = Yret[:, :, None] - q                    # (N, B, Q)
                dq = np.where(e > 0, -self._taus, 1.0 - self._taus)
                dq = dq * (self.quantile_weight / (N * B * Q))
                draw[:, :, 0] = dq.sum(axis=2)
                # d(loss)/d(inc_k) is the reverse cumulative sum of the upper dq.
                d_inc = np.cumsum(dq[:, :, 1:][:, :, ::-1], axis=2)[:, :, ::-1]
                draw[:, :, 1:] += d_inc * _sigmoid(raw[:, :, 1:])
            if self.additivity_lambda > 0:
                draw += self._additivity_grad(q, raw)
            draw_flat = draw.reshape(N, B * Q)
            g["W_q"] = A.T @ draw_flat
            g["b_q"] = draw_flat.sum(0)
            dA = dA + draw_flat @ self.params["W_q"].T
        else:
            g["W_q"] = np.zeros_like(self.params["W_q"])
            g["b_q"] = np.zeros_like(self.params["b_q"])

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
        # Static context gradient through the hierarchy gate, if present.
        if self.d_ctx > 0:
            if "ctx" in c:
                base_ctx = (2 * B - 1) * H
                d_ctx_g = d_fuse[:, base_ctx:base_ctx + self.d_ctx]
                gate = c["gate"]
                dgate = gate * (1.0 - gate / 2.0)   # derivative of 2*sigmoid
                g["w_ctx"] = (d_ctx_g * c["ctx"]).sum(0) * dgate
            else:
                g["w_ctx"] = np.zeros_like(self.params["w_ctx"])
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

    def _fit_scalers(self, seq_list):
        """Learn a per branch mean and std from the training sequences."""
        self.scalers = []
        for s in seq_list:
            flat = s.reshape(-1, s.shape[2])
            mu = flat.mean(0); sd = flat.std(0) + 1e-9
            self.scalers.append((mu, sd))

    def _apply_scalers(self, seq_list):
        """Standardize each branch with its stored scaler, if present."""
        if not self.scalers:
            return seq_list
        return [(s - mu) / sd for s, (mu, sd) in zip(seq_list, self.scalers)]

    def _apply_ctx(self, ctx):
        """Standardize a static context matrix with the stored scaler."""
        if ctx is None or self.ctx_scaler is None:
            return None
        ctx = to_device(ctx).astype(self.dtype)
        mu, sd = self.ctx_scaler
        return (ctx - mu) / sd

    def fit(self, seq_list, Y, Yret=None, ctx=None):
        """seq_list : list of B arrays (N, T_b, d).  Y : (N, B) vol labels.
        Yret : optional (N, B) forward returns for the quantile band heads.
        ctx : optional (N, d_ctx) static cross sectional features, gated and
        fused into the trunk so the hierarchy signal reaches every horizon."""
        seq_list = [to_device(s) for s in seq_list]
        if self.standardize:
            self._fit_scalers(seq_list)
            seq_list = self._apply_scalers(seq_list)
        ctx_d = None
        if ctx is not None:
            ctx_d = to_device(ctx).astype(self.dtype)
            mu = ctx_d.mean(0); sd = ctx_d.std(0) + 1e-9
            self.ctx_scaler = (mu, sd)
            ctx_d = (ctx_d - mu) / sd
        Y = to_device(Y)
        Yret_d = to_device(Yret) if Yret is not None else None
        d = seq_list[0].shape[2]
        d_ctx = int(ctx_d.shape[1]) if ctx_d is not None else 0
        if not self.params:
            self._init_weights(d, d_ctx)
        N = seq_list[0].shape[0]
        n_val = max(int(N * 0.15), 1)
        tr = slice(0, N - n_val); va = slice(N - n_val, N)
        idx = _np.arange(N - n_val)
        best = 1e18; best_p = None; bad = 0
        cur_lr = self.lr; plateau = 0
        # Fixed slices for monitoring train and validation AUC on log steps.
        seq_va = [seq_list[k][va] for k in range(self.B)]
        Yc_va = to_cpu(Y[va])
        ctx_va = ctx_d[va] if ctx_d is not None else None
        n_eval = min(N - n_val, 16384)
        seq_tre = [seq_list[k][0:n_eval] for k in range(self.B)]
        Yc_tre = to_cpu(Y[0:n_eval])
        ctx_tre = ctx_d[0:n_eval] if ctx_d is not None else None
        cycle_len = int(self.restart_period)
        cycle_start = 0
        for epoch in range(self.epochs):
            # Cosine warm restart schedule with a moment reset at each restart.
            if self.warm_restarts:
                t_in = epoch - cycle_start
                cur_lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * \
                    (1.0 + math.cos(math.pi * t_in / max(cycle_len, 1)))
                if t_in + 1 >= cycle_len:
                    for k in self.params:
                        self.m[k] = np.zeros_like(self.params[k])
                        self.v[k] = np.zeros_like(self.params[k])
                    self.t = 0
                    cycle_start = epoch + 1
                    cycle_len = max(1, int(cycle_len * self.restart_mult))
            self._idx_rng.shuffle(idx)
            ep_bce = 0.0; ep_acc = 0.0; n_b = 0
            for s in range(0, len(idx), self.batch_size):
                b = to_device(idx[s:s + self.batch_size])
                sl = [seq_list[k][tr][b] for k in range(self.B)]
                cb = ctx_d[tr][b] if ctx_d is not None else None
                c = self._forward(sl, cb, training=True)
                yr = Yret_d[tr][b] if Yret_d is not None else None
                g = self._backward(c, Y[tr][b], yr)
                self._update(g, cur_lr)
                Pbc = to_cpu(c["P"]); Ybc = to_cpu(Y[tr][b]); eps = 1e-12
                mb = ~_np.isnan(Ybc)
                if mb.any():
                    bce_el = -(Ybc*_np.log(Pbc+eps)
                               + (1-Ybc)*_np.log(1-Pbc+eps))
                    ep_bce += float(bce_el[mb].mean())
                    ep_acc += float(((Pbc[mb] >= 0.5) == (Ybc[mb] >= 0.5)).mean())
                    n_b += 1
            cval = self._forward(seq_va, ctx_va, training=False)
            P = cval["P"]; eps = 1e-12
            Pc = to_cpu(P); Yc = Yc_va
            mv = ~_np.isnan(Yc)
            bce_el = -(Yc*_np.log(Pc+eps) + (1-Yc)*_np.log(1-Pc+eps))
            bce = float(bce_el[mv].mean())
            val_acc = float(((Pc[mv] >= 0.5) == (Yc[mv] >= 0.5)).mean())
            tr_bce = ep_bce / max(n_b, 1)
            tr_acc = ep_acc / max(n_b, 1)
            if bce < best - 1e-6:
                best = bce; best_p = {k: v.copy() for k, v in self.params.items()}; bad = 0
                plateau = 0
            else:
                bad += 1
                plateau += 1
                # Decay the rate on a shorter fuse than the early-stop patience
                # so training keeps refining once the loss flattens. The plateau
                # decay is off under warm restarts, which own the schedule.
                if (not self.warm_restarts and plateau >= self.lr_patience
                        and cur_lr > self.min_lr):
                    cur_lr = max(cur_lr * self.lr_decay, self.min_lr)
                    plateau = 0
            if self.verbose and (epoch + 1) % self.verbose == 0:
                marker = " *" if bad == 0 else ""
                val_auc = _nan_auc_mean(Yc, Pc)
                Ptre = to_cpu(self._forward(seq_tre, ctx_tre, training=False)["P"])
                tr_auc = _nan_auc_mean(Yc_tre, Ptre)
                print(f"  Epoch {epoch+1:4d}/{self.epochs}  "
                      f"CE={tr_bce:.5f}  acc={tr_acc:.4f}  AUC={tr_auc:.4f}  "
                      f"val_CE={bce:.5f}  val_acc={val_acc:.4f}  "
                      f"val_AUC={val_auc:.4f}  LR={cur_lr:.2e}{marker}",
                      flush=True)
            if self.patience and not self.warm_restarts and bad >= self.patience:
                if self.verbose:
                    print(f"  Early stop epoch {epoch+1}", flush=True)
                break
        if best_p is not None:
            self.params = best_p

        # Conformal calibration of the band edges on the validation slice, so
        # the outer band covers its nominal level out of sample by construction.
        if Yret is not None:
            cval = self._forward([seq_list[k][va] for k in range(self.B)],
                                 ctx_va, training=False)
            q = to_cpu(cval["q"])                 # (nval, B, Q)
            yv = to_cpu(Yret_d[va])               # (nval, B)
            lo = q[:, :, 0]; hi = q[:, :, -1]
            scores = _np.maximum(lo - yv, yv - hi)   # conformity score per row
            n = scores.shape[0]
            k = int(_np.ceil((1 - self.alpha) * (n + 1))) - 1
            k = min(max(k, 0), n - 1)
            self.conformal_delta = _np.sort(scores, axis=0)[k]   # (B,)
            self.conformal_delta = _np.maximum(self.conformal_delta, 0.0)
        return self

    def predict_proba(self, seq_list, ctx=None):
        seq_list = [to_device(s) for s in seq_list]
        seq_list = self._apply_scalers(seq_list)
        ctx_d = self._apply_ctx(ctx)
        return to_cpu(self._forward(seq_list, ctx_d, training=False)["P"])

    def predict_bands(self, seq_list, ctx=None):
        """Return conformally calibrated return quantile bands, shape (N, B, Q)."""
        seq_list = [to_device(s) for s in seq_list]
        seq_list = self._apply_scalers(seq_list)
        ctx_d = self._apply_ctx(ctx)
        q = to_cpu(self._forward(seq_list, ctx_d, training=False)["q"])
        if self.conformal_delta is not None:
            d = self.conformal_delta.reshape(1, self.B)
            q[:, :, 0] = q[:, :, 0] - d
            q[:, :, -1] = q[:, :, -1] + d
        return q

    def save(self, path):
        """Persist weights and metadata to a .npz checkpoint."""
        arrs = {f"param::{k}": to_cpu(v) for k, v in self.params.items()}
        cd = self.conformal_delta
        meta = {
            "meta::windows": _np.asarray(self.windows),
            "meta::quantiles": _np.asarray(self.quantiles),
            "meta::hidden": _np.asarray(self.H),
            "meta::trunk_sizes": _np.asarray(self.trunk_sizes),
            "meta::conformal_delta": (_np.asarray(cd) if cd is not None
                                      else _np.zeros(0)),
        }
        if self.scalers:
            meta["meta::scaler_mu"] = _np.stack([to_cpu(m) for m, _ in self.scalers])
            meta["meta::scaler_sd"] = _np.stack([to_cpu(s) for _, s in self.scalers])
        meta["meta::d_ctx"] = _np.asarray(self.d_ctx)
        if self.ctx_scaler is not None:
            meta["meta::ctx_mu"] = to_cpu(self.ctx_scaler[0])
            meta["meta::ctx_sd"] = to_cpu(self.ctx_scaler[1])
        _np.savez(path, **arrs, **meta)
        return path

    @classmethod
    def load(cls, path, **kw):
        """Reconstruct a trained network from a .npz checkpoint."""
        if not str(path).endswith(".npz"):
            path = str(path) + ".npz"
        d = _np.load(path, allow_pickle=True)
        net = cls(windows=d["meta::windows"].tolist(),
                  hidden=int(d["meta::hidden"]),
                  trunk_sizes=tuple(d["meta::trunk_sizes"].tolist()),
                  quantiles=d["meta::quantiles"].tolist(), **kw)
        net.params = {k[len("param::"):]: to_device(d[k])
                      for k in d.files if k.startswith("param::")}
        cd = d["meta::conformal_delta"]
        net.conformal_delta = cd if cd.size else None
        if "meta::scaler_mu" in d.files:
            mus = d["meta::scaler_mu"]; sds = d["meta::scaler_sd"]
            net.scalers = [(to_device(mus[i]), to_device(sds[i]))
                           for i in range(len(mus))]
        if "meta::d_ctx" in d.files:
            net.d_ctx = int(d["meta::d_ctx"])
        if "meta::ctx_mu" in d.files:
            net.ctx_scaler = (to_device(d["meta::ctx_mu"]),
                              to_device(d["meta::ctx_sd"]))
        return net

    def warm_start_from(self, path, d, verbose=True):
        """Initialize from a saved checkpoint, copying only tensors whose shapes
        match this model. The trunk, the heads, and the recurrent U and b
        transfer directly, while the input projection lstmW differs when the
        feature dimension differs and is left at its fresh initialization. The
        optimizer moments are reset so training starts clean from the transfer.
        """
        if not str(path).endswith(".npz"):
            path = str(path) + ".npz"
        if not self.params:
            self._init_weights(d)
        src = _np.load(path, allow_pickle=True)
        copied, reinit = [], []
        for k in list(self.params.keys()):
            dk = f"param::{k}"
            if dk in src.files and tuple(src[dk].shape) == tuple(self.params[k].shape):
                self.params[k] = to_device(src[dk]); copied.append(k)
            else:
                reinit.append(k)
        for k in self.params:
            self.m[k] = np.zeros_like(self.params[k])
            self.v[k] = np.zeros_like(self.params[k])
        self.t = 0
        if verbose:
            print(f"  [warm start] copied {len(copied)} tensors from {path}, "
                  f"reinitialized {len(reinit)}: {sorted(reinit)}", flush=True)
        return self

