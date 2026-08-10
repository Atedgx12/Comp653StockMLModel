"""
UnifiedCourseNetwork — modular, fine-tunable version.

Key additions over the monolithic pipeline_course.py version:
  - UCNConfig drives every hyperparameter
  - freeze_branch() / unfreeze_all() for branch-selective fine-tuning
  - Per-branch learning rates via config.branch_lrs
  - save_checkpoint() / load_checkpoint() for persisting trained weights
  - Fine-tune from a pretrained checkpoint in 3 lines:
        ucn = UnifiedCourseNetwork.from_checkpoint('weights.npz', cfg_finetune)
        ucn.freeze_branch('lr', 'nb', 'sent')   # keep LR, NB, Sent frozen
        ucn.fit(X_new, y_new)                    # only update MLP + meta
"""
import math
import numpy as _np
from typing import Optional
from ..backend import xp as np, to_device, to_cpu, new_rng
from ..config import UCNConfig
from ..utils import sigmoid, softmax, cross_entropy_softmax
from ..training.metrics import accuracy


class UnifiedCourseNetwork:
    """
    Four-branch parallel network:
      Branch A — Logistic Regression   (Lec 5-2)
      Branch B — Naive Bayes norm       (Lec 5-3)
      Branch C — Deep MLP               (Lec 5-5)
      Branch D — VADER Sentiment        (external)
      Meta-layer learns optimal branch weighting.

    Trained by Adam (Lec 6-5) with cosine LR annealing, gradient clipping,
    Gaussian noise augmentation, and PGD adversarial training (Module 8).
    """

    def __init__(self, cfg: Optional[UCNConfig] = None):
        self.cfg    = cfg or UCNConfig()
        self.params = {}
        self.m      = {}
        self.v      = {}
        self.t      = 0
        self.n_classes     = None
        self.loss_history  = []
        self.val_loss_history = []
        self._frozen: set  = set(self.cfg.frozen_branches)
        self._rng = new_rng(self.cfg.seed)
        # A CPU generator used only to shuffle the batch index order, because
        # the CuPy generator does not implement permutation.
        self._idx_rng = _np.random.default_rng(self.cfg.seed)
        self._groups = None   # hierarchy gate column-to-layer map, set on init

    # ── Branch freeze / unfreeze ─────────────────────────────────────────

    def freeze_branch(self, *branches: str) -> "UnifiedCourseNetwork":
        """Freeze one or more branches so their weights are not updated.
        Branch names: 'lr', 'nb', 'mlp', 'sent', 'meta'.
        """
        self._frozen.update(branches)
        return self

    def unfreeze_all(self) -> "UnifiedCourseNetwork":
        self._frozen.clear()
        return self

    def frozen_branches(self):
        return frozenset(self._frozen)

    # ── Weight initialisation ────────────────────────────────────────────

    def _init_weights(self, d: int, K: int):
        cfg  = self.cfg
        rng  = new_rng(cfg.seed)
        self.n_classes = K
        dp   = d - 1 if cfg.use_sent else d

        # Branch A
        self.params['W_lr'] = rng.standard_normal((dp, K)) * math.sqrt(1.0/dp)
        self.params['b_lr'] = np.zeros(K)

        # Branch B
        self.params['mu_nb']   = np.zeros(dp)
        self.params['lsig_nb'] = np.zeros(dp)
        self.params['W_nb']    = rng.standard_normal((dp, K)) * math.sqrt(1.0/dp)
        self.params['b_nb']    = np.zeros(K)

        # Branch C
        sizes = [dp, *list(cfg.hidden_sizes)]
        for i in range(len(sizes) - 1):
            s = math.sqrt(2.0 / sizes[i])
            self.params[f'Wm{i+1}'] = rng.standard_normal((sizes[i], sizes[i+1])) * s
            self.params[f'bm{i+1}'] = np.zeros(sizes[i+1])

        # Branch D
        if cfg.use_sent:
            self.params['W_sent'] = rng.standard_normal((1, K)) * 0.1
            self.params['b_sent'] = np.zeros(K)

        # Meta layer
        H_last  = cfg.hidden_sizes[-1]
        # Branch E: LSTM — processes the full feature vector (price + sentiment)
        if cfg.use_lstm:
            H_l   = cfg.lstm_hidden
            lstm_d = d   # LSTM sees all features including sentiment column
            s_l   = math.sqrt(2.0 / (lstm_d + H_l))
            self.params['lstm_W'] = rng.standard_normal((lstm_d,  4*H_l)) * s_l
            self.params['lstm_U'] = rng.standard_normal((H_l, 4*H_l)) * s_l
            self.params['lstm_b'] = np.zeros(4 * H_l)

        meta_in = K + K + H_last + (K if cfg.use_sent else 0) + (cfg.lstm_hidden if cfg.use_lstm else 0)
        self.params['W_meta'] = rng.standard_normal((meta_in, K)) * math.sqrt(2.0/meta_in)
        self.params['b_meta'] = np.zeros(K)

        # Learnable hierarchy gate: one scalar per layer, initialized to zero so
        # 2*sigmoid(0)=1 leaves every feature unscaled at the start of training.
        if cfg.use_hierarchy_gate and cfg.n_hierarchy_groups > 0:
            self.params['w_group'] = np.zeros(cfg.n_hierarchy_groups)
            self._groups = np.asarray(cfg.feature_groups)

        for key in self.params:
            self.m[key] = np.zeros_like(self.params[key])
            self.v[key] = np.zeros_like(self.params[key])

    # ── Forward pass ─────────────────────────────────────────────────────

    def _forward(self, X: _np.ndarray, training: bool = True,
                 seqs: _np.ndarray = None) -> dict:
        cfg = self.cfg
        if cfg.use_sent:
            X_price = X[:, :-1]
            X_sent  = X[:, -1:]
        else:
            X_price = X

        c = {'X': X_price}

        # Learnable hierarchy gate: scale each price feature by its layer gate
        # (2*sigmoid keeps the scale in (0, 2), init 1). The gate weights train
        # by backprop so the model learns how much each layer matters.
        if 'w_group' in self.params:
            scale_g    = 2.0 * sigmoid(self.params['w_group'])
            scale_cols = scale_g[self._groups]
            c['X_raw']      = X_price
            c['scale_cols'] = scale_cols
            c['scale_g']    = scale_g
            X_price = X_price * scale_cols
            c['X'] = X_price

        # Branch A
        z_lr = X_price @ self.params['W_lr'] + self.params['b_lr']
        a_lr = sigmoid(z_lr)
        c.update({'z_lr': z_lr, 'a_lr': a_lr})

        # Branch B
        sig  = np.exp(self.params['lsig_nb']) + 1e-8
        X_n  = (X_price - self.params['mu_nb']) / sig
        z_nb = X_n @ self.params['W_nb'] + self.params['b_nb']
        a_nb = sigmoid(z_nb)
        c.update({'sig': sig, 'X_n': X_n, 'z_nb': z_nb, 'a_nb': a_nb})

        # Branch C
        A = X_price; mlp = {'A0': X_price}
        p = cfg.dropout_rate
        for i in range(len(cfg.hidden_sizes)):
            Z = A @ self.params[f'Wm{i+1}'] + self.params[f'bm{i+1}']
            A = np.maximum(0, Z)
            if training and p > 0:
                mask = (self._rng.random(A.shape) >= p).astype(float) / (1.0 - p)
                A = A * mask
                mlp[f'drop{i+1}'] = mask
            mlp[f'Z{i+1}'] = Z; mlp[f'A{i+1}'] = A
        c['mlp'] = mlp

        # Branch D
        parts = [a_lr, a_nb, A]
        if cfg.use_sent:
            z_s = X_sent @ self.params['W_sent'] + self.params['b_sent']
            a_s = sigmoid(z_s)
            c.update({'X_sent': X_sent, 'z_sent': z_s, 'a_sent': a_s})
            parts.append(a_s)

        # Branch E: LSTM — processes a lookback window of past feature vectors.
        # seqs shape: (N, T, dp)  h_T shape: (N, lstm_hidden)
        if cfg.use_lstm and seqs is not None:
            W_l, U_l, b_l = (self.params['lstm_W'],
                             self.params['lstm_U'],
                             self.params['lstm_b'])
            H_l    = cfg.lstm_hidden
            N_s, T = seqs.shape[0], seqs.shape[1]
            lstm_h = np.zeros((N_s, T + 1, H_l))
            lstm_c = np.zeros((N_s, T + 1, H_l))
            lstm_g = np.zeros((N_s, T, 4 * H_l))
            for t in range(T):
                z_t  = seqs[:, t, :] @ W_l + lstm_h[:, t, :] @ U_l + b_l
                lstm_g[:, t, :] = z_t
                f_t  = sigmoid(z_t[:,    :H_l])
                i_t  = sigmoid(z_t[:,   H_l:2*H_l])
                gt_t = np.tanh(z_t[:, 2*H_l:3*H_l])
                o_t  = sigmoid(z_t[:, 3*H_l:])
                lstm_c[:, t+1, :] = f_t * lstm_c[:, t, :] + i_t * gt_t
                lstm_h[:, t+1, :] = o_t * np.tanh(lstm_c[:, t+1, :])
            h_T = lstm_h[:, T, :]
            c.update({'lstm_h': lstm_h, 'lstm_c': lstm_c,
                      'lstm_g': lstm_g, 'seqs': seqs, 'h_T': h_T})
            parts.append(h_T)

        # Meta layer
        cat = np.hstack(parts)
        if training and cfg.meta_dropout > 0:
            md  = (self._rng.random(cat.shape) >= cfg.meta_dropout).astype(float) \
                  / (1.0 - cfg.meta_dropout)
            cat = cat * md
            c['meta_drop'] = md
        z_meta = cat @ self.params['W_meta'] + self.params['b_meta']
        Y_hat  = softmax(z_meta)
        c.update({'cat': cat, 'Y_hat': Y_hat})
        return c

    # ── Backward pass ────────────────────────────────────────────────────

    def _backward(self, c: dict, Y_oh: _np.ndarray,
                  sample_weights: _np.ndarray = None):
        cfg = self.cfg
        X   = c['X']; N = Y_oh.shape[0]; K = self.n_classes
        g   = {}

        # Softmax + CE gradient: (Y_hat - Y_oh) / N
        # If sample_weights provided, scale each row by its weight.
        # This implements weighted cross-entropy without touching the labels.
        d = (c['Y_hat'] - Y_oh) / N
        if sample_weights is not None:
            d = d * sample_weights[:, None]

        # Meta layer
        g['W_meta'] = c['cat'].T @ d
        g['b_meta'] = d.sum(0)
        dc = d @ self.params['W_meta'].T
        if 'meta_drop' in c:
            dc = dc * c['meta_drop']
        H      = cfg.hidden_sizes[-1]
        d_lr   = dc[:, :K]
        d_nb   = dc[:, K:2*K]
        d_mlp  = dc[:, 2*K:2*K+H]

        # Branch A
        dz_lr     = d_lr * c['a_lr'] * (1 - c['a_lr'])
        g['W_lr']  = X.T @ dz_lr
        g['b_lr']  = dz_lr.sum(0)

        # Branch B
        dz_nb         = d_nb * c['a_nb'] * (1 - c['a_nb'])
        g['W_nb']     = c['X_n'].T @ dz_nb
        g['b_nb']     = dz_nb.sum(0)
        dX_n          = dz_nb @ self.params['W_nb'].T
        g['mu_nb']    = (-dX_n / c['sig']).sum(0)
        g['lsig_nb']  = (-dX_n * c['X_n']).sum(0)

        # Branch D — sentiment occupies exactly K columns after the MLP block
        if cfg.use_sent:
            d_sent  = dc[:, 2*K+H:2*K+H+K]
            dz_sent = d_sent * c['a_sent'] * (1 - c['a_sent'])
            g['W_sent'] = c['X_sent'].T @ dz_sent
            g['b_sent'] = dz_sent.sum(0)

        # Branch E: LSTM BPTT
        if cfg.use_lstm and 'lstm_h' in c:
            H_l    = cfg.lstm_hidden
            lstm_h = c['lstm_h']; lstm_c = c['lstm_c']
            lstm_g = c['lstm_g']; seqs   = c['seqs']
            T      = seqs.shape[1]
            # gradient from meta-layer: last columns after sent
            sent_k = K if cfg.use_sent else 0
            d_hT   = dc[:, 2*K+H+sent_k:]   # (N, lstm_hidden)

            dW_l = np.zeros_like(self.params['lstm_W'])
            dU_l = np.zeros_like(self.params['lstm_U'])
            db_l = np.zeros_like(self.params['lstm_b'])

            d_h_next = d_hT.copy()
            d_c_next = np.zeros_like(d_hT)

            for t in reversed(range(T)):
                z    = lstm_g[:, t, :]
                f_t  = sigmoid(z[:,    :H_l])
                i_t  = sigmoid(z[:,   H_l:2*H_l])
                gt_t = np.tanh(z[:, 2*H_l:3*H_l])
                o_t  = sigmoid(z[:, 3*H_l:])
                tanh_c = np.tanh(lstm_c[:, t+1, :])

                d_o  = d_h_next * tanh_c
                d_c  = d_h_next * o_t * (1.0 - tanh_c**2) + d_c_next
                d_f  = d_c * lstm_c[:, t, :]
                d_i  = d_c * gt_t
                d_g  = d_c * i_t
                d_cp = d_c * f_t

                dz_f = d_f * f_t  * (1 - f_t)
                dz_i = d_i * i_t  * (1 - i_t)
                dz_g = d_g * (1 - gt_t**2)
                dz_o = d_o * o_t  * (1 - o_t)
                dz   = np.concatenate([dz_f, dz_i, dz_g, dz_o], axis=1)

                dW_l       += seqs[:, t, :].T @ dz
                dU_l       += lstm_h[:, t, :].T @ dz
                db_l       += dz.sum(0)
                d_h_next    = dz @ self.params['lstm_U'].T
                d_c_next    = d_cp

            g['lstm_W'] = dW_l
            g['lstm_U'] = dU_l
            g['lstm_b'] = db_l

        # Branch C
        dm = d_mlp; mlp = c['mlp']
        for i in range(len(cfg.hidden_sizes), 0, -1):
            if f'drop{i}' in mlp:
                dm = dm * mlp[f'drop{i}']
            dm = dm * (mlp[f'Z{i}'] > 0).astype(float)
            g[f'Wm{i}']  = mlp[f'A{i-1}'].T @ dm
            g[f'bm{i}']  = dm.sum(0)
            if i > 1:
                dm = dm @ self.params[f'Wm{i}'].T

        # Input gradient for PGD
        dX = (dz_lr @ self.params['W_lr'].T
              + dX_n / c['sig']
              + dm @ self.params['Wm1'].T)

        # Learnable hierarchy gate gradient. dX is the gradient with respect to
        # the gated price features, so I chain it back through the per column
        # scale and sum into one gradient per hierarchy layer.
        if 'w_group' in self.params:
            d_scale_cols = (dX * c['X_raw']).sum(0)
            scale_g = c['scale_g']
            n_g = self.params['w_group'].shape[0]
            d_scale_g = np.zeros(n_g)
            for gi in range(n_g):
                d_scale_g[gi] = d_scale_cols[self._groups == gi].sum()
            # d(2*sigmoid(w))/dw = scale_g * (1 - scale_g/2)
            g['w_group'] = d_scale_g * scale_g * (1.0 - scale_g / 2.0)

        return g, dX

    # ── Adam update with per-branch LR and frozen param support ──────────

    def _update(self, g: dict, lr: float):
        cfg = self.cfg

        # Gradient clipping
        if cfg.grad_clip > 0:
            total_norm = np.sqrt(sum(np.sum(v**2) for v in g.values()))
            if total_norm > cfg.grad_clip:
                scale = cfg.grad_clip / (total_norm + 1e-8)
                g = {k: v * scale for k, v in g.items()}

        self.t += 1
        eps = 1e-8
        b1c = 1.0 - cfg.beta1 ** self.t
        b2c = 1.0 - cfg.beta2 ** self.t

        for k, val in self.params.items():
            if cfg.is_frozen(k) or k in {f for f in self._frozen
                                          for _ in [None]
                                          if cfg.param_belongs_to(k, f)}:
                continue  # skip frozen parameters

            # Resolve effective LR for this parameter
            effective_lr = cfg.lr_for(k) if cfg.branch_lrs else lr

            grad = g[k]
            if k.startswith('W'):
                grad = grad + cfg.lam * val
            self.m[k] = cfg.beta1 * self.m[k] + (1 - cfg.beta1) * grad
            self.v[k] = cfg.beta2 * self.v[k] + (1 - cfg.beta2) * grad**2
            m_hat = self.m[k] / b1c
            v_hat = self.v[k] / b2c
            self.params[k] = val - effective_lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── Training loop ────────────────────────────────────────────────────

    def fit(self, X: _np.ndarray, y: _np.ndarray,
            sample_weights: _np.ndarray = None,
            seqs: _np.ndarray = None) -> "UnifiedCourseNetwork":
        """seqs : (N, T, d) optional LSTM sequence input aligned with X."""
        cfg = self.cfg
        # Move all inputs onto the compute device once, so every matmul below
        # runs on the GPU when the CuPy backend is active.
        X = to_device(X); y = to_device(y)
        seqs = to_device(seqs)
        sample_weights = to_device(sample_weights)
        K   = int(len(np.unique(y)))
        if not self.params:
            self._init_weights(X.shape[1], K)

        n_val  = max(int(len(X) * cfg.val_frac), 1)
        X_tr   = X[:len(X)-n_val];  y_tr = y[:len(y)-n_val]
        X_val  = X[len(X)-n_val:];  y_val = y[len(y)-n_val:]
        # Slice sample weights to match training split
        w_tr = (sample_weights[:len(X)-n_val]
                if sample_weights is not None else None)
        # Slice sequence tensor to match training split
        seqs_tr  = seqs[:len(X)-n_val]  if seqs  is not None else None
        seqs_val = seqs[len(X)-n_val:]  if seqs  is not None else None
        idx_tr = _np.arange(len(X_tr))

        noise_scale = (X_tr.std(axis=0) * cfg.noise_frac
                       if cfg.noise_frac > 0 else None)

        best_val    = 0.0   # track best val_acc (maximise, not minimise)
        best_params = None
        no_improve  = 0

        if cfg.verbose:
            frozen_str = f"  frozen={list(self._frozen)}" if self._frozen else ""
            print(f"  UCN fit: {len(X_tr):,} train  {n_val:,} val  "
                  f"epochs={cfg.epochs}  patience={cfg.patience}"
                  f"{frozen_str}", flush=True)

        for epoch in range(cfg.epochs):
            if epoch < cfg.warmup_epochs:
                lr_t = cfg.lr * (epoch + 1) / max(cfg.warmup_epochs, 1)
            else:
                ce   = epoch - cfg.warmup_epochs
                ct   = max(cfg.epochs - cfg.warmup_epochs, 1)
                lr_t = cfg.lr * (0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * ce / ct)))

            self._idx_rng.shuffle(idx_tr)
            ep_loss = 0.0; n_b = 0

            for s in range(0, len(X_tr), cfg.batch_size):
                b    = to_device(idx_tr[s:s + cfg.batch_size])
                X_b  = (X_tr[b] + self._rng.standard_normal(X_tr[b].shape) * noise_scale
                        if noise_scale is not None else X_tr[b])
                s_b  = seqs_tr[b] if seqs_tr is not None else None
                Y_oh = np.eye(K)[y_tr[b].astype(int)]
                # Sample weights are applied to the gradient delta AFTER
                # the softmax-CE shortcut, not to the target labels.
                # Multiplying Y_oh by weights would push targets > 1
                # and cause numerical explosion.
                w_b = w_tr[b] if w_tr is not None else None

                if cfg.use_fgsm and cfg.pgd_steps > 0:
                    step  = cfg.fgsm_eps / max(cfg.pgd_steps, 1)
                    X_adv = X_b.copy()
                    for _ in range(cfg.pgd_steps):
                        c_tmp  = self._forward(X_adv, training=False, seqs=s_b)
                        _, dX_s = self._backward(c_tmp, Y_oh)
                        dX_full = (np.hstack([dX_s, np.zeros((len(b), 1))])
                                   if cfg.use_sent else dX_s)
                        X_adv = X_adv + step * np.sign(dX_full)
                    c_adv = self._forward(X_adv, training=True, seqs=s_b)
                    g, _  = self._backward(c_adv, Y_oh, sample_weights=w_b)
                else:
                    c    = self._forward(X_b, training=True, seqs=s_b)
                    g, _ = self._backward(c, Y_oh, sample_weights=w_b)

                # Unweighted CE for monitoring so values stay interpretable
                loss = cross_entropy_softmax(
                    self._forward(X_b, training=False, seqs=s_b)['Y_hat'], Y_oh)
                self._update(g, lr_t)
                ep_loss += float(loss); n_b += 1

            self.loss_history.append(ep_loss / n_b)

            Y_oh_val     = np.eye(K)[y_val.astype(int)]
            c_val        = self._forward(X_val, training=False, seqs=seqs_val)
            val_ce       = float(cross_entropy_softmax(c_val['Y_hat'], Y_oh_val))
            val_acc_curr = accuracy(to_cpu(y_val),
                                    _np.argmax(to_cpu(c_val['Y_hat']), axis=1))
            self.val_loss_history.append(val_ce)

            # Early stopping tracks val_acc (not val_CE).
            # For financial data the CE diverges due to calibration drift even
            # when accuracy is still improving; using val_acc gives a more
            # stable stopping criterion across regime shifts.
            if val_acc_curr > best_val + 1e-5:
                best_val    = val_acc_curr
                best_params = {k: v.copy() for k, v in self.params.items()}
                no_improve  = 0
            else:
                no_improve += 1

            if cfg.verbose and (epoch + 1) % cfg.verbose == 0:
                acc_tr  = accuracy(to_cpu(y_tr), self.predict(X_tr, seqs=seqs_tr))
                marker  = " *" if no_improve == 0 else ""
                print(f"  Epoch {epoch+1:4d}/{cfg.epochs}  "
                      f"CE={ep_loss/n_b:.5f}  acc={acc_tr:.4f}  "
                      f"val_CE={val_ce:.5f}  val_acc={val_acc_curr:.4f}  "
                      f"LR={lr_t:.2e}{marker}", flush=True)

            if cfg.patience and no_improve >= cfg.patience:
                if cfg.verbose:
                    print(f"  Early stop epoch {epoch+1}", flush=True)
                break

        if best_params is not None:
            self.params = best_params
            if cfg.verbose:
                print(f"  Restored best checkpoint (val_acc={best_val:.5f})",
                      flush=True)
        return self

    def hierarchy_gate_report(self, group_names) -> None:
        """Print the learned per-layer hierarchy gate values.

        The gate scale is 2*sigmoid(w), so 1.0 means the layer is left as is,
        above 1.0 means the model amplified that layer, and below 1.0 means it
        attenuated it.  These are trained by backprop, so they read out how much
        the model chose to rely on each hierarchy layer.
        """
        if 'w_group' not in self.params:
            return
        w = to_cpu(self.params['w_group'])
        scale = 2.0 / (1.0 + _np.exp(-w))
        print("\n=== Learned hierarchy gate (2*sigmoid(w), 1.0 = neutral) ===")
        print(f"{'Layer':<20} {'Gate':>8}")
        print("-" * 30)
        order = _np.argsort(-scale)
        for i in order:
            name = group_names[i] if i < len(group_names) else f"group{i}"
            print(f"{name:<20} {float(scale[i]):>8.3f}")

    def predict_proba(self, X: _np.ndarray,
                      seqs: _np.ndarray = None) -> _np.ndarray:
        out = self._forward(to_device(X), training=False,
                            seqs=to_device(seqs))['Y_hat']
        return to_cpu(out)

    def predict(self, X: _np.ndarray,
                seqs: _np.ndarray = None) -> _np.ndarray:
        return _np.argmax(self.predict_proba(X, seqs=seqs), axis=1)

    # ── Checkpoint persistence ───────────────────────────────────────────

    def save_checkpoint(self, path: str):
        """Save trained weights to a .npz file."""
        _np.savez_compressed(path, **{k: to_cpu(v)
                                      for k, v in self.params.items()})
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load weights from a .npz file into this model."""
        data = _np.load(path)
        self.params = {k: to_device(data[k]) for k in data.files}
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        # Restore the hierarchy gate column-to-layer map if the gate is present.
        if 'w_group' in self.params and self.cfg.feature_groups:
            self._groups = _np.asarray(self.cfg.feature_groups)
        # Infer n_classes from the meta-layer bias shape
        if 'b_meta' in self.params:
            self.n_classes = int(self.params['b_meta'].shape[0])
        print(f"Checkpoint loaded: {path}  ({len(self.params)} tensors  K={self.n_classes})")
        return self

    @classmethod
    def from_checkpoint(cls, path: str,
                        cfg: Optional[UCNConfig] = None) -> "UnifiedCourseNetwork":
        """Instantiate a model from a saved checkpoint (for fine-tuning)."""
        model = cls(cfg)
        model.load_checkpoint(path)
        return model

    # ── Branch weight inspection and assignment ──────────────────────────

    def get_branch_weights(self, branch: str) -> dict:
        """Return a copy of all parameter tensors belonging to a branch.

        branch: 'lr' | 'nb' | 'mlp' | 'sent' | 'meta'
        """
        return {k: v.copy() for k, v in self.params.items()
                if self.cfg.param_belongs_to(k, branch)}

    def set_branch_weights(self, branch: str, weights: dict):
        """Overwrite parameter tensors for a branch.

        Useful for:
          - Manually transferring weights from another model
          - Setting specific weight values for analysis
          - Resetting a branch to its initialised state

        weights: dict of {param_key: np.ndarray}  (must match current shapes)
        """
        for k, v in weights.items():
            if k not in self.params:
                raise KeyError(f"Unknown param key: {k!r}")
            if self.params[k].shape != v.shape:
                raise ValueError(
                    f"Shape mismatch for {k!r}: "
                    f"model={self.params[k].shape}  given={v.shape}")
            self.params[k] = v.copy()
            # Reset Adam moments for this param so LR warmup applies cleanly
            self.m[k] = np.zeros_like(v)
            self.v[k] = np.zeros_like(v)

    def branch_weight_norms(self) -> dict:
        """Return L2 norm of each branch's weight matrices.
        Useful for monitoring which branches have large / small weights.
        """
        norms = {}
        for branch in ("lr", "nb", "mlp", "sent", "meta"):
            w = self.get_branch_weights(branch)
            # Only weight matrices (not biases) for a meaningful norm
            mats  = {k: v for k, v in w.items() if k.startswith("W") or k.startswith("Wm")}
            if mats:
                total = float(np.sqrt(sum(np.sum(v**2) for v in mats.values())))
                norms[branch] = total
        return norms

    def branch_summary(self):
        """Print a human-readable summary of each branch: param count,
        weight norm, frozen status, and effective learning rate.
        """
        cfg   = self.cfg
        lines = []
        lines.append(f"\n{'Branch':<8} {'Params':>8} {'W-norm':>9} "
                     f"{'Frozen':>7} {'Eff-LR':>10}  Keys")
        lines.append("-" * 72)
        norms = self.branch_weight_norms()
        for branch in ("lr", "nb", "mlp", "sent", "meta"):
            bw     = self.get_branch_weights(branch)
            n_p    = sum(v.size for v in bw.values())
            frozen = branch in self._frozen
            e_lr   = 0.0 if frozen else cfg.lr_for(
                next(iter(bw), ""))
            norm   = norms.get(branch, 0.0)
            keys   = ", ".join(sorted(bw.keys()))
            lines.append(f"{branch:<8} {n_p:>8,} {norm:>9.4f} "
                         f"{'yes' if frozen else 'no':>7} {e_lr:>10.2e}  {keys}")
        print("\n".join(lines))
        print(f"\nTotal trainable params: "
              f"{sum(v.size for k, v in self.params.items() if not cfg.is_frozen(k)):,}")
        print(f"Total frozen    params: "
              f"{sum(v.size for k, v in self.params.items() if cfg.is_frozen(k)):,}")
