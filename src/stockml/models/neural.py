"""Neural network models for the stockml package.

Contains two families:

1. **UnifiedCourseNetwork** — pure-NumPy 4-branch network (LR + NB + MLP +
   optional Sentiment) trained end-to-end with Adam and cosine LR annealing.
   No external dependencies beyond NumPy.  Implements the COMP 653 Module 5
   algorithm ensemble with a learned meta-layer.

2. **Torch sequence models** (TCN, Transformer, LSTM) — gated behind the
   optional ``torch`` extra.  Calling any factory in this group without
   PyTorch installed raises an informative error.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import BaseModel

# ---------------------------------------------------------------------------
# Shared math primitives (no external deps)
# ---------------------------------------------------------------------------

def _softmax(Z: np.ndarray) -> np.ndarray:
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def _sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))


def _cross_entropy(Y_hat: np.ndarray, Y_oh: np.ndarray) -> float:
    return -np.mean(np.sum(Y_oh * np.log(np.clip(Y_hat, 1e-12, 1.0)), axis=1))


# ---------------------------------------------------------------------------
# UnifiedCourseNetwork — COMP 653 Module 5 ensemble
# ---------------------------------------------------------------------------

class UnifiedCourseNetwork(BaseModel):
    """4-branch network trained jointly via backprop + Adam (COMP 653).

    Architecture
    ------------
    Input x  (d MI-selected cross-sectional rank features)
          |
    +-----+----------------------------+-------------------+------------+
    | Branch A  LR (Lec 5-2)          | Branch B  NB      | Branch C   |
    | Linear(d->K) -> Sigmoid          | Gaussian norm     | MLP        |
    |                                  | -> Linear -> Sig  | ReLU stack |
    +-----+----------------------------+-------------------+-----+------+
          |                                                       |
          +------ concat(a_lr, a_nb, a_mlp [, a_sent]) ----------+
          |
    Meta-layer: Linear -> Softmax   (with optional meta dropout)
          |
    P(Up), P(Down)

    Branch D (optional) processes a single VADER sentiment score through
    Linear(1->K)->Sigmoid and appends its output to the meta-layer input.

    Parameters
    ----------
    hidden_sizes : tuple of int
        Hidden-layer widths for Branch C MLP.
    lr : float
        Initial Adam learning rate.  Decayed via cosine annealing.
    epochs : int
        Total training epochs (no hard early stop; best checkpoint restored).
    lam : float
        L2 weight-decay coefficient applied to all W matrices.
    batch_size : int
        Mini-batch size.
    beta1, beta2 : float
        Adam moment decay rates.
    dropout_rate : float
        Inverted dropout probability for MLP branch activations.
    meta_dropout : float
        Inverted dropout probability applied to the concatenated meta-layer
        input.  Forces the meta-layer to learn robust cross-branch weights.
    val_frac : float
        Fraction of (time-ordered) rows held out for validation tracking.
    verbose : int
        Print progress every ``verbose`` epochs.  0 = silent.
    seed : int
        RNG seed for reproducibility.
    use_sent : bool
        If True, the last column of X is treated as the VADER sentiment score
        (Branch D); all other columns are price features.
    """

    name = "unified_course_network"

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128, 64),
        lr: float = 0.001,
        epochs: int = 200,
        lam: float = 1e-4,
        batch_size: int = 2048,
        beta1: float = 0.9,
        beta2: float = 0.999,
        dropout_rate: float = 0.4,
        meta_dropout: float = 0.2,
        val_frac: float = 0.15,
        verbose: int = 20,
        seed: int = 42,
        use_sent: bool = False,
    ) -> None:
        self.hidden_sizes  = hidden_sizes
        self.lr            = lr
        self.epochs        = epochs
        self.lam           = lam
        self.batch_size    = batch_size
        self.beta1         = beta1
        self.beta2         = beta2
        self.dropout_rate  = dropout_rate
        self.meta_dropout  = meta_dropout
        self.val_frac      = val_frac
        self.verbose       = verbose
        self.seed          = seed
        self.use_sent      = use_sent

        self._params: dict[str, np.ndarray] = {}
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._t: int = 0
        self._n_classes: int | None = None
        self._rng = np.random.default_rng(seed)

        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, d: int, K: int) -> None:
        rng_ = np.random.default_rng(self.seed)
        self._n_classes = K
        self._rng = np.random.default_rng(self.seed + 1)  # separate rng for dropout

        dp = d - 1 if self.use_sent else d  # price feature count

        # Branch A: Logistic Regression  dp -> K
        self._params["W_lr"] = rng_.standard_normal((dp, K)) * math.sqrt(1.0 / dp)
        self._params["b_lr"] = np.zeros(K)

        # Branch B: NB learnable Gaussian normalisation  dp -> K
        self._params["mu_nb"]   = np.zeros(dp)
        self._params["lsig_nb"] = np.zeros(dp)
        self._params["W_nb"]    = rng_.standard_normal((dp, K)) * math.sqrt(1.0 / dp)
        self._params["b_nb"]    = np.zeros(K)

        # Branch C: MLP hidden layers  dp -> h1 -> h2 -> ...
        sizes = [dp, *list(self.hidden_sizes)]
        for i in range(len(sizes) - 1):
            s = math.sqrt(2.0 / sizes[i])
            self._params[f"Wm{i+1}"] = rng_.standard_normal((sizes[i], sizes[i+1])) * s
            self._params[f"bm{i+1}"] = np.zeros(sizes[i+1])

        # Branch D: Sentiment  1 -> K
        if self.use_sent:
            self._params["W_sent"] = rng_.standard_normal((1, K)) * 0.1
            self._params["b_sent"] = np.zeros(K)

        # Meta-layer
        H_last  = self.hidden_sizes[-1]
        meta_in = K + K + H_last + (K if self.use_sent else 0)
        self._params["W_meta"] = rng_.standard_normal((meta_in, K)) * math.sqrt(2.0 / meta_in)
        self._params["b_meta"] = np.zeros(K)

        for key in self._params:
            self._m[key] = np.zeros_like(self._params[key])
            self._v[key] = np.zeros_like(self._params[key])

    # ------------------------------------------------------------------
    # Forward / backward / update
    # ------------------------------------------------------------------

    @staticmethod
    def _relu(Z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, Z)

    @staticmethod
    def _relu_g(Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)

    def _forward(self, X: np.ndarray, training: bool = True) -> dict:
        if self.use_sent:
            X_price, X_sent = X[:, :-1], X[:, -1:]
        else:
            X_price = X

        c: dict = {"X": X_price}

        # Branch A
        z_lr = X_price @ self._params["W_lr"] + self._params["b_lr"]
        a_lr = _sigmoid(z_lr)
        c.update({"z_lr": z_lr, "a_lr": a_lr})

        # Branch B
        sig  = np.exp(self._params["lsig_nb"]) + 1e-8
        X_n  = (X_price - self._params["mu_nb"]) / sig
        z_nb = X_n @ self._params["W_nb"] + self._params["b_nb"]
        a_nb = _sigmoid(z_nb)
        c.update({"sig": sig, "X_n": X_n, "z_nb": z_nb, "a_nb": a_nb})

        # Branch C
        A = X_price
        mlp: dict = {"A0": X_price}
        p = self.dropout_rate
        for i in range(len(self.hidden_sizes)):
            Z = A @ self._params[f"Wm{i+1}"] + self._params[f"bm{i+1}"]
            A = self._relu(Z)
            if training and p > 0:
                mask = (self._rng.random(A.shape) >= p).astype(float) / (1.0 - p)
                A    = A * mask
                mlp[f"drop{i+1}"] = mask
            mlp[f"Z{i+1}"] = Z
            mlp[f"A{i+1}"] = A
        c["mlp"] = mlp

        # Branch D (optional)
        parts = [a_lr, a_nb, A]
        if self.use_sent:
            z_sent = X_sent @ self._params["W_sent"] + self._params["b_sent"]
            a_sent = _sigmoid(z_sent)
            c.update({"X_sent": X_sent, "z_sent": z_sent, "a_sent": a_sent})
            parts.append(a_sent)

        # Meta-layer with optional dropout
        cat = np.hstack(parts)
        if training and self.meta_dropout > 0:
            md  = (self._rng.random(cat.shape) >= self.meta_dropout).astype(float) \
                  / (1.0 - self.meta_dropout)
            cat = cat * md
            c["meta_drop"] = md
        z_meta = cat @ self._params["W_meta"] + self._params["b_meta"]
        c.update({"cat": cat, "Y_hat": _softmax(z_meta)})
        return c

    def _backward(self, c: dict, Y_oh: np.ndarray) -> dict:
        X = c["X"]
        N = Y_oh.shape[0]
        K = self._n_classes
        g: dict = {}

        d = (c["Y_hat"] - Y_oh) / N

        g["W_meta"] = c["cat"].T @ d
        g["b_meta"] = d.sum(0)
        dc = d @ self._params["W_meta"].T
        if "meta_drop" in c:
            dc = dc * c["meta_drop"]

        H     = self.hidden_sizes[-1]
        d_lr  = dc[:, :K]
        d_nb  = dc[:, K:2*K]
        d_mlp = dc[:, 2*K:2*K+H]

        # Branch A
        dz_lr      = d_lr * c["a_lr"] * (1 - c["a_lr"])
        g["W_lr"]  = X.T @ dz_lr
        g["b_lr"]  = dz_lr.sum(0)

        # Branch B
        dz_nb        = d_nb * c["a_nb"] * (1 - c["a_nb"])
        g["W_nb"]    = c["X_n"].T @ dz_nb
        g["b_nb"]    = dz_nb.sum(0)
        dX_n         = dz_nb @ self._params["W_nb"].T
        g["mu_nb"]   = (-dX_n / c["sig"]).sum(0)
        g["lsig_nb"] = (-dX_n * c["X_n"]).sum(0)

        # Branch D
        if self.use_sent:
            d_sent  = dc[:, 2*K+H:]
            dz_sent = d_sent * c["a_sent"] * (1 - c["a_sent"])
            g["W_sent"] = c["X_sent"].T @ dz_sent
            g["b_sent"] = dz_sent.sum(0)

        # Branch C
        dm  = d_mlp
        mlp = c["mlp"]
        for i in range(len(self.hidden_sizes), 0, -1):
            if f"drop{i}" in mlp:
                dm = dm * mlp[f"drop{i}"]
            dm             = dm * self._relu_g(mlp[f"Z{i}"])
            g[f"Wm{i}"]   = mlp[f"A{i-1}"].T @ dm
            g[f"bm{i}"]   = dm.sum(0)
            if i > 1:
                dm = dm @ self._params[f"Wm{i}"].T
        return g

    def _adam_update(self, g: dict, lr: float) -> None:
        self._t += 1
        eps = 1e-8
        b1c = 1.0 - self.beta1 ** self._t
        b2c = 1.0 - self.beta2 ** self._t
        for k, val in self._params.items():
            grad = g[k]
            if k.startswith("W"):
                grad = grad + self.lam * val
            self._m[k] = self.beta1 * self._m[k] + (1 - self.beta1) * grad
            self._v[k] = self.beta2 * self._v[k] + (1 - self.beta2) * grad ** 2
            m_hat = self._m[k] / b1c
            v_hat = self._v[k] / b2c
            self._params[k] = val - lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> UnifiedCourseNetwork:
        K = len(np.unique(y_train))
        self._init_weights(X_train.shape[1], K)
        self.loss_history.clear()
        self.val_loss_history.clear()

        # Use caller-supplied validation set or split temporally
        if X_val is not None and y_val is not None:
            X_tr, y_tr = X_train, y_train
        else:
            n_val = max(int(len(X_train) * self.val_frac), 1)
            X_tr, y_tr  = X_train[:-n_val], y_train[:-n_val]
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]

        idx_tr     = np.arange(len(X_tr))
        best_val   = np.inf
        best_state: dict | None = None

        for epoch in range(self.epochs):
            lr_t = self.lr * (0.01 + 0.99 * 0.5 *
                              (1.0 + math.cos(math.pi * epoch / self.epochs)))
            self._rng.shuffle(idx_tr)
            ep_loss = 0.0
            n_b = 0
            for s in range(0, len(X_tr), self.batch_size):
                b    = idx_tr[s:s + self.batch_size]
                Y_oh = np.eye(K)[y_tr[b].astype(int)]
                c    = self._forward(X_tr[b], training=True)
                loss = _cross_entropy(c["Y_hat"], Y_oh)
                self._adam_update(self._backward(c, Y_oh), lr_t)
                ep_loss += loss
                n_b += 1
            self.loss_history.append(ep_loss / n_b)

            Y_oh_val = np.eye(K)[y_val.astype(int)]
            c_val    = self._forward(X_val, training=False)
            val_ce   = _cross_entropy(c_val["Y_hat"], Y_oh_val)
            self.val_loss_history.append(val_ce)

            if val_ce < best_val - 1e-6:
                best_val  = val_ce
                best_state = {k: v.copy() for k, v in self._params.items()}

            if self.verbose and (epoch + 1) % self.verbose == 0:
                val_acc = (y_val.astype(int) ==
                           np.argmax(c_val["Y_hat"], axis=1)).mean()
                marker  = " *" if self._params is best_state else ""
                print(
                    f"  Epoch {epoch+1:4d}/{self.epochs}  "
                    f"CE={self.loss_history[-1]:.5f}  "
                    f"val_CE={val_ce:.5f}  val_acc={val_acc:.4f}  "
                    f"LR={lr_t:.2e}{marker}",
                    flush=True,
                )

        if best_state is not None:
            self._params = best_state
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X, training=False)["Y_hat"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ===========================================================================
# Torch sequence models (optional dependency)
# ===========================================================================

try:  # pragma: no cover - depends on optional dependency
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for sequence models. Install with `pip install -e .[torch]`."
        )


class _TorchSequenceModel(BaseModel):
    """Common skeleton: store config, expose fit/predict, keep weights in memory."""

    name = "torch_sequence"

    def __init__(self, params: dict[str, Any], task: str) -> None:
        _require_torch()
        self.params = dict(params)
        self.task = task
        self.input_window = int(self.params.get("input_window", 64))
        self.epochs = int(self.params.get("epochs", 1))
        self.batch_size = int(self.params.get("batch_size", 64))
        self.learning_rate = float(self.params.get("learning_rate", 1e-3))
        self.weight_decay = float(self.params.get("weight_decay", 0.0))
        self.module: nn.Module | None = None
        self._device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    def _build_module(self, input_features: int, output_dim: int) -> nn.Module:
        raise NotImplementedError

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        _require_torch()
        if X_train.ndim != 3:
            raise ValueError(
                "Sequence models expect X with shape (n_samples, window, n_features)"
            )
        n_samples, _window, n_features = X_train.shape
        output_dim = 1 if y_train.ndim == 1 else y_train.shape[1]
        self.module = self._build_module(n_features, output_dim).to(self._device)
        opt = torch.optim.AdamW(
            self.module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss() if self.task != "classification" else nn.CrossEntropyLoss()
        x = torch.tensor(X_train, dtype=torch.float32, device=self._device)
        y_dtype = torch.float32 if self.task != "classification" else torch.long
        y = torch.tensor(y_train, dtype=y_dtype, device=self._device)
        if self.task != "classification" and y.ndim == 1:
            y = y.unsqueeze(-1)
        self.module.train()
        for _epoch in range(self.epochs):
            perm = torch.randperm(n_samples, device=self._device)
            for i in range(0, n_samples, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                out = self.module(x[idx])
                loss = loss_fn(out, y[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad() if _HAS_TORCH else (lambda f: f)
    def predict(self, X):
        _require_torch()
        if self.module is None:
            raise RuntimeError("Model has not been fit")
        self.module.eval()
        x = torch.tensor(X, dtype=torch.float32, device=self._device)
        out = self.module(x)
        return out.detach().cpu().numpy()


class TCNRegressor(_TorchSequenceModel):
    """Plain dilated temporal CNN for sequence regression."""

    name = "tcn"

    def _build_module(self, input_features: int, output_dim: int) -> nn.Module:
        channels = self.params.get("channels", [64, 64, 64, 64])
        kernel_size = int(self.params.get("kernel_size", 3))
        dropout = float(self.params.get("dropout", 0.1))
        layers: list[nn.Module] = []
        in_ch = input_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        body = nn.Sequential(*layers)
        head = nn.Linear(channels[-1], output_dim)

        class _TCN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = body
                self.head = head

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.transpose(1, 2)
                x = self.body(x)
                x = x[:, :, -1]
                return self.head(x)

        return _TCN()


class TransformerRegressor(_TorchSequenceModel):
    """PatchTST inspired transformer encoder."""

    name = "transformer"

    def _build_module(self, input_features: int, output_dim: int) -> nn.Module:
        d_model = int(self.params.get("d_model", 128))
        num_heads = int(self.params.get("num_heads", 4))
        num_layers = int(self.params.get("num_layers", 3))
        dropout = float(self.params.get("dropout", 0.1))
        patch_len = int(self.params.get("patch_len", 8))
        if self.input_window % patch_len != 0:
            raise ValueError("input_window must be divisible by patch_len")
        n_patches = self.input_window // patch_len
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        proj = nn.Linear(patch_len * input_features, d_model)
        head = nn.Linear(d_model * n_patches, output_dim)

        class _Transformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = proj
                self.encoder = encoder
                self.head = head
                self.patch_len = patch_len
                self.n_patches = n_patches

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, _t, f = x.shape
                x = x.reshape(b, self.n_patches, self.patch_len * f)
                x = self.proj(x)
                x = self.encoder(x)
                x = x.reshape(b, -1)
                return self.head(x)

        return _Transformer()


class LSTMRegressor(_TorchSequenceModel):
    """Two layer LSTM regressor."""

    name = "lstm"

    def _build_module(self, input_features: int, output_dim: int) -> nn.Module:
        hidden_size = int(self.params.get("hidden_size", 128))
        num_layers = int(self.params.get("num_layers", 2))
        dropout = float(self.params.get("dropout", 0.2))
        rnn = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        head = nn.Linear(hidden_size, output_dim)

        class _LSTM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = rnn
                self.head = head

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.rnn(x)
                last = out[:, -1, :]
                return self.head(last)

        return _LSTM()


def build_torch_model(family: str, params: dict[str, Any], task: str) -> BaseModel:
    """Factory used by ``stockml.models.build_model`` for sequence learners."""
    _require_torch()
    if family == "tcn":
        return TCNRegressor(params, task=task)
    if family == "transformer":
        return TransformerRegressor(params, task=task)
    if family == "lstm":
        return LSTMRegressor(params, task=task)
    raise ValueError(f"Unknown torch model family: {family}")


def _ensure_unused_imports_referenced() -> None:
    # Reference numpy so static analyzers do not flag an unused import. The
    # tensor conversion in fit/predict relies on numpy arrays as the input
    # contract.
    _ = np.zeros(0)
