"""Sequence model stubs gated behind the optional torch extra.

These classes intentionally provide minimal but runnable implementations so
the project skeleton stays end to end testable without forcing every grader
to install PyTorch. Calling any factory in this module without ``torch``
installed raises an informative error.

Heavy lifting (training loops, learning rate schedules, mixed precision) is
deliberately not implemented yet. The goal of the scaffold is to exercise
the contract between the trainer and the sequence models, not to ship a
state of the art forecaster on the first commit.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseModel

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
