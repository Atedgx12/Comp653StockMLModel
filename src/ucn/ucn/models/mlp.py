"""
MLP from scratch — Module 5, Lec 5-5.
Two or more hidden ReLU layers, softmax output, mini-batch backprop.
"""
import numpy as np
from ..utils import softmax, cross_entropy_softmax
from ..training.metrics import accuracy


class MLPScratch:
    def __init__(self, hidden_sizes=(128, 64), lr=0.01, epochs=200,
                 lam=1e-4, batch_size=2048, decay=0.99,
                 verbose=10, seed=42):
        self.hidden_sizes = hidden_sizes
        self.lr          = lr
        self.epochs      = epochs
        self.lam         = lam
        self.batch_size  = batch_size
        self.decay       = decay
        self.verbose     = verbose
        self.seed        = seed
        self.params      = {}
        self.loss_history = []

    def _init_weights(self, d_in: int, d_out: int):
        rng   = np.random.default_rng(self.seed)
        sizes = [d_in] + list(self.hidden_sizes) + [d_out]
        for i in range(len(sizes) - 1):
            s = np.sqrt(2.0 / sizes[i])
            self.params[f"W{i+1}"] = rng.standard_normal((sizes[i], sizes[i+1])) * s
            self.params[f"b{i+1}"] = np.zeros(sizes[i+1])

    @staticmethod
    def relu(Z): return np.maximum(0, Z)

    @staticmethod
    def relu_grad(Z): return (Z > 0).astype(float)

    def _forward(self, X: np.ndarray) -> dict:
        cache = {"A0": X}
        n = len(self.hidden_sizes) + 1
        for i in range(1, n):
            Z = cache[f"A{i-1}"] @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache[f"Z{i}"] = Z
            cache[f"A{i}"] = self.relu(Z)
        Z_out = cache[f"A{n-1}"] @ self.params[f"W{n}"] + self.params[f"b{n}"]
        cache[f"Z{n}"] = Z_out
        cache[f"A{n}"] = softmax(Z_out)
        return cache

    def _backward(self, cache: dict, Y_oh: np.ndarray) -> dict:
        grads = {}
        N = Y_oh.shape[0]
        n = len(self.hidden_sizes) + 1
        delta = (cache[f"A{n}"] - Y_oh) / N
        for i in range(n, 0, -1):
            grads[f"dW{i}"] = cache[f"A{i-1}"].T @ delta
            grads[f"db{i}"] = delta.sum(axis=0)
            if i > 1:
                delta = (delta @ self.params[f"W{i}"].T) * self.relu_grad(cache[f"Z{i-1}"])
        return grads

    def _update(self, grads: dict, lr: float):
        for key in self.params:
            g = grads[f"d{key}"]
            if key.startswith("W"):
                g = g + self.lam * self.params[key]
            self.params[key] -= lr * g

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPScratch":
        rng = np.random.default_rng(self.seed)
        K   = len(np.unique(y))
        self._init_weights(X.shape[1], K)
        lr  = self.lr
        idx = np.arange(len(X))
        for epoch in range(self.epochs):
            rng.shuffle(idx)
            ep_loss = 0.0; n_b = 0
            for s in range(0, len(X), self.batch_size):
                b    = idx[s:s + self.batch_size]
                Y_oh = np.eye(K)[y[b].astype(int)]
                c    = self._forward(X[b])
                loss = cross_entropy_softmax(c[f"A{len(self.hidden_sizes)+1}"], Y_oh)
                self._update(self._backward(c, Y_oh), lr)
                ep_loss += loss; n_b += 1
            self.loss_history.append(ep_loss / n_b)
            lr *= self.decay
            if self.verbose and (epoch + 1) % self.verbose == 0:
                acc = accuracy(y, self.predict(X))
                print(f"  Epoch {epoch+1:4d}  loss={ep_loss/n_b:.5f}  acc={acc:.4f}",
                      flush=True)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[f"A{len(self.hidden_sizes)+1}"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
