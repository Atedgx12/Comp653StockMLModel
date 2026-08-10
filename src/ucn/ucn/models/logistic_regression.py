"""
Logistic Regression from scratch — Module 5, Lec 5-2.
Gradient descent on binary NLL with L2 regularisation and LR decay.
"""
import numpy as np
from ..utils import sigmoid, nll_loss
from ..training.metrics import accuracy


class LogisticRegressionScratch:
    def __init__(self, lr: float = 0.05, epochs: int = 300,
                 lam: float = 1e-3, decay: float = 0.995,
                 verbose: int = 50, seed: int = 42):
        self.lr      = lr
        self.epochs  = epochs
        self.lam     = lam
        self.decay   = decay
        self.verbose = verbose
        self.seed    = seed
        self.beta    = None
        self.loss_history = []

    def _step(self, beta, lr, X_aug, y):
        p        = sigmoid(X_aug @ beta)
        cost     = nll_loss(y, p)
        grad     = (X_aug.T @ (p - y)) / len(y) + self.lam * beta
        grad[0] -= self.lam * beta[0]          # no bias regularisation
        return cost, beta - lr * grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        rng   = np.random.default_rng(self.seed)
        N, d  = X.shape
        X_aug = np.hstack([X, np.ones((N, 1))])
        self.beta = rng.standard_normal(d + 1) * 0.01
        lr = self.lr
        for epoch in range(self.epochs):
            cost, self.beta = self._step(self.beta, lr, X_aug, y)
            self.loss_history.append(cost)
            lr *= self.decay
            if self.verbose and (epoch + 1) % self.verbose == 0:
                acc = accuracy(y, self.predict(X))
                print(f"  Epoch {epoch+1:4d}/{self.epochs}  "
                      f"NLL={cost:.5f}  acc={acc:.4f}", flush=True)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return sigmoid(X_aug @ self.beta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
