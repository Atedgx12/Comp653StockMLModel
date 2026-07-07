"""
Gaussian Naive Bayes from scratch — Module 5, Lec 5-3.
p(y=k|x) ∝ p(y=k) * prod_j N(x_j | mu_jk, sigma_jk^2)
"""
import numpy as np


class GaussianNaiveBayesScratch:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayesScratch":
        self.classes_ = np.unique(y)
        self.priors_  = {}
        self.means_   = {}
        self.vars_    = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / len(X)
            self.means_[c]  = Xc.mean(axis=0)
            self.vars_[c]   = Xc.var(axis=0) + 1e-9
        return self

    def _log_likelihood(self, X: np.ndarray, c) -> np.ndarray:
        mu  = self.means_[c]
        var = self.vars_[c]
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mu) ** 2 / var, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_posts = np.column_stack([
            np.log(self.priors_[c]) + self._log_likelihood(X, c)
            for c in self.classes_
        ])
        log_posts -= log_posts.max(axis=1, keepdims=True)
        probs = np.exp(log_posts)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
