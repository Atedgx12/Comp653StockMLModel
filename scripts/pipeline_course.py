"""
Stock Direction Prediction Pipeline — Course-Aligned Implementation
COMP 653 Statistical Machine Learning, Summer 2026
Zachary Powell  zp21@rice.edu

Every algorithm is implemented from scratch using only NumPy to demonstrate
mastery of Module 2 (information theory, optimization) and Module 5
(logistic regression, MLP, backpropagation, regularization, softmax).

LightGBM is included at the end as a professional baseline for comparison.
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED    = 42
rng     = np.random.default_rng(SEED)

# ===========================================================================
# MODULE 2 — Information Theory: Mutual Information Feature Selection
# ===========================================================================

def entropy(x, bins=20):
    """
    H(X) = -sum p(x) log2 p(x)
    Estimated via histogram binning (Module 2, Lec 2-6).
    """
    counts, _ = np.histogram(x, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def mutual_information(x, y, bins=20):
    """
    I(X; Y) = H(X) + H(Y) - H(X, Y)
    Measures how much knowing X reduces uncertainty about Y.
    Used here to rank and select the most predictive features
    (Module 2, Lec 2-6: information theory applied to feature selection).
    """
    hx  = entropy(x, bins)
    hy  = entropy(y, bins)
    counts2d, _, _ = np.histogram2d(x, y, bins=bins)
    probs2d = counts2d / counts2d.sum()
    probs2d_pos = probs2d[probs2d > 0]
    hxy = -np.sum(probs2d_pos * np.log2(probs2d_pos))
    return max(0.0, hx + hy - hxy)


def select_features_by_mi(X: np.ndarray, y: np.ndarray,
                           feature_names: list, k: int = 20) -> list:
    """
    Rank all features by I(feature; target) and keep the top-k.
    This grounds the feature set in information-theoretic principles
    rather than arbitrary engineering choices.
    """
    print(f"\n[MI Feature Selection] Computing mutual information for "
          f"{X.shape[1]} features ...", flush=True)
    scores = []
    for j in range(X.shape[1]):
        mi = mutual_information(X[:, j], y.astype(float))
        scores.append((feature_names[j], mi))
    scores.sort(key=lambda t: t[1], reverse=True)
    print("  Top features by I(X_j ; Y):")
    for name, mi in scores[:10]:
        print(f"    {name:30s}  I = {mi:.4f} bits")
    selected = [name for name, _ in scores[:k]]
    return selected

# ===========================================================================
# MODULE 5 — Logistic Regression from Scratch
# ===========================================================================

def sigmoid(z):
    """sigma(z) = 1 / (1 + exp(-z))  — clips to avoid overflow."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def nll_loss(y, p, eps=1e-12):
    """
    Negative Log Likelihood (binary cross-entropy):
    L = -1/N sum [ y log p + (1-y) log(1-p) ]
    Derived in Module 5 Lec 5-2 as the MLE loss for the Bernoulli model.
    """
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def logistic_regression_step(beta, lr, X_aug, y, lam=1e-3):
    """
    One gradient descent iteration on the NLL with L2 regularization.

    The gradient derivation (Module 5, Lec 5-2):
      p_i  = sigma(beta^T x_i)
      dL/d beta = (1/N) X^T (p - y)  +  lam * beta   (L2 regularization, Lec 5-2)

    Args:
        beta  : weight vector (d+1,) including bias
        lr    : learning rate
        X_aug : design matrix (N, d+1) with bias column appended
        y     : binary labels (N,)
        lam   : L2 regularization coefficient
    Returns:
        cost, beta_next
    """
    p        = sigmoid(X_aug @ beta)
    cost     = nll_loss(y, p)
    grad     = (X_aug.T @ (p - y)) / len(y)  +  lam * beta
    grad[0] -= lam * beta[0]                  # do not regularize the bias
    beta_next = beta - lr * grad
    return cost, beta_next


class LogisticRegressionScratch:
    """
    Full logistic regression trained by gradient descent from scratch.
    Implements the exact same algorithm as M5 Homework Task 1,
    extended to use L2 regularization and learning rate decay.
    """
    def __init__(self, lr=0.05, epochs=300, lam=1e-3, decay=0.995,
                 verbose=50, seed=SEED):
        self.lr      = lr
        self.epochs  = epochs
        self.lam     = lam
        self.decay   = decay
        self.verbose = verbose
        self.beta    = None
        self.loss_history = []

    def fit(self, X, y):
        N, d = X.shape
        X_aug = np.hstack([X, np.ones((N, 1))])
        self.beta = rng.standard_normal(d + 1) * 0.01
        lr = self.lr
        for epoch in range(self.epochs):
            cost, self.beta = logistic_regression_step(
                self.beta, lr, X_aug, y, self.lam)
            self.loss_history.append(cost)
            lr *= self.decay
            if self.verbose and (epoch + 1) % self.verbose == 0:
                acc = accuracy(y, self.predict(X))
                print(f"    Epoch {epoch+1:4d}/{self.epochs}  "
                      f"NLL={cost:.5f}  acc={acc:.4f}  lr={lr:.6f}",
                      flush=True)
        return self

    def predict_proba(self, X):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return sigmoid(X_aug @ self.beta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# ===========================================================================
# MODULE 5 — MLP + Backpropagation from Scratch
# ===========================================================================

def softmax(Z):
    """
    Numerically stable softmax along axis=1.
    Applied at the output layer for multi-class probability (Lec 5-5a).
    """
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy_softmax(Y_hat, Y_oh, eps=1e-12):
    """
    Cross-entropy loss with softmax output (Lec 5-5b):
    L = -1/N sum_i sum_k y_ik log(y_hat_ik)
    """
    return -np.mean(np.sum(Y_oh * np.log(Y_hat + eps), axis=1))


class MLPScratch:
    """
    Two-layer MLP:  Input -> Hidden (ReLU) -> Hidden (ReLU) -> Output (Softmax)
    Trained by full backpropagation derived from the chain rule (Lec 5-5b/c).
    Architecture mirrors M5 Homework Task 3/4, scaled up for tabular data.
    """
    def __init__(self, hidden_sizes=(128, 64), lr=0.01, epochs=200,
                 lam=1e-4, batch_size=2048, decay=0.99,
                 verbose=10, seed=SEED):
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

    def _init_weights(self, d_in, d_out):
        rng_ = np.random.default_rng(self.seed)
        sizes = [d_in] + list(self.hidden_sizes) + [d_out]
        for i in range(len(sizes) - 1):
            # He initialization for ReLU layers
            scale = np.sqrt(2.0 / sizes[i])
            self.params[f"W{i+1}"] = rng_.standard_normal(
                (sizes[i], sizes[i+1])) * scale
            self.params[f"b{i+1}"] = np.zeros(sizes[i+1])

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_grad(Z):
        return (Z > 0).astype(float)

    def _forward(self, X):
        """
        Forward pass through the network.
        Returns all intermediate activations needed for backprop.
        """
        cache = {"A0": X}
        n_layers = len(self.hidden_sizes) + 1
        for i in range(1, n_layers):
            Z = cache[f"A{i-1}"] @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache[f"Z{i}"] = Z
            cache[f"A{i}"] = self.relu(Z)      # hidden layers use ReLU
        # Output layer: linear pre-activation then softmax
        i = n_layers
        Z_out = cache[f"A{i-1}"] @ self.params[f"W{i}"] + self.params[f"b{i}"]
        cache[f"Z{i}"] = Z_out
        cache[f"A{i}"] = softmax(Z_out)        # output layer uses softmax
        return cache

    def _backward(self, cache, Y_oh):
        """
        Backpropagation via the chain rule (Module 5, Lec 5-5b).

        For softmax + cross-entropy, the output layer gradient simplifies to:
            delta_out = Y_hat - Y  (derived in Lec 5-5b)
        Then chain rule propagates delta back through each ReLU layer:
            delta_{l} = (delta_{l+1} W_{l+1}^T) * relu'(Z_l)
        Weight gradients:
            dW_l = A_{l-1}^T delta_l  /  N
        """
        grads  = {}
        N      = Y_oh.shape[0]
        n_layers = len(self.hidden_sizes) + 1

        # Output layer delta (softmax + cross-entropy shortcut)
        delta = (cache[f"A{n_layers}"] - Y_oh) / N   # (N, K)

        for i in range(n_layers, 0, -1):
            grads[f"dW{i}"] = cache[f"A{i-1}"].T @ delta
            grads[f"db{i}"] = delta.sum(axis=0)
            if i > 1:
                # Backprop through the activation of the previous hidden layer
                delta = (delta @ self.params[f"W{i}"].T) \
                        * self.relu_grad(cache[f"Z{i-1}"])
        return grads

    def _update(self, grads, lr):
        """Gradient descent weight update with L2 regularization."""
        for key in self.params:
            layer_idx = key[1:]          # "W1", "b2", etc.
            g = grads[f"d{key}"]
            if key.startswith("W"):
                g = g + self.lam * self.params[key]   # L2 on weights only
            self.params[key] -= lr * g

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        self._init_weights(X.shape[1], n_classes)
        lr = self.lr
        idx = np.arange(len(X))
        print(f"    MLP architecture: {X.shape[1]} -> "
              f"{' -> '.join(str(h) for h in self.hidden_sizes)} "
              f"-> {n_classes}  (ReLU hidden, Softmax output)", flush=True)

        for epoch in range(self.epochs):
            rng.shuffle(idx)
            epoch_loss = 0.0
            n_batches  = 0
            for start in range(0, len(X), self.batch_size):
                batch = idx[start:start + self.batch_size]
                X_b   = X[batch]
                y_b   = y[batch]
                # One-hot encode the batch labels
                Y_oh  = np.eye(n_classes)[y_b.astype(int)]
                cache = self._forward(X_b)
                loss  = cross_entropy_softmax(cache[f"A{len(self.hidden_sizes)+1}"],
                                              Y_oh)
                grads = self._backward(cache, Y_oh)
                self._update(grads, lr)
                epoch_loss += loss
                n_batches  += 1
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            lr *= self.decay

            if self.verbose and (epoch + 1) % self.verbose == 0:
                pred = self.predict(X)
                acc  = accuracy(y, pred)
                print(f"    Epoch {epoch+1:4d}/{self.epochs}  "
                      f"CrossEntropy={avg_loss:.5f}  acc={acc:.4f}  lr={lr:.6f}",
                      flush=True)
        return self

    def predict_proba(self, X):
        cache = self._forward(X)
        return cache[f"A{len(self.hidden_sizes)+1}"]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# ===========================================================================
# MODULE 5 — Naive Bayes (Gaussian) from Scratch
# ===========================================================================

class GaussianNaiveBayesScratch:
    """
    Gaussian Naive Bayes: assumes feature likelihoods are Gaussian per class.
    p(y=k | x) ∝ p(y=k) * prod_j N(x_j | mu_jk, sigma_jk^2)
    Derived in Module 5, Lec 5-3.
    The naive independence assumption makes computation tractable even with
    high-dimensional feature vectors.
    """
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_  = {}
        self.means_   = {}
        self.vars_    = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / len(X)
            self.means_[c]  = Xc.mean(axis=0)
            self.vars_[c]   = Xc.var(axis=0) + 1e-9   # Laplace-style floor
        return self

    def _log_likelihood(self, X, c):
        mu  = self.means_[c]
        var = self.vars_[c]
        return -0.5 * np.sum(np.log(2 * np.pi * var)
                             + (X - mu) ** 2 / var, axis=1)

    def predict_proba(self, X):
        log_posts = np.column_stack([
            np.log(self.priors_[c]) + self._log_likelihood(X, c)
            for c in self.classes_
        ])
        log_posts -= log_posts.max(axis=1, keepdims=True)
        probs = np.exp(log_posts)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# ===========================================================================
# PRIMARY MODEL — Unified Course Network
# ===========================================================================
#
# Architecture: three parallel branches trained jointly end-to-end.
# Each branch implements one Module 5 algorithm as an intentional
# architectural component; the meta-layer learns to weight them.
#
#   Input x  (d MI-selected features)
#         │
#   ┌─────┴──────────────────────────────────────────────────────┐
#   │ Branch A — Logistic Regression (Module 5, Lec 5-2)        │
#   │   Linear(d -> K) -> Sigmoid                                 │
#   │   A single linear + sigmoid layer IS logistic regression. │
#   │   It contributes a linear decision boundary to the final  │
#   │   classification.                                         │
#   ├────────────────────────────────────────────────────────────┤
#   │ Branch B — Naive Bayes (Module 5, Lec 5-3)               │
#   │   Learnable per-feature Gaussian normalization            │
#   │   (trainable μ_j, σ_j for each feature j)                │
#   │   followed by Linear(d -> K) -> Sigmoid.                   │
#   │   The normalization layer encodes the Naive Bayes         │
#   │   conditional independence assumption as differentiable   │
#   │   parameters that are updated by backpropagation.        │
#   ├────────────────────────────────────────────────────────────┤
#   │ Branch C — Deep MLP (Module 5, Lec 5-5b/c)              │
#   │   Linear(d -> H1) -> ReLU                                  │
#   │   Linear(H1 -> H2) -> ReLU                                 │
#   │   Captures nonlinear feature interactions that neither   │
#   │   logistic regression nor Naive Bayes can express.       │
#   └─────┬──────────────────────────────────────────────────────┘
#         │ concat(a_lr, a_nb, a_mlp): shape (N, K + K + H2)
#         │
#   Meta-layer: Linear(K+K+H2 -> K) -> Softmax
#         │
#   Output: P(Up), P(Down)
#
# All parameters are updated by a single backpropagation pass per
# mini-batch.  The meta-layer learns optimal weighting across branches.

class UnifiedCourseNetwork:
    """
    Single end-to-end network integrating all three Module 5 algorithms
    as parallel branches trained jointly via backpropagation.
    """

    def __init__(self, hidden_sizes=(128, 64), lr=0.001, epochs=200,
                 lam=1e-4, batch_size=2048, beta1=0.9, beta2=0.999,
                 verbose=20, seed=SEED,
                 dropout_rate=0.4, patience=40, val_frac=0.15,
                 use_sent=False, meta_dropout=0.2,
                 grad_clip=1.0, noise_frac=0.02, warmup_epochs=5,
                 use_fgsm=False, fgsm_eps=0.01, pgd_steps=3):
        self.hidden_sizes  = hidden_sizes
        self.lr            = lr
        self.epochs        = epochs
        self.lam           = lam
        self.batch_size    = batch_size
        self.beta1         = beta1
        self.beta2         = beta2
        self.verbose       = verbose
        self.seed          = seed
        self.dropout_rate  = dropout_rate   # MLP branch inverted dropout
        self.meta_dropout  = meta_dropout   # meta-layer input dropout
        self.patience      = patience       # best-checkpoint patience (not stop)
        self.val_frac      = val_frac       # fraction of rows held out for val
        self.use_sent      = use_sent       # Branch D: NLP sentiment branch
        self.grad_clip     = grad_clip      # global gradient-norm clip threshold
        self.noise_frac    = noise_frac     # Gaussian augmentation scale (Module 8)
        self.warmup_epochs = warmup_epochs  # linear LR warmup before cosine decay
        self.use_fgsm      = use_fgsm       # FGSM adversarial perturbation (Module 8)
        self.fgsm_eps      = fgsm_eps       # FGSM epsilon budget
        self.pgd_steps     = pgd_steps      # PGD inner steps (1 = standard FGSM)
        self.params       = {}
        self.m            = {}   # Adam first moment  (per param)
        self.v            = {}   # Adam second moment (per param)
        self.t            = 0    # global step counter for bias correction
        self.loss_history = []
        self.val_loss_history = []
        self.n_classes    = None

    def _init_weights(self, d, K):
        rng_ = np.random.default_rng(self.seed)
        self.n_classes = K

        # When Branch D is active the last column of X is the sentiment score.
        # Branches A/B/C receive only the price features; Branch D receives
        # only the sentiment score so the meta-layer can learn optimal weights
        # for each information source independently.
        dp = d - 1 if self.use_sent else d   # price feature dimension

        # Branch A: LR head — dp -> K
        self.params['W_lr'] = rng_.standard_normal((dp, K)) * np.sqrt(1.0 / dp)
        self.params['b_lr'] = np.zeros(K)

        # Branch B: NB — learnable per-feature Gaussian normalization -> dp -> K
        self.params['mu_nb']   = np.zeros(dp)
        self.params['lsig_nb'] = np.zeros(dp)
        self.params['W_nb']    = rng_.standard_normal((dp, K)) * np.sqrt(1.0 / dp)
        self.params['b_nb']    = np.zeros(K)

        # Branch C: MLP hidden layers — dp -> h1 -> h2 -> ... (feeds meta)
        sizes = [dp] + list(self.hidden_sizes)
        for i in range(len(sizes) - 1):
            s = np.sqrt(2.0 / sizes[i])   # He initialization for ReLU
            self.params[f'Wm{i+1}'] = rng_.standard_normal((sizes[i], sizes[i+1])) * s
            self.params[f'bm{i+1}'] = np.zeros(sizes[i+1])

        # Branch D: NLP Sentiment — 1 -> K -> Sigmoid
        # A single VADER compound score (already cross-sectionally ranked)
        # is projected to K logits so the meta-layer treats it symmetrically
        # with the other branches.
        if self.use_sent:
            self.params['W_sent'] = rng_.standard_normal((1, K)) * 0.1
            self.params['b_sent'] = np.zeros(K)

        # Meta-layer: concat(a_lr, a_nb, A_mlp [, a_sent]) -> K -> Softmax
        H_last  = self.hidden_sizes[-1]
        meta_in = K + K + H_last + (K if self.use_sent else 0)
        self.params['W_meta'] = rng_.standard_normal((meta_in, K)) * np.sqrt(2.0 / meta_in)
        self.params['b_meta'] = np.zeros(K)

        # Adam moment accumulators — one pair per parameter tensor
        for key in self.params:
            self.m[key] = np.zeros_like(self.params[key])
            self.v[key] = np.zeros_like(self.params[key])

    @staticmethod
    def _relu(Z):     return np.maximum(0, Z)
    @staticmethod
    def _relu_g(Z):   return (Z > 0).astype(float)

    def _forward(self, X, training=True):
        # Separate price features from the sentiment score (last column).
        # Branch D processes the sentiment independently so the meta-layer
        # can learn how much to trust news signals vs price signals.
        if self.use_sent:
            X_price = X[:, :-1]
            X_sent  = X[:, -1:]
        else:
            X_price = X
        c = {'X': X_price}

        # Branch A: Logistic Regression (Lec 5-2)
        z_lr = X_price @ self.params['W_lr'] + self.params['b_lr']
        a_lr = sigmoid(z_lr)
        c.update({'z_lr': z_lr, 'a_lr': a_lr})

        # Branch B: Naive Bayes normalization (Lec 5-3)
        sig  = np.exp(self.params['lsig_nb']) + 1e-8
        X_n  = (X_price - self.params['mu_nb']) / sig
        z_nb = X_n @ self.params['W_nb'] + self.params['b_nb']
        a_nb = sigmoid(z_nb)
        c.update({'sig': sig, 'X_n': X_n, 'z_nb': z_nb, 'a_nb': a_nb})

        # Branch C: Deep MLP with inverted dropout (Lec 5-5b/c)
        A = X_price;  mlp = {'A0': X_price}
        p = self.dropout_rate
        for i in range(len(self.hidden_sizes)):
            Z = A @ self.params[f'Wm{i+1}'] + self.params[f'bm{i+1}']
            A = self._relu(Z)
            if training and p > 0:
                mask = (rng.random(A.shape) >= p).astype(float) / (1.0 - p)
                A    = A * mask
                mlp[f'drop{i+1}'] = mask
            mlp[f'Z{i+1}'] = Z
            mlp[f'A{i+1}'] = A
        c['mlp'] = mlp

        # Branch D: NLP Sentiment (Lec not in course — extension)
        # The cross-sectionally ranked VADER score passes through a single
        # Linear -> Sigmoid layer.  This gives the meta-layer a dedicated
        # sentiment logit to weigh against the price-based branches.
        parts = [a_lr, a_nb, A]
        if self.use_sent:
            z_sent = X_sent @ self.params['W_sent'] + self.params['b_sent']
            a_sent = sigmoid(z_sent)
            c.update({'X_sent': X_sent, 'z_sent': z_sent, 'a_sent': a_sent})
            parts.append(a_sent)

        # Meta-layer — with optional inverted dropout on the concatenated
        # branch outputs.  Dropping branch activations before the final
        # linear prevents the meta-layer from over-relying on any one branch
        # and forces it to learn robust cross-branch combinations.
        cat = np.hstack(parts)
        if training and self.meta_dropout > 0:
            md   = (rng.random(cat.shape) >= self.meta_dropout).astype(float) \
                   / (1.0 - self.meta_dropout)
            cat  = cat * md
            c['meta_drop'] = md
        z_meta = cat @ self.params['W_meta'] + self.params['b_meta']
        Y_hat  = softmax(z_meta)
        c.update({'cat': cat, 'Y_hat': Y_hat})
        return c

    def _backward(self, c, Y_oh):
        X  = c['X']
        N  = Y_oh.shape[0]
        K  = self.n_classes
        g  = {}

        # Output gradient: softmax + cross-entropy shortcut (Lec 5-5b)
        d = (c['Y_hat'] - Y_oh) / N

        # Meta-layer backward — apply meta-dropout mask if present
        g['W_meta'] = c['cat'].T @ d
        g['b_meta'] = d.sum(0)
        dc = d @ self.params['W_meta'].T
        if 'meta_drop' in c:
            dc = dc * c['meta_drop']   # gradient through meta dropout
        H  = self.hidden_sizes[-1]
        d_lr  = dc[:, :K]
        d_nb  = dc[:, K:2*K]
        d_mlp = dc[:, 2*K:2*K+H]   # MLP contributes H columns, not open-ended

        # Branch A backward: sigmoid gradient
        dz_lr    = d_lr * c['a_lr'] * (1 - c['a_lr'])
        g['W_lr'] = X.T @ dz_lr
        g['b_lr'] = dz_lr.sum(0)

        # Branch B backward: sigmoid -> linear -> normalization
        dz_nb        = d_nb * c['a_nb'] * (1 - c['a_nb'])
        g['W_nb']    = c['X_n'].T @ dz_nb
        g['b_nb']    = dz_nb.sum(0)
        dX_n         = dz_nb @ self.params['W_nb'].T
        g['mu_nb']   = (-dX_n / c['sig']).sum(0)
        g['lsig_nb'] = (-dX_n * c['X_n']).sum(0)     # chain rule through exp

        # Branch D backward: sigmoid gradient on sentiment branch
        if self.use_sent:
            H   = self.hidden_sizes[-1]
            d_sent = dc[:, 2*K+H:]
            dz_sent = d_sent * c['a_sent'] * (1 - c['a_sent'])
            g['W_sent'] = c['X_sent'].T @ dz_sent
            g['b_sent'] = dz_sent.sum(0)

        # Branch C backward: dropout mask then ReLU chain rule (Lec 5-5b)
        # The dropout mask stored in the forward cache has already been
        # inverted-scaled, so multiplying dm by the same mask propagates
        # the gradient through only the active units.
        dm = d_mlp;  mlp = c['mlp']
        for i in range(len(self.hidden_sizes), 0, -1):
            if f'drop{i}' in mlp:
                dm = dm * mlp[f'drop{i}']
            dm           = dm * self._relu_g(mlp[f'Z{i}'])
            g[f'Wm{i}']  = mlp[f'A{i-1}'].T @ dm
            g[f'bm{i}']  = dm.sum(0)
            if i > 1:
                dm = dm @ self.params[f'Wm{i}'].T

        # Input-space gradient — sum of contributions from all three price branches.
        # Used by FGSM to find the direction that maximises the training loss so the
        # model can be retrained on the adversarial worst-case sample (Module 8, Lec 8-4).
        #   Branch A (LR):  dz_lr @ W_lr.T
        #   Branch B (NB):  (dz_nb @ W_nb.T) / sig  — chain rule through Gaussian norm
        #   Branch C (MLP): dm @ Wm1.T               — through the first hidden layer
        dX = (dz_lr @ self.params['W_lr'].T
              + dX_n / c['sig']
              + dm @ self.params['Wm1'].T)
        return g, dX

    def _update(self, g, lr):
        """
        Adam update (Module 6, Lec 6-5).
        Combines RMSProp adaptive scaling with momentum, plus bias correction
        so that early steps are not artificially damped toward zero.

            m_t = beta1 * m_{t-1} + (1-beta1) * grad
            v_t = beta2 * v_{t-1} + (1-beta2) * grad^2
            m_hat = m_t / (1 - beta1^t)
            v_hat = v_t / (1 - beta2^t)
            theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
        """
        # Global gradient-norm clip (Pascanu et al. 2013):
        # financial data heavy tails can produce gradient spikes that send
        # a weight tensor into a region Adam cannot recover from.
        if self.grad_clip > 0:
            total_norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))
            if total_norm > self.grad_clip:
                scale = self.grad_clip / (total_norm + 1e-8)
                g = {k: v * scale for k, v in g.items()}

        self.t += 1
        eps = 1e-8
        b1c = 1.0 - self.beta1 ** self.t    # bias-correction factor for m
        b2c = 1.0 - self.beta2 ** self.t    # bias-correction factor for v
        for k, val in self.params.items():
            grad = g[k]
            if k.startswith('W'):
                grad = grad + self.lam * val          # L2 regularization
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[k] / b1c
            v_hat = self.v[k] / b2c
            self.params[k] = val - lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X, y):
        K = len(np.unique(y))
        self._init_weights(X.shape[1], K)
        H = self.hidden_sizes[-1]

        # Hold out the last val_frac fraction of rows (date-ordered) as a
        # validation set for early stopping.  Rows are already sorted by
        # date, so this is a strict temporal hold-out with no lookahead.
        n_val   = max(int(len(X) * self.val_frac), 1)
        X_tr    = X[:len(X) - n_val];   y_tr = y[:len(y) - n_val]
        X_val   = X[len(X) - n_val:];   y_val = y[len(y) - n_val:]
        idx_tr  = np.arange(len(X_tr))

        dp      = X.shape[1] - 1 if self.use_sent else X.shape[1]
        mlp_str = ' -> '.join(str(h) for h in self.hidden_sizes)
        meta_in = K + K + H + (K if self.use_sent else 0)
        print(f"    UnifiedCourseNetwork:", flush=True)
        print(f"      Branch A (LR)  {dp}d -> {K}   [sigmoid, Lec 5-2]", flush=True)
        print(f"      Branch B (NB)  {dp}d -> learnable Gaussian norm "
              f"-> {K}  [Lec 5-3]", flush=True)
        print(f"      Branch C (MLP) {dp}d -> {mlp_str}  "
              f"[ReLU, dropout={self.dropout_rate}, Lec 5-5]", flush=True)
        if self.use_sent:
            print(f"      Branch D (Sent) 1d -> {K}   "
                  f"[VADER NLP sentiment, text branch]", flush=True)
        print(f"      Meta-layer     {meta_in} -> {K}  "
              f"[meta_dropout={self.meta_dropout}, learned branch weighting]",
              flush=True)
        print(f"      Full schedule: {self.epochs} epochs  "
              f"val_rows={n_val:,}  warmup={self.warmup_epochs}ep  "
              f"cosine LR  grad_clip={self.grad_clip}  "
              f"noise_frac={self.noise_frac}  best-checkpoint restore",
              flush=True)

        best_val   = np.inf
        best_params = None
        no_improve  = 0

        # Per-feature noise scale for augmentation (Module 8, Lec 8-3).
        # Gaussian noise is added to each mini-batch at training time,
        # scaled to noise_frac * empirical std of each feature column so
        # the perturbation matches the natural day-to-day variation.
        noise_scale = (X_tr.std(axis=0) * self.noise_frac
                       if self.noise_frac > 0 else None)

        for epoch in range(self.epochs):
            # Linear warmup for the first warmup_epochs epochs stabilises
            # Adam whose moment accumulators are zero-initialised and would
            # otherwise take an oversized effective step at epoch 0.
            # After warmup, cosine annealing decays LR from self.lr to
            # self.lr/100 over the remaining budget.
            if epoch < self.warmup_epochs:
                lr_t = self.lr * (epoch + 1) / max(self.warmup_epochs, 1)
            else:
                ce   = epoch - self.warmup_epochs
                ct   = max(self.epochs - self.warmup_epochs, 1)
                lr_t = self.lr * (0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * ce / ct)))

            rng.shuffle(idx_tr)
            ep_loss = 0.0;  n_b = 0
            for s in range(0, len(X_tr), self.batch_size):
                b    = idx_tr[s:s + self.batch_size]
                # Gaussian noise augmentation: adds a perturbation drawn
                # from N(0, noise_scale) to each feature of the batch.
                # The scale is proportional to each feature's empirical std
                # so the signal-to-noise ratio is consistent across features.
                X_b  = (X_tr[b] + rng.standard_normal(X_tr[b].shape) * noise_scale
                        if noise_scale is not None else X_tr[b])
                Y_oh = np.eye(K)[y_tr[b].astype(int)]
                c        = self._forward(X_b, training=True)
                loss     = cross_entropy_softmax(c['Y_hat'], Y_oh)
                g, dX    = self._backward(c, Y_oh)
                # FGSM adversarial perturbation (Module 8, Lec 8-4):
                # step the input in the sign of the loss gradient to build
                # a worst-case sample, then retrain the model on that sample.
                # The inner maximisation uses the gradient already computed
                # above so the cost is one extra forward + backward pass.
                if self.use_fgsm:
                    # PGD: pgd_steps iterations of smaller FGSM steps.
                    # Each step uses fgsm_eps / pgd_steps so the total
                    # perturbation budget stays constant regardless of step count.
                    # More steps produce a tighter adversarial example that sits
                    # closer to the true worst-case input (Madry et al. 2017).
                    step = self.fgsm_eps / max(self.pgd_steps, 1)
                    X_adv = X_b.copy()
                    for _ in range(self.pgd_steps):
                        c_tmp   = self._forward(X_adv, training=False)
                        _, dX_s = self._backward(c_tmp, Y_oh)
                        dX_full = (np.hstack([dX_s, np.zeros((len(b), 1))])
                                   if self.use_sent else dX_s)
                        X_adv   = X_adv + step * np.sign(dX_full)
                    c_adv = self._forward(X_adv, training=True)
                    g, _  = self._backward(c_adv, Y_oh)
                self._update(g, lr_t)
                ep_loss += loss;  n_b += 1
            avg = ep_loss / n_b
            self.loss_history.append(avg)

            # Validation CE (no dropout — training=False)
            Y_oh_val = np.eye(K)[y_val.astype(int)]
            c_val    = self._forward(X_val, training=False)
            val_ce   = cross_entropy_softmax(c_val['Y_hat'], Y_oh_val)
            self.val_loss_history.append(val_ce)

            if val_ce < best_val - 1e-6:
                best_val    = val_ce
                best_params = {k: v.copy() for k, v in self.params.items()}
                no_improve  = 0
            else:
                no_improve += 1

            if self.verbose and (epoch + 1) % self.verbose == 0:
                acc     = accuracy(y_tr, self.predict(X_tr))
                val_acc = accuracy(y_val,
                                   np.argmax(c_val['Y_hat'], axis=1))
                marker = " *" if no_improve == 0 else ""
                print(f"    Epoch {epoch+1:4d}/{self.epochs}  "
                      f"CE={avg:.5f}  acc={acc:.4f}  "
                      f"val_CE={val_ce:.5f}  val_acc={val_acc:.4f}  "
                      f"LR={lr_t:.2e}{marker}", flush=True)

            # Early stopping: halt once patience epochs pass without
            # improvement.  Best-checkpoint restore still applies so the
            # returned model uses the weights from the best validation epoch.
            if self.patience and no_improve >= self.patience:
                if self.verbose:
                    print(f"    Early stop at epoch {epoch+1}"
                          f" (no val improvement for {self.patience} epochs)",
                          flush=True)
                break

        if best_params is not None:
            self.params = best_params
            print(f"    Restored best checkpoint  "
                  f"(val_CE={best_val:.5f})", flush=True)
        return self

    def predict_proba(self, X):
        return self._forward(X, training=False)['Y_hat']

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# ===========================================================================
# Utility: metrics
# ===========================================================================

def accuracy(y_true, y_pred):
    return (y_true.astype(int) == y_pred.astype(int)).mean()


def roc_auc(y_true, y_score):
    """
    AUC via the Wilcoxon rank-sum statistic — O(n log n), no sklearn.

    Equivalent to the Mann-Whitney U but computed by sorting once:
      AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    where R_pos is the sum of ranks of positive samples when all scores
    are sorted in ascending order.
    """
    y_true = y_true.astype(int)
    n_pos  = int(y_true.sum())
    n_neg  = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order    = np.argsort(y_score)
    ranks    = np.arange(1, len(y_true) + 1, dtype=np.float64)
    rank_sum = ranks[y_true[order] == 1].sum()
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def majority_class_baseline(y_true):
    """Always-predict-majority reference: accuracy = majority-class rate, AUC = 0.5."""
    p_up = float(np.mean(y_true))
    return max(p_up, 1.0 - p_up), 0.5


def paired_bootstrap_auc_diff(y_true, prob_a, prob_b, n_boot=1000, seed=SEED):
    """Paired bootstrap on the A-B AUC gap; returns gap, 95% CI, and p-value for H0: gap=0."""
    rng_b = np.random.default_rng(seed)
    n     = len(y_true)
    obs   = roc_auc(y_true, prob_a) - roc_auc(y_true, prob_b)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng_b.integers(0, n, n)
        yt  = y_true[idx]
        if yt.min() == yt.max():
            diffs[b] = 0.0
            continue
        diffs[b] = roc_auc(yt, prob_a[idx]) - roc_auc(yt, prob_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_val  = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return obs, float(lo), float(hi), min(p_val, 1.0)


def make_eda_figures(X, y, feat_names, out_dir=OUT_DIR):
    """Correlation heatmap, PCA scree, and class-balance figures (report Section 4.2)."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)

    corr = np.corrcoef(Xs, rowvar=False)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=7)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_correlation.png"), dpi=150)
    plt.close(fig)

    eigvals = np.clip(np.linalg.eigvalsh(np.cov(Xs, rowvar=False))[::-1], 0, None)
    evr     = eigvals / eigvals.sum()
    comps   = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(comps, evr, color="steelblue", label="Individual")
    ax.plot(comps, np.cumsum(evr), "r-o", markersize=3, label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA Scree — Feature Variance Structure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_pca_variance.png"), dpi=150)
    plt.close(fig)

    n_up, n_down = int((y == 1).sum()), int((y == 0).sum())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Down", "Up"], [n_down, n_up], color=["indianred", "seagreen"])
    ax.set_ylabel("Samples")
    ax.set_title(f"Class Balance (Up = {n_up / (n_up + n_down):.3f})")
    for i, v in enumerate([n_down, n_up]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_class_balance.png"), dpi=150)
    plt.close(fig)

    print(f"    Saved eda_correlation.png, eda_pca_variance.png, eda_class_balance.png "
          f"(Up={n_up:,}  Down={n_down:,})", flush=True)

# ===========================================================================
# Data: S&P 500 download and feature engineering
# ===========================================================================

# ---------------------------------------------------------------------------
# News sentiment via yfinance + VADER NLP
# ---------------------------------------------------------------------------

def fetch_sentiment(tickers, close_index):
    """
    Fetch recent news headlines per ticker via yfinance and score each
    headline with VADER (Valence Aware Dictionary and sEntiment Reasoner),
    a rule-based NLP sentiment analyser that requires no training data and
    is specifically calibrated for short social/financial text.

    Pipeline
    --------
    1. For each ticker call yf.Ticker(t).news to get the last ~3 months of
       headlines (title strings).
    2. Score each headline: VADER returns compound in [-1, +1] where
       +1 is maximally positive and -1 is maximally negative.
    3. Group by (ticker, date) and average the compound scores.
    4. Align to the full trading calendar (close_index), forward-fill up
       to 3 business days, then fill remaining NaN with 0 (neutral).
    5. Cache result to sentiment_cache.parquet so re-runs are instant.

    Returns
    -------
    pd.DataFrame  shape (len(close_index), len(tickers))
                  columns = ticker symbols, index = trading dates
                  values  = daily average VADER compound score
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    cache = os.path.join(OUT_DIR, "sentiment_cache.parquet")
    if os.path.exists(cache):
        print("    Loading cached sentiment scores ...", flush=True)
        sent = pd.read_parquet(cache)
        # Realign to current trading calendar in case it changed
        return sent.reindex(close_index).ffill(limit=3).fillna(0.0)

    print(f"    Fetching news for {len(tickers)} tickers (VADER NLP) ...",
          flush=True)
    analyzer = SentimentIntensityAnalyzer()
    records  = []

    for i, t in enumerate(tickers):
        try:
            news_items = yf.Ticker(t).news or []
            for item in news_items:
                # yfinance 1.4.x wraps items under a 'content' sub-dict;
                # older versions put keys directly on the item dict.
                content = item.get("content", item)
                # Timestamp: try multiple key names across versions
                raw_ts = (content.get("pubDate")
                          or content.get("providerPublishTime")
                          or item.get("providerPublishTime"))
                if raw_ts is None:
                    continue
                # pubDate may be an ISO string; providerPublishTime is a Unix int
                if isinstance(raw_ts, str):
                    try:
                        date = pd.Timestamp(raw_ts).normalize()
                    except Exception:
                        continue
                else:
                    date = pd.Timestamp(int(raw_ts), unit="s").normalize()
                title = (content.get("title")
                         or content.get("headline")
                         or item.get("title", ""))
                if not title:
                    continue
                score = analyzer.polarity_scores(str(title))["compound"]
                records.append({"date": date, "ticker": t, "score": score})
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"    Sentiment: {i+1}/{len(tickers)} tickers ...",
                  flush=True)

    if not records:
        print("    No news data available — sentiment set to neutral (0).",
              flush=True)
        sent = pd.DataFrame(0.0, index=close_index, columns=tickers)
        sent.to_parquet(cache)
        return sent

    df_news = pd.DataFrame(records)
    # Average VADER compound per (date, ticker)
    sent_pivot = (df_news
                  .groupby(["date", "ticker"])["score"]
                  .mean()
                  .unstack(fill_value=np.nan))

    # Align to trading calendar, forward-fill up to 3 days, zero-fill rest
    sent = (sent_pivot
            .reindex(close_index)
            .reindex(columns=tickers, fill_value=np.nan)
            .ffill(limit=3)
            .fillna(0.0))

    n_nonzero = (sent != 0).sum().sum()
    print(f"    Sentiment cache: {n_nonzero:,} non-neutral scores across "
          f"{len(tickers)} tickers.", flush=True)
    sent.to_parquet(cache)
    return sent


def get_sp500_tickers():
    """
    Return a broad universe of US equities.
    Priority order:
      1. Wikipedia S&P 500 table  (fast, authoritative, blocked sometimes)
      2. Hardcoded ~500-ticker list covering S&P 500 + NASDAQ 100 + DJIA
         (zero-auth fallback — always works)
    """
    print("[1] Fetching ticker universe ...", flush=True)
    try:
        tables  = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            flavor="lxml")
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"    Wikipedia OK: {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"    Wikipedia failed ({e}). Using hardcoded 500-ticker universe.")

    # Hardcoded S&P 500 + NASDAQ 100 composite (current as of mid-2026).
    # This list is embedded so the pipeline never needs network access just
    # to get ticker symbols.
    return [
        # Information Technology
        "AAPL","MSFT","NVDA","AVGO","ORCL","AMD","QCOM","TXN","AMAT","MU",
        "LRCX","KLAC","ADI","MCHP","CDNS","SNPS","FTNT","PANW","CRWD","ZS",
        "DDOG","NET","SNOW","PLTR","UBER","TTD","VEEV","PAYC","ANSS","TER",
        "KEYS","TRMB","FSLR","ENPH","SEDG","SMCI","HPE","HPQ","DELL","WDC",
        "STX","NTAP","CDW","ZBRA","FFIV","AKAM","EPAM","CTSH","IT",
        "ACN","IBM","CSC","LDOS","SAIC","CACI",
        # Communication Services
        "GOOGL","GOOG","META","NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR",
        "WBD","PARA","FOX","FOXA","DISH","LUMN","ZAYO","SIRI","SPOT","EA",
        "TTWO","ATVI","RBLX","MTCH","IAC","LYV","OMC","IPG","NWSA","NWS",
        # Consumer Discretionary
        "AMZN","TSLA","MCD","NKE","SBUX","HD","LOW","TJX","BKNG","MAR",
        "HLT","YUM","CMG","RCL","CCL","NCLH","GM","F","APTV","GNTX",
        "LVS","WYNN","MGM","CZR","DKNG","PENN","ORLY","AZO","AAP","KMX",
        "AN","PAG","LAD","GPC","BBY","ROST","ETSY","EBAY","TRIP","EXPE",
        "ABNB","LYFT","DASH","CPRT","SAIA","XPO","ODFL","WERN","CHRW",
        "PDD","JD","BABA","SE","MELI",
        # Consumer Staples
        "WMT","COST","PG","KO","PEP","PM","MO","MDLZ","KHC","GIS",
        "CPB","SJM","CAG","HRL","MKC","CL","EL","CHD","SPB",
        "CLX","ENR","COTY","UL","BF-B","STZ","SAM",
        "TAP","MNST","CELH","FIZZ",
        # Energy
        "XOM","CVX","COP","EOG","SLB","MPC","VLO","PSX","DVN",
        "FANG","APA","HAL","BKR","OXY","CNX","RRC","EQT",
        "KMI","WMB","OKE","ET","EPD","MPLX","PAA","ENB","TRP","SU",
        # Financials
        "BRK-B","JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP",
        "USB","PNC","TFC","COF","SYF","AIG","MET","PRU","AFL",
        "ALL","PGR","TRV","CB","HIG","AON","CINF","GL","RNR",
        "SPGI","MCO","ICE","CME","CBOE","NDAQ","FIS","FISV","GPN","ADP",
        "PAYX","BR","WEX","TW","MSCI","VRSK",
        # Health Care
        "UNH","JNJ","LLY","ABBV","MRK","ABT","TMO","DHR","BMY","AMGN",
        "GILD","VRTX","REGN","ISRG","SYK","MDT","BSX","ZBH","EW","BDX",
        "BAX","IDXX","IQV","CRL","ILMN","BIIB","MRNA","PFE",
        "CI","CVS","HUM","CNC","MOH","ELV","MCK","CAH","HSIC",
        "DGX","LH","MTD","WST","PODD","DXCM","GEHC","HCA","THC",
        # Industrials
        "GE","HON","RTX","BA","LMT","NOC","GD","TDG","HII",
        "CAT","DE","EMR","ETN","PH","ROK","SWK","IR","XYL",
        "AME","FTV","OTIS","CARR","TT","JCI","NDSN","GGG","FAST","MSM",
        "PNR","IEX","MIDD","GNRC","WMS","RRX","ITT",
        "UPS","FDX","DAL","UAL","AAL","LUV","ALK","JBLU",
        "CSX","UNP","NSC","WAB","GATX","R","URI","RSG","WM",
        "CTAS","EXPO","CPNG","GWW","MAS","SNA","TKR","PCAR","CMI",
        # Materials
        "LIN","APD","ECL","SHW","PPG","RPM","DD","DOW","LYB","IFF",
        "CE","EMN","ALB","AVNT","HUN","AXTA","OLN","CF","MOS","NUE",
        "STLD","RS","CMC","ATI","CLF","AA","FCX","NEM","GOLD",
        "AEM","KGC","WPM","PAAS","HL","CDE","EXK",
        # Real Estate
        "PLD","AMT","EQIX","CCI","SPG","PSA","EQR","MAA","AVB","ESS",
        "UDR","IRM","WELL","VTR","ARE","BXP","SLG","KIM","REG","FRT",
        "O","NNN","VICI","GLPI","SBAC","DLR",
        # Utilities
        "NEE","DUK","SO","AEP","EXC","SRE","D","PCG","PEG","XEL",
        "ETR","FE","PPL","AES","CNP","CMS","NI","WEC","LNT","EVRG",
        "AWK","MSEX","CWT","YORW",
        # ETFs / indices (cross-sectional context)
        "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","IEF","HYG","LQD",
        "VXX","SVXY","EEM","EFA","VEA","VWO",
    ]


def download_prices(tickers, start="2015-01-01",
                    end=datetime.today().strftime("%Y-%m-%d"), batch_size=50):
    """
    Download adjusted close prices for every ticker.
    Primary source : yfinance   (no auth, reliable)
    Fallback source: Stooq via pandas-datareader (no auth, good US coverage)

    Tickers that fail both sources are silently dropped.
    Results are cached to close_cache_full.parquet so subsequent runs are instant.
    """
    cache = os.path.join(OUT_DIR, "close_cache_full.parquet")
    if os.path.exists(cache):
        print("[2] Loading cached price data ...", flush=True)
        close = pd.read_parquet(cache)
        print(f"    {close.shape[1]} tickers x {close.shape[0]} days.")
        return close
    print(f"[2] Downloading {len(tickers)} tickers ...", flush=True)

    # ---- yfinance (primary) ------------------------------------------------
    frames  = []
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    succeeded = set()
    for bi, batch in enumerate(batches):
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False, threads=True)
            c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            # keep only columns with enough data
            ok = c.dropna(axis=1, thresh=int(0.7 * len(c))).columns.tolist()
            c  = c[ok]
            frames.append(c)
            succeeded.update(ok)
        except Exception as e:
            print(f"    yfinance batch {bi+1} failed: {e}", flush=True)
        print(f"    yfinance batch {bi+1}/{len(batches)} done", flush=True)

    # ---- pandas-datareader / Stooq (fallback for missed tickers) -----------
    missed = [t for t in tickers if t not in succeeded]
    if missed:
        try:
            import pandas_datareader.data as web
            print(f"    Stooq fallback for {len(missed)} tickers ...", flush=True)
            stooq_frames = []
            for t in missed:
                try:
                    raw = web.DataReader(t, "stooq", start=start, end=end)
                    if "Close" in raw.columns and len(raw) > 200:
                        s = raw["Close"].rename(t).sort_index()
                        stooq_frames.append(s)
                except Exception:
                    pass
            if stooq_frames:
                stooq_df = pd.concat(stooq_frames, axis=1)
                frames.append(stooq_df)
                print(f"    Stooq recovered {stooq_df.shape[1]} tickers.")
        except ImportError:
            print("    pandas-datareader not installed; skipping Stooq fallback.")

    if not frames:
        raise RuntimeError("No price data downloaded from any source.")

    close = pd.concat(frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    close = close.dropna(axis=1, thresh=int(0.7 * len(close)))
    close = close.ffill().dropna()
    close.to_parquet(cache)
    print(f"    Saved {close.shape[1]} tickers x {close.shape[0]} days.")
    return close


def download_volume(tickers, start="2015-01-01",
                    end=datetime.today().strftime("%Y-%m-%d"), batch_size=50):
    """
    Download daily volume for every ticker and cache to vol_cache_full.parquet.
    Volume features (relative volume, volume acceleration) are among the
    strongest short-term signals: high volume confirms price moves.
    """
    cache = os.path.join(OUT_DIR, "vol_cache_full.parquet")
    if os.path.exists(cache):
        print("[2c] Loading cached volume data ...", flush=True)
        vol = pd.read_parquet(cache)
        print(f"    {vol.shape[1]} tickers x {vol.shape[0]} days.")
        return vol
    print(f"[2c] Downloading volume for {len(tickers)} tickers ...", flush=True)
    frames = []
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    for bi, batch in enumerate(batches):
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False, threads=True)
            v = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw
            ok = v.dropna(axis=1, thresh=int(0.7 * len(v))).columns.tolist()
            frames.append(v[ok])
        except Exception as e:
            print(f"    volume batch {bi+1} failed: {e}", flush=True)
        print(f"    volume batch {bi+1}/{len(batches)} done", flush=True)
    if not frames:
        print("    No volume data downloaded; volume features will be disabled.")
        return None
    vol = pd.concat(frames, axis=1)
    vol = vol.loc[:, ~vol.columns.duplicated()]
    vol = vol.dropna(axis=1, thresh=int(0.7 * len(vol)))
    vol = vol.ffill().fillna(0)
    vol.to_parquet(cache)
    print(f"    Saved {vol.shape[1]} tickers x {vol.shape[0]} days.")
    return vol


def make_features(close: pd.DataFrame, sent_df: pd.DataFrame = None,
                  vol_df: pd.DataFrame = None):
    """
    Build a cross-sectional feature matrix with two key improvements over
    a naive time-series approach:

    1. Cross-sectional rank features
       Each raw feature value (e.g. ret1 = 0.003) is replaced with its
       percentile rank within the universe on that date (e.g. ret1 = 0.82).
       This removes the effect of market-wide moves so the model learns
       relative strength, which generalises far better across regimes.
       Ranking is done date-by-date so there is zero lookahead.

    2. Cross-sectional prediction target
       Instead of asking "did this stock go up?" (essentially a coin flip),
       the target is "was this stock in the top 30% of forward returns
       across the universe that day?"  The middle 40% of stocks whose
       returns are near zero are dropped because they carry almost no
       signal and add noise.  The result is a much cleaner binary
       classification: clear outperformers (1) vs clear underperformers (0).
    """
    print("[3] Engineering features ...", flush=True)
    all_X = []
    tickers = close.columns.tolist()

    for i, ticker in enumerate(tickers):
        c = close[ticker].dropna()
        if len(c) < 300:   # need enough history for 3-year features
            continue
        r1 = np.log(c / c.shift(1))
        feat = {}

        # ---- Short-term returns (1 d – 20 d) --------------------------------
        for lag in [1, 2, 3, 5, 10, 20]:
            feat[f"ret{lag}"]    = np.log(c / c.shift(lag))

        # ---- Medium-term returns (3 mo – 1 yr) ------------------------------
        # 6-month and 12-month momentum are the strongest documented
        # cross-sectional anomalies in the empirical finance literature.
        for lag in [60, 120, 252]:
            feat[f"ret{lag}"]    = np.log(c / c.shift(lag))

        # ---- Long-term return (3 yr) — reversal signal ----------------------
        # 3–5 year prior winners tend to mean-revert (DeBondt & Thaler 1985)
        feat["ret756"]           = np.log(c / c.shift(756))

        # ---- Realised volatility at every scale -----------------------------
        for w in [5, 10, 20, 60, 120, 252]:
            feat[f"vol{w}"]      = r1.rolling(w).std()

        # ---- Price momentum (percentage move) at every scale ----------------
        for m in [5, 10, 20, 60, 120, 252]:
            feat[f"mom{m}"]      = (c - c.shift(m)) / c.shift(m)

        # ---- Short-term volatility spike: vol5 / vol20 ----------------------
        feat["vol_ratio"]        = (r1.rolling(5).std() /
                                    (r1.rolling(20).std() + 1e-9))

        # ---- Moving-average ratios ------------------------------------------
        # Price vs 50-day MA (trend following proxy)
        feat["ma50_ratio"]       = c / c.rolling(50).mean() - 1
        # Price vs 200-day MA (long-term trend)
        feat["ma200_ratio"]      = c / c.rolling(200).mean() - 1
        # Golden/death cross: 50-day MA relative to 200-day MA
        feat["ma50_200_cross"]   = (c.rolling(50).mean() /
                                    (c.rolling(200).mean() + 1e-9) - 1)

        # ---- Trend acceleration ---------------------------------------------
        # Recent 5-day return minus the prior 5–20 day return.
        # Positive = accelerating momentum; negative = decelerating.
        feat["ret_accel"]        = (np.log(c / c.shift(5)) -
                                    np.log(c.shift(5) / c.shift(20)))

        # ---- RSI-14 ---------------------------------------------------------
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        feat["rsi14"]            = 100 - 100 / (1 + gain / (loss + 1e-9))

        # ---- Distance from rolling highs / lows -----------------------------
        feat["dist52h"]          = c / c.rolling(252).max()  - 1   # 1-yr high
        feat["dist52l"]          = c / c.rolling(252).min()  - 1   # 1-yr low
        feat["dist3yh"]          = c / c.rolling(756).max()  - 1   # 3-yr high
        feat["dist3yl"]          = c / c.rolling(756).min()  - 1   # 3-yr low

        # ---- Risk-adjusted return (Sharpe-style features) -------------------
        # Normalising return by its own realised volatility removes the scale
        # difference between calm and turbulent regimes, giving the model a
        # cleaner signal about whether recent moves were unusually large.
        for w in [5, 20, 60, 252]:
            feat[f"sharpe{w}"]   = feat[f"ret{w}"] / (feat[f"vol{w}"] + 1e-9)

        # ---- Volume features ------------------------------------------------
        # Relative volume (today's volume vs rolling average) is one of the
        # strongest short-term predictors: large volume tends to confirm
        # price breakouts and momentum continuation.
        if vol_df is not None and ticker in vol_df.columns:
            v      = vol_df[ticker].reindex(c.index).fillna(0.0)
            v_ma5  = v.rolling(5,  min_periods=5).mean()  + 1e-9
            v_ma20 = v.rolling(20, min_periods=20).mean() + 1e-9
            v_ma60 = v.rolling(60, min_periods=60).mean() + 1e-9
            feat["rel_vol5"]  = v / v_ma20        # short-term spike vs 20-day avg
            feat["rel_vol20"] = v / v_ma60        # medium volume vs 60-day avg
            feat["vol_accel"] = (v / v_ma5) / (v_ma5 / v_ma20 + 1e-9)  # acceleration
        else:
            feat["rel_vol5"]  = 1.0
            feat["rel_vol20"] = 1.0
            feat["vol_accel"] = 1.0

        # Sentiment score for this ticker (raw VADER compound, pre-fetched).
        if sent_df is not None and ticker in sent_df.columns:
            feat["_sent"] = sent_df[ticker].reindex(c.index).fillna(0.0)
        else:
            feat["_sent"] = 0.0

        # Forward 1-day log-return (target).
        # With a 1-day horizon, adjacent rows from the same ticker have
        # independent labels: r[t+1] and r[t+2] share no days of outcome.
        # This avoids the label autocorrelation that plagues multi-day
        # horizons and lets the model train on all ~220K cross-sectional rows.
        feat["_fwd"]        = np.log(c.shift(-1) / c)

        df = pd.DataFrame(feat, index=c.index).dropna()
        all_X.append(df)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(tickers)} tickers ...", flush=True)

    # Stack all tickers (date is the index, many rows share the same date)
    X_full = pd.concat(all_X).sort_index()

    # Separate the raw forward return and sentiment from the feature columns
    fwd_raw   = X_full.pop("_fwd")
    sent_raw  = X_full.pop("_sent")
    feat_names = X_full.columns.tolist()

    # --- Cross-sectional percentile rank for every feature -------------------
    print("    Computing cross-sectional ranks ...", flush=True)
    X_ranked = X_full.groupby(X_full.index).rank(pct=True)

    # Rank sentiment cross-sectionally and append as the LAST column so
    # UnifiedCourseNetwork (use_sent=True) can cleanly extract it with X[:, -1:]
    if sent_df is not None:
        sent_ranked = sent_raw.groupby(sent_raw.index).rank(pct=True)
        X_ranked["sent_rank"] = sent_ranked
        feat_names_out = feat_names + ["sent_rank"]
        has_sent = True
    else:
        feat_names_out = feat_names
        has_sent = False

    # --- Cross-sectional target: top 20% vs bottom 20% ----------------------
    # Tighter bands (20% vs the previous 30%) create cleaner separation:
    # only the strongest outperformers and underperformers are labelled,
    # reducing borderline cases near the threshold that add noise.
    # The middle 60% are dropped.
    print("    Building cross-sectional target (top 20% vs bottom 20%) ...",
          flush=True)
    fwd_rank = fwd_raw.groupby(fwd_raw.index).rank(pct=True)
    y = pd.Series(np.nan, index=fwd_raw.index)
    y[fwd_rank >= 0.80] = 1   # clear outperformers  (top 20%)
    y[fwd_rank <= 0.20] = 0   # clear underperformers (bottom 20%)
    keep     = y.notna()
    X_final  = X_ranked[keep]
    y_final  = y[keep].astype(int)

    n_kept = int(keep.sum())
    print(f"    Feature matrix: {X_final.shape}  |  base rate: {y_final.mean():.4f}"
          f"  (kept {n_kept:,} / {len(fwd_raw):,} rows, dropped middle 60%)",
          flush=True)
    return X_final, y_final, feat_names_out, has_sent

# ===========================================================================
# Walk-forward evaluation
# ===========================================================================

def walk_forward(X_np, y_np, dates, model_cls, model_kwargs, model_name,
                 n_splits=5, scale=True):
    unique_dates = np.sort(np.unique(dates))
    fold_size    = len(unique_dates) // (n_splits + 1)
    accs, aucs   = [], []
    print(f"\n  [{model_name}]", flush=True)
    for fold in range(n_splits):
        tr_end = unique_dates[(fold + 1) * fold_size]
        te_end = unique_dates[min((fold + 2) * fold_size, len(unique_dates) - 1)]
        tr_m   = dates < tr_end
        te_m   = (dates >= tr_end) & (dates < te_end)
        X_tr, X_te = X_np[tr_m], X_np[te_m]
        y_tr, y_te = y_np[tr_m], y_np[te_m]
        if len(X_tr) < 500 or len(X_te) < 50:
            continue
        if scale:
            mu = X_tr.mean(axis=0);  sd = X_tr.std(axis=0) + 1e-9
            X_tr = (X_tr - mu) / sd;  X_te = (X_te - mu) / sd
        t0    = time.time()
        model = model_cls(**model_kwargs)
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        pred  = model.predict(X_te)
        prob  = model.predict_proba(X_te)
        prob1 = prob[:, 1] if prob.ndim == 2 else prob
        acc   = accuracy(y_te, pred)
        auc   = roc_auc(y_te, prob1)
        accs.append(acc);  aucs.append(auc)
        print(f"    fold {fold+1}/{n_splits}  "
              f"train={tr_m.sum():,}  test={te_m.sum():,}  "
              f"acc={acc:.4f}  auc={auc:.4f}  time={elapsed:.1f}s",
              flush=True)
    m_acc = np.mean(accs) if accs else 0
    m_auc = np.mean(aucs) if aucs else 0
    print(f"  [{model_name}] mean acc={m_acc:.4f}  mean auc={m_auc:.4f}")
    return m_acc, m_auc

# ===========================================================================
# Final retrain + plots
# ===========================================================================

def retrain_and_plot(X_np, y_np, dates, feat_names, model, model_name, scale=True):
    unique_dates = np.sort(np.unique(dates))
    split_dt     = unique_dates[int(0.85 * len(unique_dates))]
    tr_m = dates < split_dt
    te_m = dates >= split_dt
    X_tr, X_te = X_np[tr_m], X_np[te_m]
    y_tr, y_te = y_np[tr_m], y_np[te_m]
    mu = X_tr.mean(0);  sd = X_tr.std(0) + 1e-9
    X_tr_s = (X_tr - mu) / sd
    X_te_s = (X_te - mu)  / sd
    print(f"\n  Final retrain [{model_name}]  "
          f"train={tr_m.sum():,}  test={te_m.sum():,}", flush=True)
    model.fit(X_tr_s, y_tr)
    pred  = model.predict(X_te_s)
    prob  = model.predict_proba(X_te_s)
    prob1 = prob[:, 1] if prob.ndim == 2 else prob
    acc   = accuracy(y_te, pred)
    auc   = roc_auc(y_te, prob1)
    print(f"  Test  acc={acc:.4f}  auc={auc:.4f}")

    # Loss curve (if model tracks it)
    if hasattr(model, "loss_history") and len(model.loss_history) > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(model.loss_history, "b-", linewidth=1.5)
        ax.set_title(f"{model_name} — Training Loss (NLL / Cross-Entropy)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR,
                    f"loss_curve_{model_name.replace(' ','_')}.png"), dpi=150)
        print(f"  Saved loss curve.")

    return acc, auc, mu, sd, y_te, prob1


# ===========================================================================
# LightGBM baseline (production comparison)
# ===========================================================================

def lgbm_baseline(X_np, y_np, dates):
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not installed, skipping.")
        return
    unique_dates = np.sort(np.unique(dates))
    split_dt     = unique_dates[int(0.85 * len(unique_dates))]
    tr_m = dates < split_dt;  te_m = dates >= split_dt
    X_tr, X_te = X_np[tr_m], X_np[te_m]
    y_tr, y_te = y_np[tr_m], y_np[te_m]
    mu = X_tr.mean(0);  sd = X_tr.std(0) + 1e-9
    X_tr_s = (X_tr - mu) / sd;  X_te_s = (X_te - mu) / sd

    dtrain = lgb.Dataset(X_tr_s, label=y_tr)
    dvalid = lgb.Dataset(X_te_s, label=y_te, reference=dtrain)
    params = dict(objective="binary", metric="binary_logloss",
                  num_leaves=63, max_depth=7, learning_rate=0.02,
                  min_child_samples=100, subsample=0.7, subsample_freq=1,
                  colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                  is_unbalance=True, device="gpu", seed=SEED, verbosity=-1)
    print("\n  [LightGBM-GPU] Final retrain — printing every 25 rounds ...\n",
          flush=True)
    booster = lgb.train(params, dtrain, num_boost_round=1000,
                        valid_sets=[dtrain, dvalid],
                        valid_names=["train", "valid"],
                        callbacks=[lgb.early_stopping(75, verbose=False),
                                   lgb.log_evaluation(25)])
    prob = booster.predict(X_te_s)
    pred = (prob > 0.5).astype(int)
    acc  = accuracy(y_te, pred)
    auc  = roc_auc(y_te, prob)
    print(f"\n  [LightGBM-GPU]  acc={acc:.4f}  auc={auc:.4f}")
    booster.save_model(os.path.join(OUT_DIR, "lgbm_full.txt"))
    return acc, auc, y_te, prob

# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print(" COMP 653 — Stock Direction Prediction (Course-Aligned Build)")
    print("=" * 65, flush=True)

    # --- Data ---------------------------------------------------------------
    tickers = get_sp500_tickers()
    close   = download_prices(tickers)

    # Fetch VADER news sentiment scores (cached after first run)
    print("\n[2b] Fetching news sentiment (VADER NLP) ...", flush=True)
    sent_df = fetch_sentiment(tickers, close.index)

    # Download volume data (separate parquet cache)
    vol_df = download_volume(tickers)

    X_df, y_df, feat_names, has_sent = make_features(close, sent_df, vol_df)

    dates  = X_df.index.values
    X_all  = X_df.values.astype(np.float64)
    y_all  = y_df.values.astype(int)

    # --- MI Feature Selection (Module 2) ------------------------------------
    # If sentiment is present it occupies the last column; keep it out of
    # MI selection (it has very low MI on short history) but always append
    # it back so Branch D can access it.
    price_feat_names = [f for f in feat_names if f != "sent_rank"]
    price_idx        = [feat_names.index(f) for f in price_feat_names]
    X_price_only     = X_all[:, price_idx]

    # Use all features — with 39 features and 200K+ rows there is no
    # overfitting risk from keeping the full set.  MI ranking is printed
    # for interpretability (Module 2) but nothing is dropped.
    selected    = select_features_by_mi(X_price_only, y_all, price_feat_names,
                                        k=len(price_feat_names))
    sel_idx     = [price_feat_names.index(f) for f in selected]
    X_sel_price = X_price_only[:, sel_idx]

    # Append sentiment column back as the last feature for Branch D
    if has_sent:
        sent_col = X_all[:, feat_names.index("sent_rank")].reshape(-1, 1)
        X_sel    = np.hstack([X_sel_price, sent_col])
        sent_label = " + VADER sentiment"
    else:
        X_sel      = X_sel_price
        sent_label = ""
    print(f"\n    Using {len(selected)} MI-selected price features{sent_label}.")

    print("\n[3b] EDA figures — correlation, PCA, class balance ...", flush=True)
    make_eda_figures(X_sel, y_all, selected + (["sent_rank"] if has_sent else []))

    # --- Walk-forward CV on early 25% of dates (fast sanity check) ---------
    unique_dates = np.sort(np.unique(dates))
    cut     = unique_dates[int(0.25 * len(unique_dates))]
    sub_m   = dates <= cut
    X_sub, y_sub, d_sub = X_sel[sub_m], y_all[sub_m], dates[sub_m]
    print(f"\n[4] Walk-forward CV — UnifiedCourseNetwork "
          f"on {sub_m.sum():,} sample rows ...", flush=True)

    cv_acc, cv_auc = walk_forward(
        X_sub, y_sub, d_sub,
        UnifiedCourseNetwork,
        {"hidden_sizes": (64, 32), "lr": 0.001, "epochs": 60,
         "batch_size": 1024, "beta1": 0.9, "beta2": 0.999,
         "dropout_rate": 0.4, "meta_dropout": 0.2, "val_frac": 0.15,
         "verbose": 0, "use_sent": has_sent,
         "grad_clip": 1.0, "noise_frac": 0.02, "warmup_epochs": 5,
         "use_fgsm": True, "fgsm_eps": 0.01, "pgd_steps": 5},
        "UnifiedCourseNetwork (LR+NB+MLP+Sent branches, Adam)" if has_sent
        else "UnifiedCourseNetwork (LR+NB+MLP branches, Adam)",
        n_splits=3)

    pd.DataFrame([{"acc": cv_acc, "auc": cv_auc}],
                 index=["UnifiedCourseNetwork"]).to_csv(
        os.path.join(OUT_DIR, "cv_results_unified.csv"))

    # --- Full retrain on entire dataset with full epoch count ---------------
    print("\n[5] Full retrain — UnifiedCourseNetwork on entire dataset ...",
          flush=True)
    unified = UnifiedCourseNetwork(
        hidden_sizes=(256, 128, 64), lr=1e-3, epochs=3000,
        batch_size=2048, lam=3e-4, beta1=0.9, beta2=0.999,
        dropout_rate=0.4, meta_dropout=0.2, val_frac=0.15, verbose=20,
        patience=150, use_sent=has_sent,
        grad_clip=1.0, noise_frac=0.02, warmup_epochs=5,
        use_fgsm=True, fgsm_eps=0.01, pgd_steps=5)
    final_acc, final_auc, mu_f, sd_f, y_te_u, prob_u = retrain_and_plot(
        X_sel, y_all, dates, selected, unified,
        "UnifiedCourseNetwork_Adam_Sent" if has_sent else "UnifiedCourseNetwork_Adam")

    # --- LightGBM GPU baseline (professional comparison) -------------------
    print("\n[6] LightGBM-GPU production baseline ...", flush=True)
    lgbm_res = lgbm_baseline(X_sel, y_all, dates)

    # --- Summary table ------------------------------------------------------
    label = "UnifiedCourseNetwork (LR+NB+MLP+Sent)" if has_sent \
            else "UnifiedCourseNetwork (LR+NB+MLP)"
    maj_acc, maj_auc = majority_class_baseline(y_te_u)
    rows = {
        label: {"acc": final_acc, "auc": final_auc},
        "MajorityClass (always-up)": {"acc": maj_acc, "auc": maj_auc},
    }
    if lgbm_res:
        rows["LightGBM-GPU (baseline)"] = {"acc": lgbm_res[0], "auc": lgbm_res[1]}
    df_summary = pd.DataFrame(rows).T
    print("\n=== Final Model Comparison ===")
    print(df_summary.to_string())
    df_summary.to_csv(os.path.join(OUT_DIR, "final_results_unified.csv"))

    # --- Statistical significance of the UCN vs LightGBM AUC gap ------------
    print("\n[7] Significance test — paired bootstrap on the AUC gap ...",
          flush=True)
    print(f"    Test-set class balance: Up={float(np.mean(y_te_u)):.4f}  "
          f"majority-class accuracy={maj_acc:.4f}", flush=True)
    if lgbm_res:
        obs, lo, hi, p_val = paired_bootstrap_auc_diff(y_te_u, prob_u, lgbm_res[3])
        verdict = "significant (CI excludes 0)" if lo > 0 or hi < 0 \
                  else "not significant (CI includes 0)"
        print(f"    UCN - LightGBM AUC gap = {obs:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  p={p_val:.3f}  -> {verdict}",
              flush=True)
        pd.DataFrame([{
            "auc_gap_ucn_minus_lgbm": obs, "ci_low": lo, "ci_high": hi,
            "bootstrap_p_value": p_val, "majority_class_acc": maj_acc,
            "test_up_frac": float(np.mean(y_te_u)),
        }]).to_csv(os.path.join(OUT_DIR, "significance_test.csv"), index=False)
        print("    Saved significance_test.csv")
    else:
        print("    LightGBM unavailable — skipping paired test.", flush=True)

    print("\nAll done. Artifacts saved to", OUT_DIR)
