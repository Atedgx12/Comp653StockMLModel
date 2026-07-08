"""Shared math utilities used by all model modules."""
from .backend import xp as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def softmax(Z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def nll_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy / negative log-likelihood."""
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def cross_entropy_softmax(Y_hat: np.ndarray, Y_oh: np.ndarray,
                          eps: float = 1e-12) -> float:
    """Categorical cross-entropy with softmax output."""
    return -np.mean(np.sum(Y_oh * np.log(Y_hat + eps), axis=1))
