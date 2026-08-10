"""
RNN Branch (Branch E) from scratch — implements the professor's equation:

    h_t = phi(omega_x @ x_t + omega_h @ h_{t-1})

Processes a sequence of L daily feature snapshots for each sample,
capturing the TRAJECTORY of momentum rather than a single static snapshot.
This is the key architectural addition over Branches A-D which are memoryless.

Backpropagation through time (BPTT) is implemented explicitly so every
gradient follows from first principles (Lec 5-5 chain rule extended to
recurrent connections).
"""
import numpy as np
import math


class ElmanRNNBranch:
    """
    Single-layer Elman RNN operating on a lookback sequence of feature vectors.

    Input  : X_seq  shape (N, L, d_in)   — L days of d_in-dimensional features
    Output : h_T    shape (N, d_hidden)  — final hidden state fed to meta-layer

    Forward:
        h_0 = zeros(N, d_hidden)
        for t = 0 .. L-1:
            h_t = tanh(X_seq[:,t,:] @ W_x + h_{t-1} @ W_h + b_h)

    Parameters
    ----------
    d_in     : number of input features per time step (same as price features)
    d_hidden : RNN hidden state dimension
    seed     : random seed for He initialisation
    """

    def __init__(self, d_in: int, d_hidden: int = 32, seed: int = 42):
        self.d_in     = d_in
        self.d_hidden = d_hidden
        rng = np.random.default_rng(seed)

        # He initialisation for tanh: scale = sqrt(1 / fan_in)
        self.params = {
            "W_x": rng.standard_normal((d_in,     d_hidden)) * math.sqrt(1.0 / d_in),
            "W_h": rng.standard_normal((d_hidden, d_hidden)) * math.sqrt(1.0 / d_hidden),
            "b_h": np.zeros(d_hidden),
        }
        # BPTT cache populated by forward()
        self._cache = {}

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(self, X_seq: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        X_seq : (N, L, d_in) — sequence of L snapshots per sample

        Returns
        -------
        h_T : (N, d_hidden) — final hidden state
        """
        N, L, _ = X_seq.shape
        W_x = self.params["W_x"]
        W_h = self.params["W_h"]
        b_h = self.params["b_h"]

        h = np.zeros((N, self.d_hidden))
        hs  = [h]                  # h_0 .. h_T, shape list of (N, d_hidden)
        zs  = []                   # pre-activations z_t = W_x x_t + W_h h_{t-1} + b

        for t in range(L):
            z = X_seq[:, t, :] @ W_x + h @ W_h + b_h
            h = np.tanh(z)
            hs.append(h)
            zs.append(z)

        self._cache = {"X_seq": X_seq, "hs": hs, "zs": zs, "L": L}
        return h                   # final hidden state h_T

    # ── Backward pass (BPTT) ─────────────────────────────────────────────

    def backward(self, d_hT: np.ndarray) -> dict:
        """
        Backpropagation through time.

        Parameters
        ----------
        d_hT : (N, d_hidden) — gradient of loss w.r.t. final hidden state h_T

        Returns
        -------
        grads : dict with keys W_x, W_h, b_h
                plus 'dX_seq' of shape (N, L, d_in) for further backprop
        """
        X_seq = self._cache["X_seq"]
        hs    = self._cache["hs"]
        zs    = self._cache["zs"]
        L     = self._cache["L"]
        N     = X_seq.shape[0]

        W_x = self.params["W_x"]
        W_h = self.params["W_h"]

        dW_x = np.zeros_like(W_x)
        dW_h = np.zeros_like(W_h)
        db_h = np.zeros_like(self.params["b_h"])
        dX_seq = np.zeros_like(X_seq)

        d_h = d_hT                 # gradient flowing back through time

        for t in reversed(range(L)):
            # Gradient through tanh: d_z = d_h * (1 - tanh^2(z_t))
            d_z = d_h * (1.0 - np.tanh(zs[t]) ** 2)

            dW_x  += X_seq[:, t, :].T @ d_z
            dW_h  += hs[t].T        @ d_z    # hs[t] = h_{t-1}
            db_h  += d_z.sum(axis=0)
            dX_seq[:, t, :] = d_z @ W_x.T

            d_h = d_z @ W_h.T      # gradient to h_{t-1}

        return {
            "W_x": dW_x / N,
            "W_h": dW_h / N,
            "b_h": db_h / N,
            "dX_seq": dX_seq,
        }

    # ── Weight access ─────────────────────────────────────────────────────

    def param_keys(self):
        return list(self.params.keys())

    def get_params(self):
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, d: dict):
        for k, v in d.items():
            if k in self.params:
                self.params[k] = v.copy()
