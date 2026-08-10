"""
LSTM Branch E — from scratch in NumPy.
Implements the professor's recurrent equation extended with gating:

    Elman RNN:  h_t = phi(W_x @ x_t + W_h @ h_{t-1})
    LSTM adds:  forget gate, input gate, output gate, cell state

Gates prevent vanishing gradients over 20-50 day lookback windows,
letting the model remember multi-week momentum trajectories.

Architecture
------------
Input  : (N, T, d)  — N samples, T timesteps, d features
Output : (N, H)     — final hidden state h_T for each sample

Integrated as Branch E in UnifiedCourseNetwork.
Meta-layer input grows from 70 to (70 + H).

Usage in training loop
----------------------
The forward pass processes the full (N, T, d) sequence each batch.
Backpropagation Through Time (BPTT) propagates gradients back T steps.
"""
import math
import numpy as np


class LSTMScratch:
    """
    Single-layer LSTM implemented entirely in NumPy.

    Parameters
    ----------
    input_size  : d — number of features per timestep
    hidden_size : H — size of h_t and c_t
    seed        : random seed for weight initialisation
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        self.d = input_size
        self.H = hidden_size
        rng    = np.random.default_rng(seed)

        # Each gate has weight matrices W (input) and U (hidden) and bias b.
        # All four gates are stacked for efficiency:
        #   index 0:H   = forget  gate  f
        #   index H:2H  = input   gate  i
        #   index 2H:3H = cell    gate  g  (tanh, not sigmoid)
        #   index 3H:4H = output  gate  o
        s = math.sqrt(2.0 / (input_size + hidden_size))  # Xavier
        self.params = {
            "W": rng.standard_normal((input_size,  4 * hidden_size)) * s,
            "U": rng.standard_normal((hidden_size, 4 * hidden_size)) * s,
            "b": np.zeros(4 * hidden_size),
        }

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray, training: bool = True) -> dict:
        """
        Forward pass through the full sequence.

        Parameters
        ----------
        X : (N, T, d)

        Returns
        -------
        cache : dict with keys
          'h'      : (N, T, H)  hidden states at every timestep
          'c'      : (N, T, H)  cell states
          'gates'  : (N, T, 4H) pre-activation gate values (for BPTT)
          'h_T'    : (N, H)     final hidden state  ← feed to meta-layer
        """
        N, T, d = X.shape
        H       = self.H
        W, U, b = self.params["W"], self.params["U"], self.params["b"]

        h = np.zeros((N, T + 1, H))   # h[:, 0, :] = h_0 = zeros
        c = np.zeros((N, T + 1, H))   # c[:, 0, :] = c_0 = zeros
        gates_pre = np.zeros((N, T, 4 * H))

        for t in range(T):
            z        = X[:, t, :] @ W + h[:, t, :] @ U + b   # (N, 4H)
            gates_pre[:, t, :] = z

            f = self._sigmoid(z[:,    :H])     # forget
            i = self._sigmoid(z[:,   H:2*H])   # input
            g = np.tanh(z[:,  2*H:3*H])        # cell candidate
            o = self._sigmoid(z[:, 3*H:])      # output

            c[:, t+1, :] = f * c[:, t, :] + i * g
            h[:, t+1, :] = o * np.tanh(c[:, t+1, :])

        return {
            "X":      X,
            "h":      h,
            "c":      c,
            "gates":  gates_pre,
            "h_T":    h[:, T, :],    # (N, H)
        }

    # ── Backward (BPTT) ──────────────────────────────────────────────────

    def backward(self, cache: dict, d_h_T: np.ndarray) -> dict:
        """
        Backpropagation Through Time.

        Parameters
        ----------
        cache  : output of forward()
        d_h_T  : (N, H)  gradient from the meta-layer w.r.t. h_T

        Returns
        -------
        grads  : dict with keys 'dW', 'dU', 'db'
        dX     : (N, T, d)  gradient w.r.t. input sequence
        """
        X, h, c, gates_pre = cache["X"], cache["h"], cache["c"], cache["gates"]
        N, T, d = X.shape
        H       = self.H
        W, U    = self.params["W"], self.params["U"]

        dW = np.zeros_like(self.params["W"])
        dU = np.zeros_like(self.params["U"])
        db = np.zeros_like(self.params["b"])
        dX = np.zeros_like(X)

        d_h_next = d_h_T.copy()   # gradient of loss w.r.t. h_T
        d_c_next = np.zeros((N, H))

        for t in reversed(range(T)):
            z = gates_pre[:, t, :]
            f = self._sigmoid(z[:,    :H])
            i = self._sigmoid(z[:,   H:2*H])
            g = np.tanh(z[:,  2*H:3*H])
            o = self._sigmoid(z[:, 3*H:])

            tanh_c_next = np.tanh(c[:, t+1, :])

            # Gradient through h_{t+1} = o * tanh(c_{t+1})
            d_o      = d_h_next * tanh_c_next
            d_c_curr = d_h_next * o * (1.0 - tanh_c_next**2) + d_c_next

            # Gradient through c_{t+1} = f*c_t + i*g
            d_f = d_c_curr * c[:, t, :]
            d_i = d_c_curr * g
            d_g = d_c_curr * i
            d_c_prev = d_c_curr * f

            # Gate gradients through activation functions
            dz_f = d_f * f * (1 - f)
            dz_i = d_i * i * (1 - i)
            dz_g = d_g * (1 - g**2)
            dz_o = d_o * o * (1 - o)
            dz   = np.concatenate([dz_f, dz_i, dz_g, dz_o], axis=1)  # (N, 4H)

            dW       += X[:, t, :].T @ dz             # (d, 4H)
            dU       += h[:, t, :].T @ dz             # (H, 4H)
            db       += dz.sum(axis=0)                # (4H,)
            dX[:, t, :] = dz @ W.T                   # (N, d)
            d_h_next = dz @ U.T                       # (N, H)
            d_c_next = d_c_prev

        return {"dW": dW, "dU": dU, "db": db}, dX

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))


def build_sequences(
    X_df: "pd.DataFrame",
    ticker_col: str = "_ticker",
    lookback: int = 20,
) -> "tuple":
    """
    Build a (N_valid, lookback, d) sequence tensor from the cross-sectional
    feature matrix returned by make_features().

    Each row in X_df represents one (ticker, date) pair.  For each row that
    has at least `lookback` prior rows from the SAME TICKER, a sequence of
    the past `lookback` feature vectors is created.  Rows without enough
    history are excluded and indicated by mask=False.

    Parameters
    ----------
    X_df     : DataFrame with date index; must contain a ``_ticker`` column.
    lookback : number of past days to include (T in the LSTM input).

    Returns
    -------
    seqs  : float64 array  (N_valid, lookback, d)
    mask  : bool array     (len(X_df),)  — True for rows included in seqs.
    """
    import pandas as pd

    feat_cols = [c for c in X_df.columns if c != ticker_col]
    d    = len(feat_cols)
    N    = len(X_df)
    mask = np.zeros(N, dtype=bool)

    if ticker_col not in X_df.columns:
        print("  build_sequences: no ticker column — LSTM disabled.", flush=True)
        return np.empty((0, lookback, d)), mask

    seqs_list = []

    # Use a reset index so we have stable integer positions 0..N-1
    df_pos = X_df.reset_index()          # adds original date as a column
    date_col = df_pos.columns[0]         # first col is the date index

    for ticker_name, grp in df_pos.groupby(ticker_col):
        grp_s   = grp.sort_values(date_col)           # sort by date
        pos     = grp_s.index.values                   # positions in original df
        vals    = grp_s[feat_cols].values.astype(np.float64)

        for j in range(lookback, len(grp_s)):
            seq = vals[j - lookback:j]                 # (lookback, d)
            seqs_list.append(seq)
            mask[pos[j]] = True

    if not seqs_list:
        return np.empty((0, lookback, d)), mask

    seqs = np.stack(seqs_list, axis=0).astype(np.float64)
    print(f"  Sequence tensor: {seqs.shape}  "
          f"(valid={mask.sum():,} / {N:,} rows  lookback={lookback})",
          flush=True)
    return seqs, mask
