"""
Extended technical indicators drawn from NomadicStockBot methodology.
All implemented in pure NumPy / pandas — no TA-Lib dependency.

Adds the following beyond the base 39 UCN features:
  Oscillators : CCI, Williams %R, CMO, Awesome Oscillator, TRIX
  Volume      : OBV, Chaikin Money Flow, Money Flow Index, Klinger
  Trend       : ADX, +DI spread, Ichimoku cloud distance
  VWAP        : Daily rolling VWAP deviation
  Breakout    : Donchian channel breakout, BB squeeze release
  Temporal    : RSI rate-of-change, MACD-hist ROC (for LSTM momentum detection)

Usage
-----
from ucn.data.nomadic_features import add_nomadic_features

# After computing the base feature dict in make_features():
feat = add_nomadic_features(c, feat)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── Oscillators ──────────────────────────────────────────────────────────────

def cci(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 20) -> pd.Series:
    """Commodity Channel Index — deviation from typical-price SMA."""
    tp  = (high + low + close) / 3.0
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-9)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               n: int = 14) -> pd.Series:
    """Williams %R — position within the N-day range [-100, 0]."""
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return -100 * (hh - close) / (hh - ll + 1e-9)


def cmo(close: pd.Series, n: int = 14) -> pd.Series:
    """Chande Momentum Oscillator — net momentum divided by total movement."""
    diff  = close.diff()
    up    = diff.clip(lower=0).rolling(n).sum()
    down  = (-diff).clip(lower=0).rolling(n).sum()
    total = up + down
    return 100.0 * (up - down) / (total + 1e-9)


def awesome_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    """Awesome Oscillator = SMA5(midpoint) - SMA34(midpoint)."""
    mid = (high + low) / 2.0
    return mid.rolling(5).mean() - mid.rolling(34).mean()


def trix(close: pd.Series, n: int = 15) -> pd.Series:
    """TRIX — rate of change of triple-smoothed EMA."""
    e1  = close.ewm(span=n, adjust=False).mean()
    e2  = e1.ewm(span=n, adjust=False).mean()
    e3  = e2.ewm(span=n, adjust=False).mean()
    return e3.pct_change() * 100.0


# ── Volume oscillators ────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative signed volume."""
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).cumsum()


def chaikin_money_flow(high: pd.Series, low: pd.Series,
                       close: pd.Series, volume: pd.Series,
                       n: int = 20) -> pd.Series:
    """CMF = sum(money_flow_volume, n) / sum(volume, n)."""
    clv = ((close - low) - (high - close)) / (high - low + 1e-9)
    mfv = clv * volume
    return mfv.rolling(n).sum() / (volume.rolling(n).sum() + 1e-9)


def money_flow_index(high: pd.Series, low: pd.Series,
                     close: pd.Series, volume: pd.Series,
                     n: int = 14) -> pd.Series:
    """MFI = volume-weighted RSI using typical price."""
    tp  = (high + low + close) / 3.0
    raw = tp * volume
    pos = raw.where(tp > tp.shift(1), 0.0)
    neg = raw.where(tp < tp.shift(1), 0.0)
    rs  = pos.rolling(n).sum() / (neg.rolling(n).sum() + 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)


def klinger_oscillator(high: pd.Series, low: pd.Series,
                       close: pd.Series, volume: pd.Series,
                       short: int = 34, long: int = 55) -> pd.Series:
    """Klinger Volume Oscillator — EMA34 minus EMA55 of signed volume force."""
    tp   = (high + low + close) / 3.0
    sv   = volume * np.where(tp > tp.shift(1), 1.0, -1.0)
    sv_s = pd.Series(sv, index=close.index)
    return (sv_s.ewm(span=short, adjust=False).mean()
            - sv_s.ewm(span=long,  adjust=False).mean())


# ── Trend ────────────────────────────────────────────────────────────────────

def adx_di(high: pd.Series, low: pd.Series, close: pd.Series,
           n: int = 14) -> pd.DataFrame:
    """ADX, +DI, -DI using Wilder smoothing."""
    tr  = pd.concat([high - low,
                     (high - close.shift(1)).abs(),
                     (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    up  = high.diff();  down = -low.diff()
    plus_dm  = up.where((up > down) & (up > 0),  0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr_w   = tr.ewm(alpha=1/n,       adjust=False).mean()
    plus_di  = 100.0 * plus_dm.ewm(alpha=1/n,  adjust=False).mean() / (atr_w + 1e-9)
    minus_di = 100.0 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / (atr_w + 1e-9)
    dx       = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx_val  = dx.ewm(alpha=1/n, adjust=False).mean()
    return pd.DataFrame({"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di})


def ichimoku_cloud_distance(high: pd.Series, low: pd.Series,
                             close: pd.Series) -> pd.Series:
    """
    Distance of close from Ichimoku cloud midpoint (normalised by close).
    Positive = above cloud (bullish), negative = below cloud (bearish).
    """
    conversion = (high.rolling(9).max()  + low.rolling(9).min())  / 2.0
    base       = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    span_a     = ((conversion + base) / 2.0).shift(26)
    span_b     = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).shift(26)
    cloud_mid  = (span_a + span_b) / 2.0
    return (close - cloud_mid) / (close + 1e-9)


# ── VWAP ─────────────────────────────────────────────────────────────────────

def rolling_vwap(high: pd.Series, low: pd.Series,
                 close: pd.Series, volume: pd.Series,
                 n: int = 20) -> pd.Series:
    """
    Rolling daily VWAP approximation (n-bar lookback).
    True intraday VWAP requires tick data; this uses typical price as proxy.
    """
    tp  = (high + low + close) / 3.0
    return (tp * volume).rolling(n).sum() / (volume.rolling(n).sum() + 1e-9)


# ── Donchian breakout ────────────────────────────────────────────────────────

def donchian_breakout(close: pd.Series, high: pd.Series,
                      n: int = 20) -> pd.Series:
    """
    +1 if close broke above Donchian upper band today,
    -1 if broke below lower band, 0 otherwise.
    """
    upper = high.rolling(n).max().shift(1)
    lower = close.rolling(n).min().shift(1)
    sig   = np.zeros(len(close))
    sig[close.values > upper.values] =  1.0
    sig[close.values < lower.values] = -1.0
    return pd.Series(sig, index=close.index)


# ── BB squeeze release (from Nomadic breakout.py methodology) ────────────────

def bb_squeeze_release(close: pd.Series, high: pd.Series,
                       low: pd.Series, n: int = 20) -> pd.Series:
    """
    1.0 when Bollinger Band width expands after being compressed below Keltner width.
    Signals a volatility squeeze releasing — often precedes a strong directional move.
    """
    std     = close.rolling(n).std()
    bb_w    = 4.0 * std / close
    kc_atr  = (high - low).rolling(n).mean()
    kc_w    = 4.0 * kc_atr / close
    squeeze = (bb_w < kc_w).astype(float)
    release = (squeeze.shift(1) == 1) & (squeeze == 0)
    return release.astype(float)


# ── Temporal derivatives (critical for LSTM momentum detection) ──────────────

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + 1e-9))


def rsi_roc(close: pd.Series, rsi_n: int = 14, roc_n: int = 3) -> pd.Series:
    """Rate of change of RSI — detects acceleration/deceleration of momentum."""
    r = rsi(close, rsi_n)
    return r.diff(roc_n)


def macd_hist(close: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram = MACD line - signal line."""
    ema_f = close.ewm(span=fast,   adjust=False).mean()
    ema_s = close.ewm(span=slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def macd_hist_roc(close: pd.Series, roc_n: int = 3) -> pd.Series:
    """Rate of change of MACD histogram — detects momentum shift."""
    return macd_hist(close).diff(roc_n)


# ── Master function: add all Nomadic features to an existing feat dict ────────

def add_nomadic_features(c: pd.Series, feat: dict,
                         high: pd.Series, low: pd.Series,
                         volume: pd.Series) -> dict:
    """
    Compute all extended indicators and add them to the feature dict.

    Parameters
    ----------
    c      : close price Series (already available in the base pipeline)
    feat   : existing feature dict from make_features()
    high   : high price Series
    low    : low price Series
    volume : daily volume Series

    Returns
    -------
    feat updated with ~25 new indicator columns.
    """
    # Oscillators
    feat["cci20"]         = cci(high, low, c)
    feat["williams_r14"]  = williams_r(high, low, c)
    feat["cmo14"]         = cmo(c)
    feat["awesome_osc"]   = awesome_oscillator(high, low)
    feat["trix15"]        = trix(c)

    # Volume oscillators
    feat["obv"]           = obv(c, volume)
    feat["cmf20"]         = chaikin_money_flow(high, low, c, volume)
    feat["mfi14"]         = money_flow_index(high, low, c, volume)
    feat["klinger"]       = klinger_oscillator(high, low, c, volume)

    # Trend
    adx_data              = adx_di(high, low, c)
    feat["adx14"]         = adx_data["adx"]
    feat["di_spread"]     = adx_data["plus_di"] - adx_data["minus_di"]
    feat["ichimoku_dist"] = ichimoku_cloud_distance(high, low, c)

    # VWAP
    vwap_val              = rolling_vwap(high, low, c, volume)
    feat["vwap_dev"]      = (c - vwap_val) / (c + 1e-9)

    # Breakout
    feat["don_breakout20"] = donchian_breakout(c, high)
    feat["bb_squeeze_rel"] = bb_squeeze_release(c, high, low)

    # Temporal derivatives (invaluable for LSTM branch)
    feat["rsi_roc3"]       = rsi_roc(c)
    feat["macd_hist_roc3"] = macd_hist_roc(c)
    feat["vol_roc5"]       = volume.pct_change(5)
    feat["price_accel"]    = c.pct_change().diff()   # 2nd derivative

    return feat
