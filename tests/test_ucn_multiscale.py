"""Focused smoke tests for the final multi-scale course model."""

from __future__ import annotations

import numpy as np

from ucn.models.multiscale import MultiScaleTermStructureNet


def test_multiscale_fit_predict_and_checkpoint(tmp_path):
    rng = np.random.default_rng(7)
    windows = [2, 4]
    sequences = [rng.normal(size=(24, window, 3)).astype(np.float32) for window in windows]
    context = rng.normal(size=(24, 2)).astype(np.float32)
    labels = rng.integers(0, 2, size=(24, len(windows))).astype(np.float32)
    forward_returns = rng.normal(0, 0.02, size=(24, len(windows))).astype(np.float32)

    model = MultiScaleTermStructureNet(
        windows=windows,
        hidden=4,
        trunk_sizes=(8, 4),
        epochs=2,
        batch_size=8,
        patience=0,
        verbose=0,
        seed=7,
    )
    model.fit(sequences, labels, Yret=forward_returns, ctx=context)

    probabilities = model.predict_proba(sequences, ctx=context)
    bands = model.predict_bands(sequences, ctx=context)
    assert probabilities.shape == labels.shape
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert bands.shape == (24, len(windows), len(model.quantiles))
    assert np.all(np.diff(bands, axis=2) >= -1e-7)

    checkpoint = tmp_path / "multiscale.npz"
    model.save(checkpoint)
    restored = MultiScaleTermStructureNet.load(checkpoint)

    np.testing.assert_allclose(
        restored.predict_proba(sequences, ctx=context),
        probabilities,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        restored.predict_bands(sequences, ctx=context),
        bands,
        rtol=1e-6,
        atol=1e-6,
    )
