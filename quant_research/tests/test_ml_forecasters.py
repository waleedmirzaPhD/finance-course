"""Sanity tests for the forecaster zoo."""

from __future__ import annotations

import numpy as np

from src.ml.forecasters import (
    GBMForecaster, MLPForecaster, RandomForestForecaster, RidgeForecaster,
)


def _toy_data(n: int = 500, p: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([1.5, -1.0, 0.5, 0.0, 0.0])
    y = X @ beta + 0.5 * rng.normal(size=n)
    return X, y


def _signed_pearson(yhat, y) -> float:
    yhat = yhat - yhat.mean()
    y = y - y.mean()
    return float((yhat * y).sum() / (np.sqrt((yhat ** 2).sum() * (y ** 2).sum()) + 1e-12))


def test_ridge_recovers_signal() -> None:
    X, y = _toy_data()
    m = RidgeForecaster(alpha=0.1)
    m.fit(X[:400], y[:400])
    assert _signed_pearson(m.predict(X[400:]), y[400:]) > 0.5


def test_rf_recovers_signal() -> None:
    X, y = _toy_data(n=800)
    m = RandomForestForecaster(n_estimators=80, max_depth=4, min_samples_leaf=10)
    m.fit(X[:600], y[:600])
    assert _signed_pearson(m.predict(X[600:]), y[600:]) > 0.4


def test_gbm_recovers_signal() -> None:
    X, y = _toy_data(n=800)
    m = GBMForecaster(n_estimators=100, max_depth=3, learning_rate=0.05)
    m.fit(X[:600], y[:600])
    assert _signed_pearson(m.predict(X[600:]), y[600:]) > 0.4


def test_mlp_runs_and_predicts() -> None:
    """We don't assert performance — MLPs on 500 samples are noisy.
    We just check the interface works end-to-end."""
    X, y = _toy_data(n=500)
    m = MLPForecaster(hidden=(8,), dropout=0.0, epochs=5, verbose=0)
    m.fit(X[:400], y[:400])
    preds = m.predict(X[400:])
    assert preds.shape == (100,)
    assert np.isfinite(preds).all()
