"""Tests for unsupervised + deep modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data
from src.unsup.autoencoder import train_autoencoder
from src.unsup.pca_factors import fit_pca_factors, market_factor_diagnostic
from src.unsup.regimes import cluster_regimes


def _synth_returns(n_days: int = 800, n_tickers: int = 12, seed: int = 0
                   ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Common market factor + ticker-specific noise.
    mkt = rng.normal(0.0003, 0.012, n_days)
    idio = rng.normal(0, 0.015, (n_days, n_tickers))
    betas = rng.uniform(0.6, 1.4, n_tickers)
    rets = mkt[:, None] * betas[None, :] + idio
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    return pd.DataFrame(rets, index=dates,
                        columns=[f"T{i}" for i in range(n_tickers)])


def test_pc1_is_the_market() -> None:
    rets = _synth_returns()
    factors, _, ev = fit_pca_factors(rets, n_components=3)
    corr = market_factor_diagnostic(factors, rets)
    assert abs(corr["PC1"]) > 0.95, f"PC1 should track market, |corr|={corr['PC1']}"
    assert ev[0] > 0.4, f"PC1 should explain most variance, got {ev[0]}"


def test_autoencoder_trains() -> None:
    rets = _synth_returns(n_days=400)
    out = train_autoencoder(rets, window=10, bottleneck=2, epochs=3,
                            batch_size=64, noise_std=0.001)
    assert out["history"]["loss"][-1] < out["history"]["loss"][0]
    assert out["embeddings"].shape[1] == 2


def test_regime_labels_have_three_clusters() -> None:
    rets = _synth_returns(n_days=600)
    labels, _ = cluster_regimes(rets, n_clusters=3)
    assert labels.nunique() == 3
    assert len(labels) > 400
