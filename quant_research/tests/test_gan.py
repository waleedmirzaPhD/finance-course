"""Smoke test for the GAN module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rl.gan import sample, train_gan


def _synth_returns(n_days: int = 400, n_assets: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(n_days, n_assets)),
        index=pd.bdate_range("2020-01-01", periods=n_days),
        columns=[f"T{i}" for i in range(n_assets)],
    )


def test_gan_trains_and_samples() -> None:
    rets = _synth_returns()
    out = train_gan(rets, window=10, noise_dim=4, epochs=20,
                    batch_size=64, seed=0)
    syn = sample(out, n=200, noise_dim=4, seed=0)
    assert syn.shape == (200, 10)
    assert np.isfinite(syn).all()
    # Loss histories shouldn't be NaN.
    assert all(np.isfinite(out["history"]["d_loss"]))
    assert all(np.isfinite(out["history"]["g_loss"]))
