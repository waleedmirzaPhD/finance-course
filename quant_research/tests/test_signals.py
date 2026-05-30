"""Sanity checks on signal math."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import signals


def test_momentum_rank_matches_known_winner() -> None:
    """Ticker with the strongest 12-1 return should be the highest-ranked."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame(
        {
            "WINNER": np.linspace(100, 200, 300),       # strong uptrend
            "FLAT": np.full(300, 100.0),
            "LOSER": np.linspace(100, 50, 300),
        },
        index=dates,
    )
    m = signals.momentum(px, lookback_days=252, skip_days=21)
    last = m.dropna().iloc[-1]
    assert last["WINNER"] > last["FLAT"] > last["LOSER"]


def test_low_vol_inverts_vol() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=200)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 200)))
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 200)))
    px = pd.DataFrame({"CALM": calm, "WILD": wild}, index=dates)
    lv = signals.low_vol(px, lookback_days=63).dropna().iloc[-1]
    assert lv["CALM"] > lv["WILD"], "low_vol should rank calm ticker higher"


def test_zscore_mean_zero_std_one() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(5, 20)),
                      index=pd.bdate_range("2020-01-01", periods=5),
                      columns=[f"T{i}" for i in range(20)])
    z = signals.zscore_cross_section(df, min_assets=5)
    np.testing.assert_allclose(z.mean(axis=1).to_numpy(), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=1, ddof=0).to_numpy(),
                               z.iloc[0].std(ddof=0), rtol=0.5)
