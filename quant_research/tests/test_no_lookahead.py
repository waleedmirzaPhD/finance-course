"""Crucial test: signals at date t must depend only on prices at <= t.

Perturbing prices AFTER day t must not change any signal value at t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import signals


def test_momentum_uses_only_past(synth_prices: pd.DataFrame) -> None:
    base = signals.momentum(synth_prices)
    perturbed = synth_prices.copy()
    cut = perturbed.index[-30]
    perturbed.loc[cut:] *= 5.0          # large future shock
    after = signals.momentum(perturbed)
    diff = (base.loc[:cut - pd.Timedelta(days=1)] -
            after.loc[:cut - pd.Timedelta(days=1)]).abs().max().max()
    assert diff < 1e-12, f"momentum leaked future info, max diff {diff}"


def test_low_vol_uses_only_past(synth_prices: pd.DataFrame) -> None:
    base = signals.low_vol(synth_prices)
    perturbed = synth_prices.copy()
    cut = perturbed.index[-50]
    perturbed.loc[cut:] *= 0.3
    after = signals.low_vol(perturbed)
    diff = (base.loc[:cut - pd.Timedelta(days=1)] -
            after.loc[:cut - pd.Timedelta(days=1)]).abs().max().max()
    assert diff < 1e-12


def test_zscore_blanks_thin_cross_section() -> None:
    df = pd.DataFrame(
        [[1.0, 2.0, np.nan, np.nan, np.nan]],
        index=pd.bdate_range("2020-01-01", periods=1),
        columns=list("ABCDE"),
    )
    z = signals.zscore_cross_section(df, min_assets=3)
    assert z.iloc[0].isna().all(), "thin cross-section should be blanked"
