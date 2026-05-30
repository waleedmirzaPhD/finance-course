"""pytest fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def synth_prices() -> pd.DataFrame:
    """Deterministic synthetic price panel: 5 tickers, 4 years, mixed regimes."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", "2022-01-01")
    n_days = len(dates)
    tickers = list("ABCDE")
    # Different drift / vol per ticker so cross-sectional ranks have signal.
    mu = np.array([0.10, 0.05, -0.02, 0.15, 0.00]) / 252
    sigma = np.array([0.15, 0.20, 0.30, 0.18, 0.25]) / np.sqrt(252)
    rets = rng.normal(mu, sigma, size=(n_days, len(tickers)))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)
