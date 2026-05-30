"""Sanity checks on performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics


def test_sharpe_of_constant_return() -> None:
    """A constant return series has ~0 vol and either huge or non-finite Sharpe."""
    r = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252 * 3))
    s = metrics.summary_stats(r)
    assert s["AnnVol"] < 1e-10
    assert np.isnan(s["Sharpe"]) or np.isinf(s["Sharpe"]) or abs(s["Sharpe"]) > 1e6


def test_max_drawdown_is_negative() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 500),
                  index=pd.bdate_range("2020-01-01", periods=500))
    s = metrics.summary_stats(r)
    assert s["MaxDrawdown"] <= 0


def test_turnover_stats() -> None:
    t = pd.Series([0.0, 0.1, 0.2, 0.05],
                  index=pd.bdate_range("2020-01-01", periods=4))
    s = metrics.turnover_stats(t)
    assert np.isclose(s["DailyTurnoverMean"], 0.0875)
    assert np.isclose(s["AnnTurnover"], 0.0875 * 252 / 2)


def test_ic_summary_perfect_signal() -> None:
    """IC of a perfectly informative signal should be ~1.0."""
    rng = np.random.default_rng(0)
    n_assets, n_days = 20, 60
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_assets)]
    fwd = pd.DataFrame(rng.normal(size=(n_days, n_assets)),
                       index=dates, columns=tickers)
    alpha = fwd.copy()                  # perfectly aligned signal
    ic = metrics.information_coefficient(alpha, fwd)
    summary = metrics.ic_summary(ic)
    assert summary["IC_mean"] > 0.95
