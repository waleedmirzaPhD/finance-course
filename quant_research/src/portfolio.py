"""Portfolio construction from a cross-sectional alpha frame.

Two weighting schemes, both dollar-neutral and capped at gross_leverage:

    'equal'         : equal weight within long / short deciles
    'tf_max_sharpe' : solve max-Sharpe within each book via the local
                      TF optimizer in src/tf_max_sharpe.py, using a
                      rolling covariance estimate

The output is a daily target-weight frame summing to 0 (dollar-neutral)
with sum(|w|) == gross_leverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.tf_max_sharpe import max_sharpe_weights


def _decile_buckets(row: pd.Series, n: int = 10) -> pd.Series:
    """Assign a 1..n bucket per name based on rank (1 = highest alpha)."""
    valid = row.dropna()
    if len(valid) < n:
        return pd.Series(np.nan, index=row.index)
    ranks = valid.rank(ascending=False, method="first")
    buckets = np.ceil(ranks * n / len(valid)).astype(int).clip(1, n)
    out = pd.Series(np.nan, index=row.index)
    out.loc[valid.index] = buckets
    return out


def equal_decile_weights(alpha: pd.DataFrame, long_decile: int = 1,
                         short_decile: int = 10,
                         gross_leverage: float = 2.0) -> pd.DataFrame:
    """Equal-weight long top decile, equal-weight short bottom decile."""
    w = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
    long_book = gross_leverage / 2.0
    short_book = gross_leverage / 2.0
    for d, row in alpha.iterrows():
        buckets = _decile_buckets(row, n=10)
        longs = buckets[buckets == long_decile].index
        shorts = buckets[buckets == short_decile].index
        if len(longs) > 0:
            w.loc[d, longs] = long_book / len(longs)
        if len(shorts) > 0:
            w.loc[d, shorts] = -short_book / len(shorts)
    return w


def _max_sharpe_book(returns: pd.DataFrame, names: list[str],
                     risk_free: float = 0.0) -> np.ndarray:
    """Long-only max-Sharpe weights on `names`, summing to 1. NaN-safe."""
    sub = returns[names].dropna(how="any")
    if sub.shape[0] < 60 or sub.shape[1] < 2:
        return np.full(len(names), 1.0 / len(names))
    mu = sub.mean().to_numpy() * 252
    cov = sub.cov().to_numpy() * 252
    try:
        return max_sharpe_weights(mu, cov, risk_free, lr=0.05, steps=800)
    except Exception:
        return np.full(len(names), 1.0 / len(names))


def tf_optimized_decile_weights(alpha: pd.DataFrame, returns: pd.DataFrame,
                                long_decile: int = 1, short_decile: int = 10,
                                gross_leverage: float = 2.0,
                                cov_window_days: int = 252,
                                refit_freq_days: int = 21) -> pd.DataFrame:
    """Within each decile, allocate via TF max-Sharpe on a trailing window.

    Refits the optimizer every `refit_freq_days` for performance — the
    decile membership still updates daily, but the per-name allocation
    inside the book changes only at refit.
    """
    w = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
    long_book = gross_leverage / 2.0
    short_book = gross_leverage / 2.0

    last_long_w: dict[str, float] = {}
    last_short_w: dict[str, float] = {}
    days_since_refit = refit_freq_days  # force refit on first eligible day

    for d, row in alpha.iterrows():
        buckets = _decile_buckets(row, n=10)
        longs = buckets[buckets == long_decile].index.tolist()
        shorts = buckets[buckets == short_decile].index.tolist()

        if days_since_refit >= refit_freq_days and d in returns.index:
            end = returns.index.searchsorted(d, side="right")
            start = max(0, end - cov_window_days)
            window = returns.iloc[start:end]
            if len(longs) >= 2 and len(window) >= 60:
                w_long = _max_sharpe_book(window, longs)
                last_long_w = dict(zip(longs, w_long))
            elif longs:
                last_long_w = {t: 1.0 / len(longs) for t in longs}
            if len(shorts) >= 2 and len(window) >= 60:
                w_short = _max_sharpe_book(window, shorts)
                last_short_w = dict(zip(shorts, w_short))
            elif shorts:
                last_short_w = {t: 1.0 / len(shorts) for t in shorts}
            days_since_refit = 0
        days_since_refit += 1

        for t, wi in last_long_w.items():
            if t in w.columns:
                w.loc[d, t] = long_book * wi
        for t, wi in last_short_w.items():
            if t in w.columns:
                w.loc[d, t] = -short_book * wi

    return w


def build_weights(alpha: pd.DataFrame, returns: pd.DataFrame,
                  cfg: dict) -> pd.DataFrame:
    method = cfg["weighting"]
    if method == "equal":
        return equal_decile_weights(
            alpha,
            long_decile=cfg["long_decile"],
            short_decile=cfg["short_decile"],
            gross_leverage=cfg["gross_leverage"],
        )
    if method == "tf_max_sharpe":
        return tf_optimized_decile_weights(
            alpha, returns,
            long_decile=cfg["long_decile"],
            short_decile=cfg["short_decile"],
            gross_leverage=cfg["gross_leverage"],
        )
    raise ValueError(f"unknown weighting: {method}")
