"""Classical cross-sectional signals.

All signals are computed using ONLY data available up to and including
day t. The backtest engine is responsible for applying any further
execution lag.

Returned frame shape: index=Date, columns=Ticker, value=signal at t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(prices: pd.DataFrame, lookback_days: int = 252,
             skip_days: int = 21) -> pd.DataFrame:
    """12-1 momentum: return from t-lookback to t-skip.

    Skipping the last month removes the well-known short-term reversal
    contamination (Jegadeesh-Titman, 1993).
    """
    log_px = np.log(prices)
    return (log_px.shift(skip_days) - log_px.shift(lookback_days))


def realized_vol(prices: pd.DataFrame, lookback_days: int = 63) -> pd.DataFrame:
    """Annualized realized volatility from daily log returns."""
    log_ret = np.log(prices).diff()
    return log_ret.rolling(lookback_days, min_periods=lookback_days // 2).std() * np.sqrt(252)


def low_vol(prices: pd.DataFrame, lookback_days: int = 63) -> pd.DataFrame:
    """Low-vol signal: negative realized vol so higher = better (long low-vol)."""
    return -realized_vol(prices, lookback_days)


def trend_quality(prices: pd.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
    """Sharpe-like ratio of the trend: 252d return / 252d vol.

    A proxy for 'quality of trend' when fundamentals aren't available.
    """
    log_ret = np.log(prices).diff()
    mu = log_ret.rolling(lookback_days, min_periods=lookback_days // 2).mean() * 252
    sd = log_ret.rolling(lookback_days, min_periods=lookback_days // 2).std() * np.sqrt(252)
    return mu / sd.replace(0, np.nan)


def zscore_cross_section(signal: pd.DataFrame, min_assets: int = 10) -> pd.DataFrame:
    """Standardize each row across assets (cross-sectional z-score).

    Days with fewer than `min_assets` non-NaN values are blanked so the
    optimizer doesn't trade in thin cross-sections.
    """
    valid = signal.notna().sum(axis=1) >= min_assets
    mean = signal.mean(axis=1)
    std = signal.std(axis=1).replace(0, np.nan)
    z = signal.sub(mean, axis=0).div(std, axis=0)
    z.loc[~valid] = np.nan
    return z


def compute_all(prices: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Compute every configured signal, return dict of z-scored frames."""
    mom = momentum(prices, cfg["momentum"]["lookback_days"],
                   cfg["momentum"]["skip_days"])
    lv = low_vol(prices, cfg["low_vol"]["lookback_days"])
    qu = trend_quality(prices, cfg["quality"]["lookback_days"])
    return {
        "momentum": zscore_cross_section(mom),
        "low_vol": zscore_cross_section(lv),
        "quality": zscore_cross_section(qu),
    }
