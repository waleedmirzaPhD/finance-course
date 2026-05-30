"""Vectorized event-driven backtest.

Sequence per bar t:
    1. Target weights w_t are known from alpha computed on data <= t.
    2. We trade to those weights at close(t + delay_days). The realized
       PnL contribution attributable to that decision is collected over
       (t+delay, t+delay+rebalance_freq], gross of costs.
    3. Transaction costs and slippage are charged on |w_new - w_old|.

The result: a daily strategy-return series, plus a turnover series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(weights: pd.DataFrame, returns: pd.DataFrame,
             delay_days: int = 1, cost_bps_per_side: float = 5.0,
             slippage_bps: float = 2.0,
             rebalance_freq_days: int = 5) -> dict:
    """Run the backtest.

    Returns dict with:
        - strategy_returns : pd.Series of daily net returns
        - gross_returns    : pd.Series of daily gross returns
        - costs            : pd.Series of daily cost drag
        - turnover         : pd.Series of daily L1 weight change
        - held_weights     : pd.DataFrame, weights actually held each day
    """
    w_target = weights.reindex(returns.index).fillna(0.0)
    w_target = w_target[w_target.columns.intersection(returns.columns)]
    r = returns[w_target.columns]

    # Apply execution delay.
    w_shift = w_target.shift(delay_days).fillna(0.0)

    # Enforce rebalance frequency: only update on every Nth day.
    held = pd.DataFrame(0.0, index=w_shift.index, columns=w_shift.columns)
    last_w = np.zeros(w_shift.shape[1])
    days_since = rebalance_freq_days
    for i, d in enumerate(w_shift.index):
        if days_since >= rebalance_freq_days:
            last_w = w_shift.iloc[i].to_numpy()
            days_since = 0
        held.iloc[i] = last_w
        days_since += 1

    # Daily turnover = L1 change in held weights.
    diff = held.diff().abs().sum(axis=1).fillna(0.0)
    cost_per_unit_turnover = (cost_bps_per_side + slippage_bps) / 1e4
    costs = diff * cost_per_unit_turnover

    gross = (held * r.fillna(0.0)).sum(axis=1)
    net = gross - costs

    return {
        "strategy_returns": net,
        "gross_returns": gross,
        "costs": costs,
        "turnover": diff,
        "held_weights": held,
    }
