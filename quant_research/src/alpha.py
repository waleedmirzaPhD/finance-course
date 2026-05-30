"""Alpha = forecast of next-period cross-sectional returns.

Two blenders:
    - 'equal'  : equal-weighted mean of signal z-scores
    - 'ridge'  : walk-forward Ridge regression mapping (signal_z_t) -> r_{t,t+h}

Both produce a daily frame `alpha[date, ticker]` where higher means more
attractive long. Generated using only data available up to and including
the reference date — the backtest engine adds execution delay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def equal_blend(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average of z-scored signals."""
    df = pd.concat(signals.values()).groupby(level=0).mean()
    common_idx = sorted(set.intersection(*[set(s.index) for s in signals.values()]))
    return df.loc[common_idx]


def _forward_return(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """log(price_{t+h} / price_t). NaN at the tail h rows."""
    log_px = np.log(prices)
    return log_px.shift(-horizon) - log_px


def _stack_xy(signals: dict[str, pd.DataFrame], fwd_ret: pd.DataFrame,
              dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Build long-form (n_obs, n_signals) X and (n_obs,) y across dates."""
    sig_names = list(signals)
    rows_x, rows_y = [], []
    for d in dates:
        if d not in fwd_ret.index:
            continue
        y_row = fwd_ret.loc[d]
        x_cols = []
        valid = pd.Series(True, index=y_row.index)
        for s in sig_names:
            if d not in signals[s].index:
                valid[:] = False
                break
            col = signals[s].loc[d]
            x_cols.append(col)
            valid &= col.notna()
        if not valid.any():
            continue
        valid &= y_row.notna()
        if valid.sum() < 5:
            continue
        x_mat = np.column_stack([c[valid].to_numpy() for c in x_cols])
        rows_x.append(x_mat)
        rows_y.append(y_row[valid].to_numpy())
    if not rows_x:
        return np.empty((0, len(sig_names))), np.empty(0)
    return np.vstack(rows_x), np.concatenate(rows_y)


def ridge_blend(signals: dict[str, pd.DataFrame], prices: pd.DataFrame,
                ridge_alpha: float, train_window_days: int,
                refit_freq_days: int, fwd_return_days: int) -> pd.DataFrame:
    """Walk-forward Ridge.

    On each refit date, fit on the most recent `train_window_days` of
    (signal, fwd_return) pairs that are FULLY OBSERVED (the forward
    return must have already realized). Coefficients are then applied to
    the live signal values from the refit date until the next refit.

    Critically: at refit date `d`, only rows with t <= d - fwd_return_days
    contribute to training, so no future information leaks.
    """
    sig_names = list(signals)
    sample_sig = signals[sig_names[0]]
    all_dates = sample_sig.dropna(how="all").index

    fwd_ret = _forward_return(prices, fwd_return_days)

    alpha = pd.DataFrame(np.nan, index=sample_sig.index, columns=sample_sig.columns,
                         dtype=float)

    if len(all_dates) == 0:
        return alpha

    first_refit_pos = max(train_window_days, fwd_return_days)
    refit_positions = list(range(first_refit_pos, len(all_dates), refit_freq_days))
    if not refit_positions:
        return alpha

    coefs: np.ndarray | None = None
    intercept: float = 0.0

    last_refit_pos = -1
    for i, d in enumerate(all_dates):
        # On scheduled refit dates, retrain using data realized BEFORE d.
        if refit_positions and i >= refit_positions[0]:
            refit_positions.pop(0)
            last_observable = d - pd.Timedelta(days=fwd_return_days + 5)
            train_end_idx = all_dates.searchsorted(last_observable, side="right")
            train_start_idx = max(0, train_end_idx - train_window_days)
            train_dates = all_dates[train_start_idx:train_end_idx]
            X, y = _stack_xy(signals, fwd_ret, train_dates)
            if len(y) > 50:
                model = Ridge(alpha=ridge_alpha, fit_intercept=True)
                model.fit(X, y)
                coefs = model.coef_
                intercept = float(model.intercept_)
                last_refit_pos = i

        if coefs is None:
            continue
        # Apply current coefs to today's signal values.
        cols = []
        valid = pd.Series(True, index=sample_sig.columns)
        for s in sig_names:
            col = signals[s].loc[d]
            cols.append(col)
            valid &= col.notna()
        if not valid.any():
            continue
        x_today = np.column_stack([c.to_numpy() for c in cols])
        pred = intercept + x_today @ coefs
        pred[~valid.to_numpy()] = np.nan
        alpha.loc[d] = pred

    return alpha


def compute_alpha(signals: dict[str, pd.DataFrame], prices: pd.DataFrame,
                  cfg: dict) -> pd.DataFrame:
    method = cfg["method"]
    if method == "equal":
        return equal_blend(signals)
    if method == "ridge":
        return ridge_blend(
            signals=signals,
            prices=prices,
            ridge_alpha=cfg["ridge_alpha"],
            train_window_days=cfg["train_window_days"],
            refit_freq_days=cfg["refit_freq_days"],
            fwd_return_days=cfg["fwd_return_days"],
        )
    raise ValueError(f"unknown blend method: {method}")
