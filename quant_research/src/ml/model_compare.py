"""Walk-forward model comparison.

For each model in the zoo:
    - retrain on a rolling 3y window of (signal_z_t, fwd_return_{t+h}) pairs
    - predict on the live signal values until the next refit
    - record OOS predictions
    - compute daily cross-sectional Spearman IC vs realized forward returns

The output is a long-form DataFrame of (date, ticker, model, pred) plus
a per-model IC summary table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.alpha import _forward_return, _stack_xy
from src.ml.forecasters import Forecaster


def _live_predict(model: Forecaster, signals: dict[str, pd.DataFrame],
                  date: pd.Timestamp) -> pd.Series:
    sig_names = list(signals)
    cols, valid = [], None
    for s in sig_names:
        if date not in signals[s].index:
            return pd.Series(dtype=float)
        col = signals[s].loc[date]
        cols.append(col)
        valid = col.notna() if valid is None else (valid & col.notna())
    if not valid.any():
        return pd.Series(dtype=float)
    X = np.column_stack([c.to_numpy() for c in cols])
    pred = model.predict(X)
    pred = np.where(valid.to_numpy(), pred, np.nan)
    return pd.Series(pred, index=cols[0].index)


def walk_forward_predict(model: Forecaster, signals: dict[str, pd.DataFrame],
                         prices: pd.DataFrame, fwd_return_days: int,
                         train_window_days: int, refit_freq_days: int
                         ) -> pd.DataFrame:
    sample = signals[next(iter(signals))]
    all_dates = sample.dropna(how="all").index
    fwd = _forward_return(prices, fwd_return_days)

    preds = pd.DataFrame(np.nan, index=sample.index, columns=sample.columns,
                         dtype=float)

    first_refit_pos = max(train_window_days, fwd_return_days)
    refit_positions = list(range(first_refit_pos, len(all_dates),
                                 refit_freq_days))
    is_fit = False
    for i, d in enumerate(all_dates):
        if refit_positions and i >= refit_positions[0]:
            refit_positions.pop(0)
            last_obs = d - pd.Timedelta(days=fwd_return_days + 5)
            end_idx = all_dates.searchsorted(last_obs, side="right")
            start_idx = max(0, end_idx - train_window_days)
            train_dates = all_dates[start_idx:end_idx]
            X, y = _stack_xy(signals, fwd, train_dates)
            if len(y) > 100:
                model.fit(X, y)
                is_fit = True
        if not is_fit:
            continue
        preds.loc[d] = _live_predict(model, signals, d).reindex(preds.columns)
    return preds


def daily_ic(preds: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    common = preds.columns.intersection(fwd_returns.columns)
    a, b = preds[common], fwd_returns[common].reindex(preds.index)
    out = pd.Series(np.nan, index=a.index, dtype=float)
    for d in a.index:
        x, y = a.loc[d], b.loc[d]
        m = x.notna() & y.notna()
        if m.sum() >= 10:
            out.loc[d] = stats.spearmanr(x[m], y[m]).statistic
    return out.dropna()


def compare(models: dict[str, Forecaster], signals: dict[str, pd.DataFrame],
            prices: pd.DataFrame, fwd_return_days: int = 21,
            train_window_days: int = 756, refit_freq_days: int = 63
            ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Returns (preds_by_model, ic_summary_table).

    refit_freq_days is intentionally larger than the Ridge baseline's
    (config default 21) because tree/MLP refits are expensive.
    """
    fwd = _forward_return(prices, fwd_return_days)
    preds_by_model: dict[str, pd.DataFrame] = {}
    ic_rows = []
    for name, model in models.items():
        print(f"  [ml] fitting {name}...")
        p = walk_forward_predict(model, signals, prices, fwd_return_days,
                                 train_window_days, refit_freq_days)
        ic = daily_ic(p, fwd)
        preds_by_model[name] = p
        if ic.empty:
            ic_rows.append({"Model": name, "IC_mean": np.nan,
                            "IC_std": np.nan, "IR": np.nan,
                            "t_stat": np.nan, "N_days": 0})
            continue
        mean, sd = ic.mean(), ic.std(ddof=1)
        ic_rows.append({
            "Model": name,
            "IC_mean": mean,
            "IC_std": sd,
            "IR": mean / sd if sd > 0 else np.nan,
            "t_stat": mean / (sd / np.sqrt(len(ic))) if sd > 0 else np.nan,
            "N_days": len(ic),
        })
    return preds_by_model, pd.DataFrame(ic_rows).set_index("Model")
