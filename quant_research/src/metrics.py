"""Performance and signal-quality metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS = 252


def summary_stats(returns: pd.Series, risk_free: float = 0.0) -> pd.Series:
    """Compact performance summary on a daily-return series."""
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else np.nan
    downside = r[r < 0]
    dvol = downside.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else np.nan
    sortino = (ann_ret - risk_free) / dvol if dvol and dvol > 0 else np.nan
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    years = len(r) / TRADING_DAYS
    cagr = cum.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    hit_rate = (r > 0).mean()
    # Newey-West-ish standard error on the Sharpe (Lo, 2002 simplification).
    se_sharpe = np.sqrt((1 + 0.5 * sharpe ** 2) / len(r)) * np.sqrt(TRADING_DAYS) \
        if not np.isnan(sharpe) else np.nan
    return pd.Series({
        "AnnReturn": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "SharpeSE": se_sharpe,
        "Sortino": sortino,
        "MaxDrawdown": max_dd,
        "CAGR": cagr,
        "HitRate": hit_rate,
        "Years": years,
        "N": len(r),
    })


def turnover_stats(turnover: pd.Series) -> pd.Series:
    """Turnover summary. Annualized turnover = daily L1 * 252 / 2."""
    t = turnover.dropna()
    return pd.Series({
        "DailyTurnoverMean": t.mean(),
        "AnnTurnover": t.mean() * TRADING_DAYS / 2,
        "TurnoverMax": t.max(),
    })


def information_coefficient(alpha: pd.DataFrame, fwd_returns: pd.DataFrame
                            ) -> pd.Series:
    """Daily Spearman rank IC between alpha_t and realized fwd return."""
    common_cols = alpha.columns.intersection(fwd_returns.columns)
    a = alpha[common_cols]
    f = fwd_returns[common_cols].reindex(a.index)
    ic = pd.Series(np.nan, index=a.index, dtype=float)
    for d in a.index:
        x, y = a.loc[d], f.loc[d]
        mask = x.notna() & y.notna()
        if mask.sum() >= 10:
            ic.loc[d] = stats.spearmanr(x[mask], y[mask]).statistic
    return ic.dropna()


def ic_summary(ic: pd.Series) -> pd.Series:
    """Mean IC, IC volatility, IR = mean/std, t-stat."""
    if ic.empty:
        return pd.Series(dtype=float)
    mean = ic.mean()
    sd = ic.std(ddof=1)
    ir = mean / sd if sd > 0 else np.nan
    tstat = mean / (sd / np.sqrt(len(ic))) if sd > 0 else np.nan
    return pd.Series({
        "IC_mean": mean,
        "IC_std": sd,
        "IR (IC mean/std)": ir,
        "t-stat": tstat,
        "N_days": len(ic),
    })


def drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns.fillna(0)).cumprod()
    return cum / cum.cummax() - 1


def yearly_breakdown(returns: pd.Series) -> pd.DataFrame:
    """Per-year return, vol, Sharpe, max drawdown."""
    rows = []
    for y, g in returns.groupby(returns.index.year):
        s = summary_stats(g)
        rows.append({
            "Year": y,
            "Return": (1 + g).prod() - 1,
            "Vol": s["AnnVol"],
            "Sharpe": s["Sharpe"],
            "MaxDD": s["MaxDrawdown"],
        })
    return pd.DataFrame(rows).set_index("Year")
