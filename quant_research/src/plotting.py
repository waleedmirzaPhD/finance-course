"""Matplotlib plots for the interview deck. PNG only, no UI."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402


def equity_curve(strategy: pd.Series, benchmark: pd.Series | None,
                 out_path: Path, title: str = "Equity curve") -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    (1 + strategy.fillna(0)).cumprod().plot(ax=ax, label="Strategy (net)", lw=2)
    if benchmark is not None and not benchmark.empty:
        (1 + benchmark.reindex(strategy.index).fillna(0)).cumprod().plot(
            ax=ax, label="Benchmark (eq-wt long-only)", lw=1.2, alpha=0.8
        )
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def drawdown(strategy: pd.Series, out_path: Path) -> None:
    cum = (1 + strategy.fillna(0)).cumprod()
    dd = cum / cum.cummax() - 1
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    ax.set_title(f"Drawdown (max {dd.min():.1%})")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def rolling_sharpe(strategy: pd.Series, out_path: Path,
                   window: int = 252) -> None:
    r = strategy.fillna(0)
    mu = r.rolling(window).mean() * 252
    sd = r.rolling(window).std() * np.sqrt(252)
    sharpe = mu / sd
    fig, ax = plt.subplots(figsize=(9, 3.2))
    sharpe.plot(ax=ax, lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Rolling {window}-day Sharpe")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def yearly_returns(yearly: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in yearly["Return"]]
    ax.bar(yearly.index, yearly["Return"] * 100, color=colors)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Yearly returns (%)")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def ic_timeseries(ic: pd.Series, out_path: Path) -> None:
    if ic.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ic.rolling(63).mean().plot(ax=ax, label="63d rolling mean IC", lw=1.5)
    ax.axhline(ic.mean(), color="k", ls="--", lw=0.8, label=f"Mean = {ic.mean():.3f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Information coefficient (Spearman)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
