"""Entry point: end-to-end backtest from config.yaml.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --config config.yaml --no-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data, signals, alpha as alpha_mod, portfolio, backtest, metrics, plotting  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def benchmark_eq_weighted(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight long-only basket on the same universe."""
    return returns.mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--no-cache", action="store_true", help="Refetch prices")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading prices for {len(cfg['universe']['tickers'])} tickers...")
    prices = data.load_prices(
        cfg["universe"]["tickers"],
        start=cfg["dates"]["start"],
        end=cfg["dates"]["end"],
        use_cache=not args.no_cache,
    )
    print(f"      panel: {prices.shape[0]} days x {prices.shape[1]} assets")
    rets = data.daily_returns(prices)

    print("[2/6] Computing signals (momentum, low-vol, quality)...")
    sig_dict = signals.compute_all(prices, cfg["signals"])
    for name, s in sig_dict.items():
        print(f"      {name}: {s.notna().any(axis=1).sum()} active days")

    print(f"[3/6] Blending alpha via {cfg['blend']['method']}...")
    alpha = alpha_mod.compute_alpha(sig_dict, prices, cfg["blend"])
    active_days = alpha.notna().any(axis=1).sum()
    print(f"      alpha active on {active_days} days")
    if active_days == 0:
        sys.exit("No alpha generated — check signal coverage / config.")

    print(f"[4/6] Building portfolio weights ({cfg['portfolio']['weighting']})...")
    weights = portfolio.build_weights(alpha, rets, cfg["portfolio"])
    print(f"      avg gross leverage: {weights.abs().sum(axis=1).mean():.2f}")
    print(f"      avg net exposure : {weights.sum(axis=1).mean():+.3f}")

    print("[5/6] Running backtest...")
    result = backtest.backtest(
        weights, rets,
        delay_days=cfg["execution"]["delay_days"],
        cost_bps_per_side=cfg["execution"]["cost_bps_per_side"],
        slippage_bps=cfg["execution"]["slippage_bps"],
        rebalance_freq_days=cfg["portfolio"]["rebalance_freq_days"],
    )
    strat = result["strategy_returns"]
    print(f"      strategy active on {(strat != 0).sum()} days")

    print("[6/6] Computing metrics and plots...")
    rf = cfg["risk"]["annual_risk_free"]

    # Trim leading days where the strategy was not yet trading (warmup
    # for the rolling Ridge / signal lookbacks). Report metrics on the
    # ACTIVE period; print the full-history view too for transparency.
    first_active = strat.ne(0).idxmax() if (strat != 0).any() else strat.index[0]
    strat_active = strat.loc[first_active:]
    print(f"      active from {first_active.date()} "
          f"({len(strat_active)} days, {len(strat) - len(strat_active)} warmup)")

    summary = metrics.summary_stats(strat_active, risk_free=rf)
    summary_full = metrics.summary_stats(strat, risk_free=rf)
    turn = metrics.turnover_stats(result["turnover"].loc[first_active:])
    bench = benchmark_eq_weighted(rets).reindex(strat_active.index)
    bench_summary = metrics.summary_stats(bench, risk_free=rf)
    yearly = metrics.yearly_breakdown(strat_active)

    # Spread stats: strategy minus benchmark, daily.
    spread = (strat_active - bench).dropna()
    spread_sharpe = (spread.mean() * 252) / (spread.std() * np.sqrt(252)) \
        if spread.std() > 0 else np.nan

    # Forward returns aligned to alpha for IC.
    fwd = np.log(prices).diff(cfg["blend"]["fwd_return_days"]).shift(
        -cfg["blend"]["fwd_return_days"]
    )
    ic = metrics.information_coefficient(alpha, fwd)
    ic_sum = metrics.ic_summary(ic)

    # Plots (active period only).
    plotting.equity_curve(strat_active, bench, out_dir / "equity_curve.png")
    plotting.drawdown(strat_active, out_dir / "drawdown.png")
    plotting.rolling_sharpe(strat_active, out_dir / "rolling_sharpe.png")
    plotting.yearly_returns(yearly, out_dir / "yearly_returns.png")
    plotting.ic_timeseries(ic, out_dir / "ic.png")

    # Excel report.
    report_path = out_dir / "backtest_report.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as w:
        side = pd.concat(
            [summary.rename("Strategy_active"),
             summary_full.rename("Strategy_full_history"),
             bench_summary.rename("Benchmark")],
            axis=1,
        )
        side.loc["SpreadSharpe", "Strategy_active"] = spread_sharpe
        side.to_excel(w, sheet_name="Summary")
        turn.to_frame("Value").to_excel(w, sheet_name="Turnover")
        ic_sum.to_frame("Value").to_excel(w, sheet_name="IC")
        yearly.to_excel(w, sheet_name="Yearly")
        weights.tail(60).to_excel(w, sheet_name="Weights_last60")
        result["held_weights"].tail(60).to_excel(w, sheet_name="Held_last60")

    # CLI summary table.
    print("\n=== Strategy (active period) vs Benchmark (eq-wt long-only) ===")
    cli_side = summary.to_frame("Strategy").join(bench_summary.to_frame("Benchmark"))
    print(cli_side.to_string(float_format=lambda x: f"{x: .4f}"))
    print(f"\nSpread (strategy - benchmark) annualized Sharpe: {spread_sharpe: .3f}")
    print("\n=== Turnover ===")
    print(turn.to_string(float_format=lambda x: f"{x: .4f}"))
    print("\n=== Signal IC ===")
    print(ic_sum.to_string(float_format=lambda x: f"{x: .4f}"))
    print(f"\nReport  -> {report_path}")
    print(f"Plots   -> {out_dir}/*.png")


if __name__ == "__main__":
    main()
