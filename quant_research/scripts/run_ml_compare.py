"""Module 1: walk-forward comparison of Ridge / RF / GBM / MLP forecasters.

Reuses the same signals (momentum, low-vol, quality) as the classical
backtest and reports per-model OOS information coefficient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402
import yaml                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data, signals                  # noqa: E402
from src.ml.forecasters import default_zoo     # noqa: E402
from src.ml.model_compare import compare       # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    print("[1/3] Loading prices and signals...")
    prices = data.load_prices(
        cfg["universe"]["tickers"],
        start=cfg["dates"]["start"],
        end=cfg["dates"]["end"],
    )
    sig = signals.compute_all(prices, cfg["signals"])

    print("[2/3] Walk-forward fit + predict for each model...")
    models = default_zoo()
    _, ic_table = compare(
        models, sig, prices,
        fwd_return_days=cfg["blend"]["fwd_return_days"],
        train_window_days=cfg["blend"]["train_window_days"],
        refit_freq_days=63,
    )

    print("\n=== OOS Information Coefficient by model ===")
    print(ic_table.to_string(float_format=lambda x: f"{x: .4f}"))

    print("\n[3/3] Writing report + plot...")
    ic_table.to_excel(out_dir / "ml_compare.xlsx", sheet_name="IC")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(ic_table.index, ic_table["t_stat"],
                  color=["#264653", "#2a9d8f", "#e9c46a", "#f4a261"])
    ax.axhline(2, color="k", ls="--", lw=0.7, label="t = 2 (rough sig)")
    ax.set_title("OOS IC t-stat by forecaster")
    ax.set_ylabel("t-stat of daily IC")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, ic_table["t_stat"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ml_compare_tstat.png", dpi=140)
    plt.close(fig)
    print(f"      report -> {out_dir / 'ml_compare.xlsx'}")
    print(f"      plot   -> {out_dir / 'ml_compare_tstat.png'}")


if __name__ == "__main__":
    main()
