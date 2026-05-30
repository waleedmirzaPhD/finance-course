"""Module 3b entry: QLBS option pricing/hedging vs Black-Scholes."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rl.qlbs import (   # noqa: E402
    QLBSConfig, bs_call_delta, qlbs_solve,
)


def main() -> None:
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    print("[1/2] Solving QLBS at-the-money European call (S0=K=100)...")
    cfg = QLBSConfig(
        S0=100.0, K=100.0, T=1.0, r=0.0, mu=0.05, sigma=0.20,
        n_steps=24, n_paths=20000, seed=0,
    )
    result = qlbs_solve(cfg)
    table = pd.DataFrame({
        "QLBS":          [result["price_t0"], result["delta_t0"]],
        "Black-Scholes": [result["bs_price"], result["bs_delta"]],
    }, index=["price_t0", "delta_t0"])
    table["abs_error"] = (table["QLBS"] - table["Black-Scholes"]).abs()
    print(table.to_string(float_format=lambda x: f"{x: .4f}"))

    print("\n[2/2] Hedge schedule: QLBS u(S_t) vs BS delta(S_t)...")
    # Visualize the hedge function at t=0 across simulated S0 values by
    # re-running with a small grid of S0.
    grid = np.linspace(80, 120, 9)
    qd, bd = [], []
    for s0 in grid:
        sub_cfg = QLBSConfig(**{**cfg.__dict__, "S0": s0, "n_paths": 8000})
        r = qlbs_solve(sub_cfg)
        qd.append(r["delta_t0"])
        bd.append(bs_call_delta(s0, cfg.K, cfg.T, cfg.r, cfg.sigma))
    deltas = pd.DataFrame({"S0": grid, "QLBS_delta": qd, "BS_delta": bd})
    print(deltas.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sample = result["paths"][:60].T
    ax1.plot(sample, color="gray", alpha=0.3, lw=0.7)
    ax1.set_title(f"60 sample GBM paths (T={cfg.T}, σ={cfg.sigma})")
    ax1.set_xlabel("step")
    ax1.set_ylabel("S_t")
    ax1.grid(alpha=0.3)

    ax2.plot(deltas["S0"], deltas["BS_delta"], "o-", label="BS delta",
             color="#264653", lw=2)
    ax2.plot(deltas["S0"], deltas["QLBS_delta"], "s--", label="QLBS delta",
             color="#e76f51", lw=1.6)
    ax2.set_title("Delta at t=0  —  QLBS vs Black-Scholes")
    ax2.set_xlabel("S0")
    ax2.set_ylabel("hedge (shares of underlying)")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "qlbs.png", dpi=140)
    plt.close(fig)

    with pd.ExcelWriter(out_dir / "qlbs_report.xlsx", engine="openpyxl") as w:
        table.to_excel(w, sheet_name="t0_compare")
        deltas.to_excel(w, sheet_name="DeltaGrid", index=False)
    print(f"\nReport -> {out_dir / 'qlbs_report.xlsx'}")
    print(f"Plot   -> {out_dir / 'qlbs.png'}")


if __name__ == "__main__":
    main()
