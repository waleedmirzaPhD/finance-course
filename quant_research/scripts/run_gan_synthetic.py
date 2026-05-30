"""Module 4 entry: GAN for synthetic price paths + stylized-fact comparison.

We train a tiny adversarial generator on real log-return windows, then
compare the distribution of synthetic windows to the real distribution
on a handful of stylized facts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import yaml                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data                                        # noqa: E402
from src.rl.gan import sample, stylized_facts, train_gan    # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    print("[1/3] Loading prices and computing log returns...")
    prices = data.load_prices(
        cfg["universe"]["tickers"],
        start=cfg["dates"]["start"],
        end=cfg["dates"]["end"],
    )
    rets = np.log(prices).diff().iloc[1:]
    print(f"      panel: {rets.shape[0]} days x {rets.shape[1]} tickers")

    print("\n[2/3] Training GAN...")
    out = train_gan(rets, window=21, noise_dim=8, epochs=400,
                    batch_size=128, seed=0)
    print(f"      final D loss: {out['history']['d_loss'][-1]:.3f}")
    print(f"      final G loss: {out['history']['g_loss'][-1]:.3f}")

    print("\n[3/3] Comparing stylized facts...")
    syn = sample(out, n=5000, noise_dim=8, seed=1)
    real_flat = rets.to_numpy().ravel()
    syn_flat = syn.ravel()
    facts = pd.DataFrame({
        "real":      stylized_facts(real_flat),
        "synthetic": stylized_facts(syn_flat),
    })
    facts["abs_pct_err"] = (facts["synthetic"] - facts["real"]).abs() / \
        (facts["real"].abs() + 1e-9)
    print(facts.to_string(float_format=lambda x: f"{x: .4f}"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(out["history"]["d_loss"], label="D loss", color="#264653")
    axes[0].plot(out["history"]["g_loss"], label="G loss", color="#e76f51")
    axes[0].set_title("GAN training loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    bins = np.linspace(-0.06, 0.06, 80)
    axes[1].hist(real_flat[np.isfinite(real_flat)], bins=bins,
                 density=True, alpha=0.5, label="real", color="#264653")
    axes[1].hist(syn_flat, bins=bins, density=True,
                 alpha=0.5, label="synthetic", color="#e76f51")
    axes[1].set_title("Log-return distribution")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Sample path plot.
    syn_paths = sample(out, n=20, noise_dim=8, seed=2)
    px = np.exp(np.cumsum(syn_paths, axis=1)) * 100.0
    axes[2].plot(px.T, alpha=0.6, lw=0.9)
    axes[2].set_title("20 synthetic price paths (start=100)")
    axes[2].set_xlabel("day")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "gan.png", dpi=140)
    plt.close(fig)

    with pd.ExcelWriter(out_dir / "gan_report.xlsx", engine="openpyxl") as w:
        facts.to_excel(w, sheet_name="StylizedFacts")
        pd.DataFrame(out["history"]).to_excel(w, sheet_name="LossHistory")
    print(f"\nReport -> {out_dir / 'gan_report.xlsx'}")
    print(f"Plot   -> {out_dir / 'gan.png'}")


if __name__ == "__main__":
    main()
