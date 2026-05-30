"""Module 2 entry: PCA factors + denoising autoencoder + k-means regimes."""

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

from src import data                                 # noqa: E402
from src.unsup.autoencoder import train_autoencoder  # noqa: E402
from src.unsup.pca_factors import (                  # noqa: E402
    fit_pca_factors, market_factor_diagnostic,
)
from src.unsup.regimes import cluster_regimes, label_regimes  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    print("[1/3] PCA on return matrix...")
    prices = data.load_prices(
        cfg["universe"]["tickers"],
        start=cfg["dates"]["start"],
        end=cfg["dates"]["end"],
    )
    rets = data.daily_returns(prices)
    factors, loadings, ev = fit_pca_factors(rets, n_components=5)
    pc_mkt_corr = market_factor_diagnostic(factors, rets)
    print(f"      explained variance: {np.round(ev, 3).tolist()}")
    print(f"      PC corr with eq-wt market basket:")
    print(pc_mkt_corr.to_string(float_format=lambda x: f"{x: .3f}"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(range(1, len(ev) + 1), ev * 100, color="#264653")
    ax1.set_title("PCA explained variance")
    ax1.set_xlabel("PC")
    ax1.set_ylabel("% variance")
    ax1.grid(alpha=0.3, axis="y")
    loadings["PC1"].sort_values().plot(kind="barh", ax=ax2, color="#2a9d8f")
    ax2.set_title("PC1 loadings (market factor)")
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "unsup_pca.png", dpi=140)
    plt.close(fig)

    print("\n[2/3] Denoising autoencoder on rolling return windows...")
    ae = train_autoencoder(rets, window=21, bottleneck=4, epochs=8,
                           batch_size=512, noise_std=0.005)
    final_val = ae["history"]["val_loss"][-1]
    print(f"      bottleneck dim: 4, final val MSE: {final_val:.5f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    pd.Series(ae["history"]["loss"], name="train").plot(ax=ax, label="train")
    pd.Series(ae["history"]["val_loss"], name="val").plot(ax=ax, label="val")
    ax.set_title("Denoising autoencoder loss")
    ax.set_ylabel("MSE")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "unsup_ae_loss.png", dpi=140)
    plt.close(fig)

    print("\n[3/3] K-means market regimes...")
    labels, _ = cluster_regimes(rets, n_clusters=3)
    names = label_regimes(labels, rets)
    market = rets.mean(axis=1).reindex(labels.index)
    print(f"      regime labels: {names}")
    regime_summary = market.groupby(labels).agg(
        n_days="count",
        ann_ret=lambda x: x.mean() * 252,
        ann_vol=lambda x: x.std() * np.sqrt(252),
    )
    regime_summary["label"] = regime_summary.index.map(names)
    print(regime_summary.to_string(float_format=lambda x: f"{x: .3f}"))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    cum = (1 + market.fillna(0)).cumprod()
    ax.plot(cum.index, cum.values, color="black", lw=1.2, label="EQ-wt market")
    colors = {0: "#2a9d8f", 1: "#e76f51", 2: "#f4a261"}
    for cid in sorted(set(labels)):
        mask = labels == cid
        ax.fill_between(cum.index, 0, cum.max() * 1.1,
                        where=mask.reindex(cum.index, fill_value=False),
                        color=colors.get(cid, "gray"), alpha=0.25,
                        label=f"regime {cid} ({names[cid]})")
    ax.set_ylim(0, cum.max() * 1.1)
    ax.set_title("Market regimes (k-means on rolling vol/trend/skew)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "unsup_regimes.png", dpi=140)
    plt.close(fig)

    with pd.ExcelWriter(out_dir / "unsup_report.xlsx", engine="openpyxl") as w:
        pd.DataFrame({"explained_var": ev,
                      "corr_market": pc_mkt_corr.values},
                     index=[f"PC{k+1}" for k in range(len(ev))]
                     ).to_excel(w, sheet_name="PCA")
        loadings.to_excel(w, sheet_name="PCA_loadings")
        regime_summary.to_excel(w, sheet_name="Regimes")
    print(f"\nReports -> {out_dir}/unsup_*.{{png,xlsx}}")


if __name__ == "__main__":
    main()
