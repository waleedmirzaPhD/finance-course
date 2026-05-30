"""Module 3a entry: train a DQN agent on a small basket and report.

We pick a small (5-asset) basket so the discrete action menu remains
tractable and the agent can plausibly learn within a few hundred
training episodes on a laptop.
"""

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

from src import data                              # noqa: E402
from src.rl.dqn_agent import (                    # noqa: E402
    DQNAgent, DQNConfig, evaluate_greedy, train_dqn,
)
from src.rl.env import PortfolioEnv, default_action_menu  # noqa: E402


SMALL_BASKET = ["SPY-proxy", "QQQ-proxy", "TLT-proxy", "GLD-proxy", "XLE-proxy"]


def main() -> None:
    # Use a small sector-diverse basket from our universe as proxies.
    proxies = {
        "SPY-proxy": "JPM",   # broad financial
        "QQQ-proxy": "AAPL",  # mega-cap tech
        "TLT-proxy": "JNJ",   # defensive
        "GLD-proxy": "PG",    # consumer staple
        "XLE-proxy": "XOM",   # energy
    }
    tickers = list(proxies.values())
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    print("[1/3] Loading prices for RL basket...")
    prices = data.load_prices(tickers, start="2010-01-01")
    rets = data.daily_returns(prices)[tickers].dropna(how="any")
    print(f"      basket: {tickers}, {rets.shape[0]} days")

    # Train / test split.
    split = int(len(rets) * 0.7)
    train_rets, test_rets = rets.iloc[:split], rets.iloc[split:]

    actions = default_action_menu(n_assets=len(tickers))
    print(f"      action menu: {actions.shape[0]} discrete weight vectors")

    env_train = PortfolioEnv(train_rets, actions=actions, window=21,
                             cost_bps_per_side=5.0, seed=0)
    env_test = PortfolioEnv(test_rets, actions=actions, window=21,
                            cost_bps_per_side=5.0, seed=1)

    print("\n[2/3] Training DQN agent...")
    cfg = DQNConfig(
        hidden=(64, 64), gamma=0.95, lr=1e-3, batch_size=64,
        replay_size=10000, target_sync_steps=200,
        eps_start=1.0, eps_end=0.05, eps_decay_steps=8000, seed=0,
    )
    agent = DQNAgent(env_train.state_dim, env_train.n_actions, cfg)
    history = train_dqn(env_train, agent, n_episodes=40, max_steps=200,
                        log_every=5)

    print("\n[3/3] Greedy evaluation on holdout...")
    eval_steps = len(test_rets) - env_test.window - 1
    eval_out = evaluate_greedy(env_test, agent, n_steps=eval_steps)
    print(f"      holdout total return: {eval_out['total_return']:.2%}")
    print(f"      holdout Sharpe (ann): {eval_out['sharpe']:.3f}")

    # Compare baselines on the same holdout.
    eq_ret = test_rets.mean(axis=1).iloc[env_test.window + 1:]
    cash_ret = pd.Series(0.0, index=eq_ret.index)
    bench = {
        "DQN_greedy": pd.Series(eval_out["rewards"], index=eq_ret.index[:len(eval_out["rewards"])]),
        "EqualWeight": eq_ret,
        "Cash": cash_ret,
    }
    summary = pd.DataFrame({
        name: {
            "TotalReturn": (1 + s.fillna(0)).prod() - 1,
            "AnnVol": s.std() * np.sqrt(252),
            "AnnSharpe": (s.mean() / (s.std() + 1e-9)) * np.sqrt(252),
            "MaxDD": float(((1 + s.fillna(0)).cumprod() /
                            (1 + s.fillna(0)).cumprod().cummax() - 1).min()),
        }
        for name, s in bench.items()
    })
    print("\n=== Holdout comparison ===")
    print(summary.to_string(float_format=lambda x: f"{x: .4f}"))

    # Plots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    h = pd.DataFrame(history).set_index("episode")
    h["reward"].plot(ax=ax1, color="#264653")
    ax1.set_title("DQN training reward per episode")
    ax1.set_ylabel("episode reward (sum)")
    ax1.grid(alpha=0.3)
    for name, s in bench.items():
        (1 + s.fillna(0)).cumprod().plot(ax=ax2, label=name, lw=1.5)
    ax2.set_title("Holdout equity curves")
    ax2.set_ylabel("Growth of $1")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rl_dqn.png", dpi=140)
    plt.close(fig)

    with pd.ExcelWriter(out_dir / "rl_dqn_report.xlsx", engine="openpyxl") as w:
        h.to_excel(w, sheet_name="TrainHistory")
        summary.to_excel(w, sheet_name="HoldoutComparison")
        pd.Series(eval_out["actions"], name="action_idx").to_excel(
            w, sheet_name="GreedyActions"
        )

    print(f"\nReport -> {out_dir / 'rl_dqn_report.xlsx'}")
    print(f"Plot   -> {out_dir / 'rl_dqn.png'}")


if __name__ == "__main__":
    main()
