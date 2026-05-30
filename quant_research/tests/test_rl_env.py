"""Tests for the RL env and a quick DQN smoke test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.rl.env import PortfolioEnv, default_action_menu


def _toy_rets(n_days: int = 200, n_assets: int = 3, seed: int = 0
              ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=(n_days, n_assets))
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(rets, index=dates,
                        columns=[f"A{i}" for i in range(n_assets)])


def test_env_step_shapes_and_reward() -> None:
    rets = _toy_rets()
    env = PortfolioEnv(rets, actions=default_action_menu(3), window=10)
    s = env.reset(start_idx=20)
    assert s.shape == (env.state_dim,)
    s2, r, done, info = env.step(action_idx=1)  # equal weight
    assert s2.shape == (env.state_dim,)
    assert isinstance(r, float)
    assert "turnover" in info and "pnl" in info
    assert not done


def test_env_episode_terminates() -> None:
    rets = _toy_rets(n_days=30)
    env = PortfolioEnv(rets, actions=default_action_menu(3), window=5)
    env.reset(start_idx=5)
    done = False
    steps = 0
    while not done and steps < 100:
        _, _, done, _ = env.step(action_idx=0)
        steps += 1
    assert done


def test_dqn_trains_one_step() -> None:
    """Push a few transitions and verify a training step runs end-to-end."""
    rets = _toy_rets(n_days=80)
    env = PortfolioEnv(rets, actions=default_action_menu(3), window=10)
    cfg = DQNConfig(hidden=(8,), batch_size=8, replay_size=200,
                    eps_decay_steps=50, seed=0)
    agent = DQNAgent(env.state_dim, env.n_actions, cfg)
    s = env.reset(start_idx=10)
    for _ in range(20):
        a = agent.act(s)
        s2, r, done, _ = env.step(a)
        agent.remember(s, a, r, s2, done)
        s = s2 if not done else env.reset(start_idx=10)
    loss = agent.train_step()
    assert loss is None or np.isfinite(loss)
