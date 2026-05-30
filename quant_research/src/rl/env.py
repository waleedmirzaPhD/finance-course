"""Gym-style portfolio environment for the DQN agent.

Discrete action space: a finite menu of candidate weight vectors over a
small basket of assets. This keeps the agent learnable in finite time
on small data while still teaching the RL plumbing end-to-end.

State: a flat vector concatenating recent (window) log returns of each
asset in the basket.

Reward: next-step portfolio return minus an L1 transaction cost on the
weight change (5 bps per side by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PortfolioEnv:
    returns: pd.DataFrame
    actions: np.ndarray             # (n_actions, n_assets), weights per action
    window: int = 21
    cost_bps_per_side: float = 5.0
    seed: int = 0

    def __post_init__(self) -> None:
        assert self.actions.shape[1] == self.returns.shape[1], \
            "each action must have one weight per asset"
        self._dates = self.returns.index
        self._r = self.returns.to_numpy().astype(np.float32)
        self.n_assets = self._r.shape[1]
        self.state_dim = self.window * self.n_assets
        self.n_actions = self.actions.shape[0]
        self._rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        if start_idx is None:
            start_idx = self._rng.integers(self.window,
                                            len(self._r) - 1)
        self._t = start_idx
        self._last_w = np.zeros(self.n_assets, dtype=np.float32)
        return self._observe()

    def _observe(self) -> np.ndarray:
        return self._r[self._t - self.window:self._t].ravel().astype(np.float32)

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict]:
        w_new = self.actions[action_idx].astype(np.float32)
        turnover = float(np.abs(w_new - self._last_w).sum())
        cost = turnover * (self.cost_bps_per_side / 1e4)
        pnl = float(w_new @ self._r[self._t])
        reward = pnl - cost
        self._last_w = w_new
        self._t += 1
        done = self._t >= len(self._r) - 1
        return self._observe(), reward, done, {"turnover": turnover, "pnl": pnl}


def default_action_menu(n_assets: int) -> np.ndarray:
    """Generate a small set of candidate weight vectors:
        - cash (all zeros)
        - equal-weight long
        - long single asset (i)  for each asset
        - short single asset (i) for each asset
    For n_assets=5 this gives 1 + 1 + 5 + 5 = 12 actions.
    """
    a = [np.zeros(n_assets), np.full(n_assets, 1.0 / n_assets)]
    for i in range(n_assets):
        v = np.zeros(n_assets); v[i] = 1.0;  a.append(v)
        v = np.zeros(n_assets); v[i] = -1.0; a.append(v)
    return np.vstack(a)
