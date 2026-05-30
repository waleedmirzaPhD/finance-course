"""Minimal but real DQN agent for the portfolio env.

Architecture: 2-layer MLP Q-network with target network and uniform
experience replay. Deliberately small — the point is to demonstrate the
RL pipeline (Bellman backups, target sync, ε-greedy) end-to-end on
finite financial data, not to claim alpha.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


@dataclass
class DQNConfig:
    hidden: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 20000
    target_sync_steps: int = 200
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10000
    seed: int = 0


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.buf: Deque[Transition] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, t: Transition) -> None:
        self.buf.append(t)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        s = np.stack([b.s for b in batch]).astype(np.float32)
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        s2 = np.stack([b.s2 for b in batch]).astype(np.float32)
        d = np.array([b.done for b in batch], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int,
                 cfg: DQNConfig = field(default_factory=DQNConfig)
                 if False else DQNConfig()) -> None:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(cfg.seed)
        self.cfg = cfg
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.online = self._build_qnet()
        self.target = self._build_qnet()
        self.target.set_weights(self.online.get_weights())
        self.opt = tf.keras.optimizers.Adam(cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size, seed=cfg.seed)
        self.step_count = 0
        self.rng = np.random.default_rng(cfg.seed)

    def _build_qnet(self):
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = inputs
        for h in self.cfg.hidden:
            x = tf.keras.layers.Dense(h, activation="relu")(x)
        out = tf.keras.layers.Dense(self.n_actions, activation="linear")(x)
        return tf.keras.Model(inputs, out)

    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_actions))
        q = self.online(state[None, :], training=False).numpy()[0]
        return int(np.argmax(q))

    def remember(self, s, a, r, s2, done) -> None:
        self.replay.push(Transition(s, a, r, s2, done))

    def train_step(self) -> float | None:
        if len(self.replay) < self.cfg.batch_size:
            return None
        import tensorflow as tf
        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)
        q_next = self.target(s2, training=False).numpy().max(axis=1)
        target_q = r + (1.0 - d) * self.cfg.gamma * q_next
        a_oh = tf.one_hot(a, self.n_actions)
        with tf.GradientTape() as tape:
            q_pred = tf.reduce_sum(self.online(s, training=True) * a_oh, axis=1)
            loss = tf.reduce_mean(tf.square(target_q - q_pred))
        grads = tape.gradient(loss, self.online.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.online.trainable_variables))
        self.step_count += 1
        if self.step_count % self.cfg.target_sync_steps == 0:
            self.target.set_weights(self.online.get_weights())
        return float(loss)


def train_dqn(env, agent: DQNAgent, n_episodes: int = 50,
              max_steps: int = 200, log_every: int = 5) -> list[dict]:
    history = []
    for ep in range(n_episodes):
        s = env.reset()
        ep_reward = 0.0
        losses = []
        for _ in range(max_steps):
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.remember(s, a, r, s2, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            ep_reward += r
            s = s2
            if done:
                break
        history.append({
            "episode": ep,
            "reward": ep_reward,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "epsilon": agent.epsilon(),
        })
        if (ep + 1) % log_every == 0:
            print(f"  ep {ep+1:3d}  reward {ep_reward:+.4f}  "
                  f"loss {history[-1]['loss']:.4e}  eps {agent.epsilon():.3f}")
    return history


def evaluate_greedy(env, agent: DQNAgent, n_steps: int) -> dict:
    s = env.reset(start_idx=env.window)
    rewards, turnovers, actions = [], [], []
    for _ in range(n_steps):
        a = agent.act(s, greedy=True)
        s, r, done, info = env.step(a)
        rewards.append(r)
        turnovers.append(info["turnover"])
        actions.append(a)
        if done:
            break
    rewards_arr = np.asarray(rewards)
    return {
        "rewards": rewards_arr,
        "turnover": np.asarray(turnovers),
        "actions": np.asarray(actions),
        "total_return": float(np.exp(np.log1p(rewards_arr).sum()) - 1),
        "sharpe": float(rewards_arr.mean() / (rewards_arr.std() + 1e-9) *
                        np.sqrt(252)),
    }
