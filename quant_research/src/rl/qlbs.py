"""QLBS — Q-Learner for the Black-Scholes world (Halperin, 2017).

Discrete-time hedging of a European call on a GBM-driven stock. At each
of T time steps the agent picks a hedge u_t (shares of underlying).
The state is (t, S_t). The reward is the negative incremental tracking
error of the hedged portfolio against the option payoff.

In the Black-Scholes limit (continuous time, frictionless), the optimal
Q-function recovers the BS price and the optimal action recovers the BS
delta. We validate by comparing the QLBS price/delta at t=0 to the
closed-form BS values.

This is the canonical RL-in-finance toy problem and the signature
example from the Coursera "RL in Finance" course. See:
    Halperin, "QLBS: Q-Learner in the Black-Scholes(-Merton) Worlds"
    (Journal of Derivatives, 2020).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))


@dataclass
class QLBSConfig:
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.0            # risk-free
    mu: float = 0.05          # real-world drift for path simulation
    sigma: float = 0.15
    n_steps: int = 24         # time grid (e.g. monthly with T=1y)
    n_paths: int = 20_000
    risk_aversion: float = 0.001  # lambda in QLBS, small for near-BS limit
    seed: int = 0


def simulate_gbm_paths(cfg: QLBSConfig) -> np.ndarray:
    """Return (n_paths, n_steps+1) GBM path matrix."""
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.T / cfg.n_steps
    z = rng.standard_normal((cfg.n_paths, cfg.n_steps))
    log_inc = (cfg.mu - 0.5 * cfg.sigma ** 2) * dt + cfg.sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((cfg.n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1
    )
    return cfg.S0 * np.exp(log_paths)


def _basis(S: np.ndarray) -> np.ndarray:
    """Polynomial basis in normalized log-price.

    Columns: [1, x, x^2, x^3] with x = log(S / 100) standardized roughly.
    For a clean QLBS demo, simple polynomial basis is enough.
    """
    x = np.log(S / 100.0)
    return np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])


def qlbs_solve(cfg: QLBSConfig) -> dict:
    """Fitted-Q value iteration for the optimal hedge.

    Uses Halperin's (2020) closed-form per-step quadratic optimization in
    the hedge action -- we don't need a function approximator over actions
    because Q is quadratic in u_t at each step and can be minimized
    analytically.

    Implementation follows the recursive least-squares scheme:
        At each t (backward), regress the per-path realized hedged loss
        on basis(S_t) to estimate Q_t coefficients, choose u_t to
        minimize, then propagate value backward.

    Returns:
        price_t0   : QLBS option price at t=0
        delta_t0   : QLBS hedge at t=0 averaged over paths near S0
        bs_price   : Black-Scholes analytic price
        bs_delta   : Black-Scholes analytic delta
        hedges     : (n_paths, n_steps) realized hedges
        paths      : (n_paths, n_steps+1) simulated paths
    """
    S = simulate_gbm_paths(cfg)
    dt = cfg.T / cfg.n_steps
    disc = np.exp(-cfg.r * dt)

    n_paths = cfg.n_paths
    n_steps = cfg.n_steps

    # Terminal portfolio value (we set initial cash so portfolio = option payoff
    # at terminal; we compute pi_t recursively backward).
    payoff = np.maximum(S[:, -1] - cfg.K, 0.0)
    pi = payoff.copy()           # portfolio value at terminal
    hedges = np.zeros((n_paths, n_steps))

    # Backward sweep. At each t we have S_t and we want to choose u_t
    # that minimizes  E[(pi_{t+1} - u_t * (S_{t+1} - S_t)*exp(-r dt)
    #                    - pi_t)^2] + lambda * Var(...)
    # In Halperin's QLBS, the optimal u_t at each t is the regression of
    # discounted next-step pi change on (S_{t+1} - S_t). We regress in the
    # basis of S_t for path-dependent hedge.
    for t in range(n_steps - 1, -1, -1):
        dS = S[:, t + 1] - S[:, t]                # next-step move
        target_pi = pi * disc                     # discounted pi_{t+1}

        phi = _basis(S[:, t])                     # (n_paths, p)
        # Regression: estimate E[target_pi * dS | S_t] and E[dS^2 | S_t]
        # then optimal u_t(S_t) = E[target * dS] / E[dS^2].
        # We approximate both conditional expectations via linear regression
        # on phi(S_t).
        A = phi.T @ phi + 1e-6 * np.eye(phi.shape[1])
        # Numerator coefs.
        beta_num = np.linalg.solve(A, phi.T @ (target_pi * dS))
        beta_den = np.linalg.solve(A, phi.T @ (dS ** 2))
        num = phi @ beta_num
        den = phi @ beta_den
        u = num / (den + 1e-9)
        hedges[:, t] = u

        # Update portfolio value backward: pi_t = pi_{t+1}*disc - u_t * dS * disc.
        pi = target_pi - u * dS * disc

    # Option price at t=0 is mean(pi_0). For lambda>0 we'd add a variance
    # penalty term; we keep it simple in the near-BS limit.
    price_t0 = float(pi.mean())
    delta_t0 = float(hedges[:, 0].mean())

    return {
        "price_t0": price_t0,
        "delta_t0": delta_t0,
        "bs_price": bs_call_price(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.sigma),
        "bs_delta": bs_call_delta(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.sigma),
        "hedges": hedges,
        "paths": S,
    }
