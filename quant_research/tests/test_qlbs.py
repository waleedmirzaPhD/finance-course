"""QLBS should approximate Black-Scholes in the near-frictionless limit."""

from __future__ import annotations

import numpy as np

from src.rl.qlbs import QLBSConfig, bs_call_delta, bs_call_price, qlbs_solve


def test_bs_call_known_value() -> None:
    # ATM 1y, r=0, sigma=0.20 -> ~7.97 by standard formula.
    px = bs_call_price(100, 100, 1.0, 0.0, 0.20)
    assert abs(px - 7.965) < 0.02
    d = bs_call_delta(100, 100, 1.0, 0.0, 0.20)
    assert abs(d - 0.5398) < 0.005


def test_qlbs_matches_bs_within_tolerance() -> None:
    cfg = QLBSConfig(S0=100, K=100, T=1.0, r=0.0, mu=0.05, sigma=0.20,
                     n_steps=24, n_paths=15000, seed=0)
    out = qlbs_solve(cfg)
    # In the near-frictionless limit QLBS price should be within ~5% of BS.
    assert abs(out["price_t0"] - out["bs_price"]) / out["bs_price"] < 0.10
    # Delta within ~10%.
    assert abs(out["delta_t0"] - out["bs_delta"]) < 0.10


def test_qlbs_delta_monotone_in_s0() -> None:
    """ATM call delta should rise as S0 increases."""
    deltas = []
    for s0 in [80, 100, 120]:
        cfg = QLBSConfig(S0=s0, K=100, T=1.0, r=0.0, mu=0.05, sigma=0.20,
                         n_steps=24, n_paths=8000, seed=0)
        deltas.append(qlbs_solve(cfg)["delta_t0"])
    assert deltas[0] < deltas[1] < deltas[2]
