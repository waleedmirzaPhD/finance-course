"""
Lecture 01 — Binomial Distribution: Moments, Monte Carlo, and √n Scaling
=========================================================================
Topic:
    The Binomial(n, p) distribution counts the number of successes in n
    independent Bernoulli(p) trials. This script:

    1. Computes the exact analytical mean and variance using closed-form formulas.
    2. Verifies these formulas by simulating the Binomial as a sum of Bernoullis
       (Monte Carlo), demonstrating agreement between theory and simulation.
    3. Visualises how the standard deviation σ scales with n — a fundamental
       result with direct applications in finance and risk management.

Finance relevance:
    - A Bernoulli trial models a single binary event: a bond defaults or it
      doesn't, a trade wins or loses, an option expires in-the-money or not.
    - The Binomial distribution naturally arises in credit risk (number of
      defaults in a portfolio of n loans, each with default probability p)
      and in options pricing (the Cox-Ross-Rubinstein binomial tree).
    - The √n scaling of σ is the mathematical reason why diversification
      reduces risk: adding more uncorrelated positions grows the standard
      deviation only as √n, not linearly with n.

Key concepts:
    - Bernoulli(p): takes value 1 with probability p, 0 with probability q = 1-p.
      E[X] = p,  Var(X) = pq.
    - Binomial(n, p): sum of n i.i.d. Bernoulli(p) variables.
      E[S] = np,  Var(S) = npq,  σ(S) = √(npq).
    - √n scaling: σ grows with √n, not n — doubling n only increases σ by √2.
    - Monte Carlo verification: simulate thousands of experiments and compare
      sample moments to the analytical formulas.
"""

import numpy as np
from scipy import stats   # available for extensions (e.g. stats.binom.pmf)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Analytical mean and variance of a Binomial(n, p)
# ─────────────────────────────────────────────────────────────────────────────

def binomial_moments(n: int, p: float) -> dict:
    """
    Compute the exact mean, variance, and standard deviation of Binomial(n, p).

    Derivation from first principles:
        Let S = X_1 + X_2 + ... + X_n, each X_i ~ Bernoulli(p).

        Mean (linearity of expectation — holds even for dependent variables):
            E[S] = E[X_1] + ... + E[X_n] = n * p

        Variance (independence required to split the sum):
            Var(S) = Var(X_1) + ... + Var(X_n)   [because X_i are independent]
                   = n * Var(X_i)
                   = n * p * q                     [since Var(Bernoulli) = pq]

        Standard deviation:
            σ(S) = √(npq)

    The q = 1 - p notation is standard shorthand inherited from the original
    Bernoulli literature; it represents the probability of *failure* per trial.

    Parameters
    ----------
    n : int
        Number of independent Bernoulli trials.
    p : float
        Probability of success on each trial.  Must be in [0, 1].

    Returns
    -------
    dict with keys 'mean', 'variance', 'std_dev'.
    """
    q = 1 - p                    # probability of failure (complement of p)
    mean     = n * p             # E[S] = np
    variance = n * p * q         # Var(S) = npq
    std_dev  = np.sqrt(variance) # σ(S) = √(npq)
    return {"mean": mean, "variance": variance, "std_dev": std_dev}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Monte Carlo verification via sum-of-Bernoullis
# ─────────────────────────────────────────────────────────────────────────────

def binomial_via_bernoulli_sum(
    n: int,
    p: float,
    n_samples: int = 500_000,
    seed: int = 42
) -> dict:
    """
    Simulate Binomial(n, p) as a sum of n independent Bernoulli(p) variables,
    then compare the sample moments to the analytical formulas.

    Why simulate Bernoulli sums instead of drawing directly from Binomial?
        This mirrors the *mathematical definition* of the Binomial: it IS a sum
        of Bernoullis. Constructing it this way makes the connection explicit,
        and confirms that the analytical formulas (derived from that same
        construction) match what we observe in simulation.

    Matrix approach for efficiency:
        Instead of looping over n_samples experiments, we build a 2-D matrix
        of shape (n_samples, n). Each row represents one experiment of n trials.
        Summing across columns (axis=1) gives us n_samples realisations of S
        simultaneously — far faster than a Python for-loop.

        X[i, j] = result of trial j in experiment i  (0 or 1)
        S[i]    = X[i,0] + X[i,1] + ... + X[i,n-1]  (successes in experiment i)

    Parameters
    ----------
    n : int
        Number of Bernoulli trials per experiment (Binomial parameter).
    p : float
        Success probability per trial (Binomial parameter).
    n_samples : int
        Number of independent experiments to simulate.
        500,000 is enough to get sample moments accurate to ~3 decimal places.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with sample moments and the analytical benchmark.
    """
    rng = np.random.default_rng(seed)  # seeded generator — results are reproducible

    # Draw n_samples × n independent Bernoulli(p) outcomes.
    # rng.binomial(1, p, size=...) draws from Bernoulli(p) — a Binomial with
    # n_trials=1, which is exactly one coin flip with probability p of heads.
    # Shape: (n_samples, n)
    X = rng.binomial(1, p, size=(n_samples, n))

    # Sum each row: S[i] = number of successes in experiment i
    # Shape: (n_samples,) — one realisation of S per experiment
    S = X.sum(axis=1)

    return {
        "sample_mean":     S.mean(),       # should be close to n*p
        "sample_variance": S.var(),        # should be close to n*p*q
        "sample_std":      S.std(),        # should be close to sqrt(n*p*q)
        "analytical":      binomial_moments(n, p)  # exact theoretical values
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualise the √n scaling of σ
# ─────────────────────────────────────────────────────────────────────────────

def sigma_vs_n(p: float = 0.5, n_max: int = 1000) -> None:
    """
    Plot σ(S) = √(npq) as a function of n, and confirm the √n scaling by
    plotting σ / √n, which should be the constant √(pq).

    Why does √n scaling matter in finance?
        - Portfolio risk: if you hold n uncorrelated assets each with volatility σ_0,
          the portfolio standard deviation is σ_0 * √n, NOT σ_0 * n.
          Dividing by n positions (equal weights), the per-asset risk contribution
          falls as 1/√n — this is the mathematical basis of diversification.
        - Sample statistics: the standard error of a sample mean is σ/√n, meaning
          you need 4× as many observations to halve estimation uncertainty.
        - Options pricing: in the binomial tree model, up/down moves are chosen so
          that the tree volatility matches σ√(Δt), a direct application of √n scaling.

    Left panel — σ vs n:
        Shows the concave (decelerating) growth of σ with n.
        The curve bends because √n grows slower than n.

    Right panel — σ/√n vs n:
        Dividing out the √n factor should leave a flat horizontal line at √(pq).
        This confirms that the *only* source of growth in σ is the √n factor.

    Parameters
    ----------
    p : float
        Success probability used for the plot (default 0.5 = fair coin).
    n_max : int
        Maximum number of trials to plot on the x-axis.
    """
    import matplotlib.pyplot as plt

    # Generate all integer n values from 1 to n_max
    n_vals = np.arange(1, n_max + 1)

    # Analytical σ for each n: σ(n) = √(n·p·q)
    sigma  = np.sqrt(n_vals * p * (1 - p))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Scaling of Binomial Std Dev with n  (p = {p})", fontsize=12)

    # ── Left panel: raw σ vs n ───────────────────────────────────────────────
    ax1.plot(n_vals, sigma, color='steelblue', lw=2)
    ax1.set_xlabel('n (number of trials)')
    ax1.set_ylabel('σ = √(npq)')
    ax1.set_title('Standard deviation grows as √n')
    # The concave shape visually confirms sub-linear (√n) growth:
    # going from n=1 to n=100 multiplies σ by 10, but from n=100 to n=400
    # only multiplies it by 2 — each additional trial contributes less.

    # ── Right panel: σ/√n vs n — should be flat ─────────────────────────────
    # If σ = √(npq) then σ/√n = √(pq), a constant independent of n.
    # A flat line here is the "proof" that the growth is exactly √n, no more.
    ax2.plot(n_vals, sigma / np.sqrt(n_vals), color='firebrick', lw=2)

    # Theoretical constant level √(pq)
    ax2.axhline(np.sqrt(p * (1 - p)), color='black', linestyle='--',
                label=f'√(pq) = {np.sqrt(p*(1-p)):.3f}')
    ax2.set_xlabel('n')
    ax2.set_ylabel('σ / √n')
    ax2.set_title('σ / √n is constant — confirms √n scaling')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('binomial_std_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Run everything
# ─────────────────────────────────────────────────────────────────────────────

# Test case: n=20 trials, p=0.3
# Analytical: E[S] = 20*0.3 = 6.0,  Var(S) = 20*0.3*0.7 = 4.2,  σ = √4.2 ≈ 2.049
result = binomial_via_bernoulli_sum(n=20, p=0.3)

print("=== Binomial(20, 0.3) ===")
print(f"  Analytical mean:  {result['analytical']['mean']:.4f}")
print(f"  Sample mean:      {result['sample_mean']:.4f}")
print(f"  Analytical var:   {result['analytical']['variance']:.4f}")
print(f"  Sample var:       {result['sample_variance']:.4f}")
# With 500,000 samples the sample moments should match the analytical values
# to at least 2-3 decimal places — any larger discrepancy would suggest a bug.

# Plot σ vs n and σ/√n vs n for a fair coin (p=0.5)
sigma_vs_n(p=0.5)
