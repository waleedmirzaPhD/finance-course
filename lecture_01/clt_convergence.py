"""
Lecture 01 — Central Limit Theorem: Convergence to Normality
=============================================================
Topic:
    This script generalises the previous exercise. Instead of fixing n=2,4,10
    and plotting side-by-side, we build a *reusable* Monte Carlo engine that
    accepts *any* distribution callable and any list of n values.

    The key question answered visually:
        "How quickly does the sum of n i.i.d. Uniform(0,1) variables start
         to look like a Gaussian?"

    Answer (CLT): remarkably fast — by n=5 the fit is already very close,
    and by n=20 it is nearly indistinguishable from a Normal.

Finance relevance:
    - Daily portfolio returns are often modelled as the sum of many small,
      independent stock-level returns → Normal approximation is justified by CLT.
    - The CLT underpins Black-Scholes (log-returns are sums of tiny shocks),
      Value-at-Risk (VaR) calculations, and the standardisation of risk factors.
    - However, real returns have fat tails (the CLT kicks in slowly for
      heavy-tailed distributions), which is why risk managers must go beyond
      the Normal assumption for extreme events.

Key concepts:
    - Central Limit Theorem (CLT): sum of n i.i.d. RVs with finite mean μ
      and variance σ² converges in distribution to N(n·μ, n·σ²) as n → ∞.
    - Irwin-Hall distribution: exact distribution of the sum of n Uniform(0,1) RVs.
    - Reusable Monte Carlo engine: passing a distribution as a callable lets us
      swap in any distribution (Exponential, Bernoulli, etc.) without rewriting logic.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats   # used for stats.norm.pdf — the Gaussian benchmark


# ─────────────────────────────────────────────────────────────────────────────
# General Monte Carlo simulator: sum of n i.i.d. random variables
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_sum(
    distribution: callable,
    n_vars: int,
    n_samples: int = 200_000,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate S = X_1 + X_2 + ... + X_n for i.i.d. X_i drawn from `distribution`.

    Design choice — accepting a callable:
        By passing the distribution as a function argument (a "strategy" pattern),
        this engine is completely agnostic about the underlying distribution.
        You can plug in Uniform, Exponential, Bernoulli, Student-t, etc. without
        changing any internal logic.  This mirrors how quant libraries expose
        generic simulation engines.

    How the matrix trick works:
        We call `distribution(rng, size=n_samples)` once per variable, producing
        a 1-D array of n_samples draws.  Stacking n such arrays as columns gives
        an (n_samples × n_vars) matrix.  Summing across columns (axis=1) gives
        one realisation of S per row — equivalent to running the experiment
        n_samples times simultaneously, which is far faster than a Python loop.

    Parameters
    ----------
    distribution : callable
        A function with signature (rng, size) -> np.ndarray.
        Example: `lambda rng, size: rng.uniform(0, 1, size=size)`
        The `rng` argument is the seeded NumPy Generator, ensuring reproducibility.
    n_vars : int
        Number of independent copies to sum (the 'n' in the CLT statement).
    n_samples : int
        Number of independent Monte Carlo realisations of S to generate.
        More samples → smoother histogram → better approximation of the true PDF.
    seed : int
        Seed for the NumPy random Generator.  Fixing the seed makes results
        reproducible across runs — important in research and backtesting.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Each element is one independent realisation of S = X_1 + ... + X_n.
    """
    rng = np.random.default_rng(seed)  # modern NumPy RNG (safer than legacy np.random.seed)

    # Build matrix: each column is n_samples draws from the distribution.
    # np.column_stack assembles a list of 1-D arrays into a 2-D matrix by
    # treating each array as a column — equivalent to np.vstack(...).T.
    samples = np.column_stack([
        distribution(rng, size=n_samples) for _ in range(n_vars)
    ])
    # samples.shape == (n_samples, n_vars)

    # Sum each row: turns the (n_samples × n_vars) matrix into a (n_samples,) vector.
    # This is S_i = X_{i,1} + X_{i,2} + ... + X_{i,n_vars} for each simulation i.
    return samples.sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation: show CLT convergence across multiple values of n
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_to_normal(n_values: list[int],
                               distribution: callable,
                               dist_name: str = "Uniform(0,1)"):
    """
    Plot histograms of the simulated sum S for each n in n_values, and overlay
    the Gaussian approximation predicted by the CLT.

    What to look for in the output:
        - n=1: the raw distribution shape (uniform rectangle for Uniform inputs).
        - n=2: triangular — exactly the Irwin-Hall PDF derived in the previous script.
        - n=5: already bell-shaped; the Normal overlay fits reasonably well.
        - n=20: nearly perfect Gaussian; CLT is fully in effect.

    The red curve is NOT fitted to the data — it uses the *theoretical* mean
    and standard deviation from the Irwin-Hall formulas:
        μ = n/2,  σ = sqrt(n/12)
    If the simulation is correct, the red curve should match the histogram.

    Parameters
    ----------
    n_values : list[int]
        List of n values to display (one subplot per value).
    distribution : callable
        Same callable passed to monte_carlo_sum.
    dist_name : str
        Human-readable name shown in the figure title (e.g. "Uniform(0,1)").
    """
    # Create one subplot per n value, laid out in a single row
    fig, axes = plt.subplots(1, len(n_values), figsize=(4 * len(n_values), 4))
    fig.suptitle(f"Distribution of sum of {dist_name} variables — CLT convergence",
                 fontsize=12)

    for ax, n in zip(axes, n_values):

        # ── Monte Carlo simulation ──────────────────────────────────────────
        samples = monte_carlo_sum(distribution, n_vars=n, n_samples=300_000)
        # 300,000 samples gives a smooth histogram even for the tails

        # ── Theoretical moments (exact for Irwin-Hall / Uniform sum) ────────
        # For X ~ Uniform(0,1): E[X] = 0.5, Var[X] = 1/12
        # By linearity of expectation and independence:
        #   E[S] = n * 0.5
        #   Var[S] = n * (1/12)  →  σ_S = sqrt(n/12)
        mu    = n * 0.5
        sigma = np.sqrt(n / 12)

        # ── Histogram ───────────────────────────────────────────────────────
        # density=True ensures area under histogram = 1, so it can be compared
        # directly to the theoretical PDF (which also integrates to 1).
        ax.hist(samples, bins=100, density=True, alpha=0.45,
                color='steelblue', label='Simulation')

        # ── Normal (Gaussian) overlay — the CLT prediction ──────────────────
        # We evaluate the Normal PDF on a fine grid spanning the simulated range.
        # The parameters (mu, sigma) come from theory, not from fitting to data.
        # Seeing the red curve align with the blue histogram confirms the CLT.
        x = np.linspace(samples.min(), samples.max(), 500)
        ax.plot(x, stats.norm.pdf(x, loc=mu, scale=sigma), 'r-', lw=2,
                label=f'N(μ={mu:.1f}, σ={sigma:.2f})')

        # ── Labels and formatting ────────────────────────────────────────────
        ax.set_title(f'n = {n}', fontsize=11)
        ax.set_xlabel('S')

        # Only label the y-axis on the leftmost panel to avoid clutter
        if n == n_values[0]:
            ax.set_ylabel('Density')

        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('clt_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

# Define the distribution as a lambda that matches the (rng, size) signature.
# Using rng.uniform (from the seeded Generator) rather than np.random.uniform
# ensures that the global random state is not touched — important when this
# module is imported alongside other simulations.
uniform_draw = lambda rng, size: rng.uniform(0, 1, size=size)

# Run the visualisation for n = 1, 2, 5, 20.
# n=1 shows the raw Uniform shape; n=20 shows near-perfect Gaussianity.
plot_convergence_to_normal(
    n_values=[1, 2, 5, 20],
    distribution=uniform_draw,
    dist_name="Uniform(0,1)"
)
