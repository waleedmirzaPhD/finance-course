"""
Lecture 01 — Sum of Uniform Random Variables (Irwin-Hall Distribution)
=======================================================================
Topic:
    When you add n independent Uniform(0,1) random variables together, the
    resulting distribution is called the Irwin-Hall distribution. This script:

    1. Derives and plots the *exact* analytical PDF for n=2 (a triangle).
    2. Uses Monte Carlo simulation to verify the result for n=2, 4, and 10.
    3. Illustrates the Central Limit Theorem (CLT): as n grows, the sum
       increasingly resembles a Normal distribution.

Finance relevance:
    - Random walks and asset-return models often aggregate many small,
      independent shocks. Understanding how sums of random variables behave
      is foundational to option pricing, risk modelling, and portfolio theory.
    - Monte Carlo simulation is one of the most widely used tools in
      quantitative finance (VaR, option pricing, stress testing).

Key concepts:
    - Convolution of PDFs
    - Irwin-Hall distribution
    - Central Limit Theorem (CLT)
    - Monte Carlo method
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform  # available if needed for further extensions


# ─────────────────────────────────────────────────────────────────────────────
# 1. Analytical triangular PDF for sum of two Uniform(0,1) variables
# ─────────────────────────────────────────────────────────────────────────────

def triangular_pdf(s: np.ndarray) -> np.ndarray:
    """
    Exact PDF of S = X1 + X2 where X1, X2 ~ Uniform(0, 1) i.i.d.

    Derivation (convolution):
        f_S(s) = integral_{-inf}^{inf} f_X(x) * f_X(s - x) dx

        Since each f_X is 1 on [0,1] and 0 elsewhere, the integrand is 1
        only when both x in [0,1] and (s-x) in [0,1], i.e. x in [max(0,s-1),
        min(1,s)]. Evaluating the length of that interval gives:

            f_S(s) = s         for 0 <= s <= 1   (rising ramp)
            f_S(s) = 2 - s     for 1 <  s <= 2   (falling ramp)
            f_S(s) = 0         otherwise

        The result is a symmetric triangle on [0, 2] with peak f_S(1) = 1.
        This is the Irwin-Hall PDF for n=2.

    Parameters
    ----------
    s : np.ndarray
        Array of evaluation points (typically linspace over [0, 2]).

    Returns
    -------
    np.ndarray
        PDF values at each point in s.
    """
    pdf = np.zeros_like(s, dtype=float)  # start with zeros (handles tails automatically)

    # Rising ramp: probability mass accumulates linearly as s increases from 0 to 1
    mask1 = (s >= 0) & (s <= 1)
    pdf[mask1] = s[mask1]

    # Falling ramp: probability mass decreases linearly as s goes from 1 to 2
    mask2 = (s > 1) & (s <= 2)
    pdf[mask2] = 2.0 - s[mask2]

    return pdf


# ─────────────────────────────────────────────────────────────────────────────
# 2. Monte Carlo verification
# ─────────────────────────────────────────────────────────────────────────────

def simulate_uniform_sum(n_vars: int, n_samples: int = 100_000,
                         seed: int = 42) -> np.ndarray:
    """
    Simulate the sum S = X_1 + X_2 + ... + X_n, each X_i ~ Uniform(0, 1).

    Monte Carlo idea:
        Instead of computing the PDF analytically (which is hard for large n),
        we draw many random samples and let the empirical histogram approximate
        the true density. With enough samples, the histogram converges to the
        PDF by the Law of Large Numbers.

    Irwin-Hall properties (exact, for any n):
        Mean:     E[S] = n / 2
        Variance: Var[S] = n / 12
        Std dev:  σ = sqrt(n / 12)

    CLT implication:
        As n → ∞, (S - n/2) / sqrt(n/12)  →  N(0, 1) in distribution.
        Even for n=10, the histogram looks nearly Gaussian.

    Parameters
    ----------
    n_vars : int
        Number of Uniform(0,1) variables to sum (the 'n' in Irwin-Hall).
    n_samples : int
        Number of Monte Carlo draws (more = smoother histogram).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        1-D array of length n_samples, each entry is one realisation of S.
    """
    rng = np.random.default_rng(seed)  # modern NumPy RNG (preferred over np.random.seed)

    # Draw an (n_samples × n_vars) matrix of independent Uniform(0,1) values,
    # then sum across columns to get one S value per row.
    samples = rng.uniform(0, 1, size=(n_samples, n_vars))
    return samples.sum(axis=1)  # shape: (n_samples,)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot: analytical vs Monte Carlo for n = 2, 4, 10
# ─────────────────────────────────────────────────────────────────────────────

# Three side-by-side panels, one per value of n
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Sum of n Uniform(0,1) Random Variables", fontsize=13)

for ax, n in zip(axes, [2, 4, 10]):

    # ── Monte Carlo simulation ──────────────────────────────────────────────
    samples = simulate_uniform_sum(n_vars=n, n_samples=200_000)

    # density=True rescales the histogram so its area = 1, making it a valid
    # empirical PDF that can be compared directly with the analytical curve.
    ax.hist(samples, bins=80, density=True, alpha=0.5, color='steelblue',
            label='Monte Carlo')

    # ── Theoretical moments from the Irwin-Hall distribution ────────────────
    theo_mean = n / 2            # E[S] = n * E[X] = n * 0.5
    theo_std  = np.sqrt(n / 12)  # Var[S] = n * Var[X] = n * 1/12  →  σ = sqrt(n/12)

    # ── Exact analytical PDF overlay (only tractable to plot for n=2) ───────
    if n == 2:
        s_vals = np.linspace(0, 2, 500)  # dense grid over the support [0, 2]
        ax.plot(s_vals, triangular_pdf(s_vals), 'r-', lw=2,
                label='Exact (triangular)')
        # The red line should sit perfectly on top of the blue histogram,
        # confirming that the Monte Carlo correctly recovers the true PDF.

    # ── Annotate mean and ±1σ band ──────────────────────────────────────────
    # Dashed vertical line at the theoretical mean
    ax.axvline(theo_mean, color='navy', linestyle='--', lw=1.5,
               label=f'Mean = {theo_mean:.1f}')

    # Shaded band: one standard deviation either side of the mean.
    # In finance, the ±1σ band is analogous to a 68% confidence interval
    # or a rough 1-day VaR region under a Normal approximation.
    ax.axvspan(theo_mean - theo_std, theo_mean + theo_std,
               alpha=0.1, color='navy', label=f'±1σ (σ={theo_std:.3f})')

    # ── Labels ──────────────────────────────────────────────────────────────
    ax.set_title(f'n = {n}', fontsize=11)
    ax.set_xlabel('Sum S')
    ax.set_ylabel('Density' if n == 2 else '')  # only label the leftmost y-axis
    ax.legend(fontsize=8)

plt.tight_layout()

# Save to disk so the figure can be included in notes / the repo
plt.savefig('uniform_sum_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
