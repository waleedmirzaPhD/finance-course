import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# ─────────────────────────────────────────────────────────────────────────────
# 1. Analytical triangular PDF for sum of two Uniform(0,1) variables
# ─────────────────────────────────────────────────────────────────────────────

def triangular_pdf(s: np.ndarray) -> np.ndarray:
    """
    Exact PDF of S = X1 + X2 where X1, X2 ~ Uniform(0, 1) i.i.d.
    This is the result of the convolution f_{X1} * f_{X2}.
    Support: [0, 2], peak at s = 1.
    """
    pdf = np.zeros_like(s, dtype=float)
    # Case 1: rising ramp
    mask1 = (s >= 0) & (s <= 1)
    pdf[mask1] = s[mask1]
    # Case 2: falling ramp
    mask2 = (s > 1) & (s <= 2)
    pdf[mask2] = 2.0 - s[mask2]
    return pdf


# ─────────────────────────────────────────────────────────────────────────────
# 2. Monte Carlo verification
# ─────────────────────────────────────────────────────────────────────────────

def simulate_uniform_sum(n_vars: int, n_samples: int = 100_000,
                         seed: int = 42) -> np.ndarray:
    """
    Simulate the sum S = X_1 + ... + X_n where each X_i ~ Uniform(0, 1).
    Returns an array of n_samples realizations of S.
    """
    rng = np.random.default_rng(seed)
    samples = rng.uniform(0, 1, size=(n_samples, n_vars))
    return samples.sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot: analytical vs Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Sum of n Uniform(0,1) Random Variables", fontsize=13)

for ax, n in zip(axes, [2, 4, 10]):
    samples = simulate_uniform_sum(n_vars=n, n_samples=200_000)

    # Histogram (density=True normalises to a PDF)
    ax.hist(samples, bins=80, density=True, alpha=0.5, color='steelblue',
            label='Monte Carlo')

    # Theoretical mean and std (Irwin-Hall distribution)
    theo_mean = n / 2
    theo_std  = np.sqrt(n / 12)

    # For n=2 overlay the exact analytical PDF
    if n == 2:
        s_vals = np.linspace(0, 2, 500)
        ax.plot(s_vals, triangular_pdf(s_vals), 'r-', lw=2,
                label='Exact (triangular)')

    ax.axvline(theo_mean, color='navy', linestyle='--', lw=1.5,
               label=f'Mean = {theo_mean:.1f}')
    ax.axvspan(theo_mean - theo_std, theo_mean + theo_std,
               alpha=0.1, color='navy', label=f'±1σ (σ={theo_std:.3f})')

    ax.set_title(f'n = {n}', fontsize=11)
    ax.set_xlabel('Sum S')
    ax.set_ylabel('Density' if n == 2 else '')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('uniform_sum_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
