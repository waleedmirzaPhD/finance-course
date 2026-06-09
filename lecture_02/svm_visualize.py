"""
Step-by-step visualisation of svm_from_scratch.py.

Reproduces the exact same Lagrangian-dual linear-SVM workflow and draws
one figure for every conceptual step:

  01_dataset.png            — raw borrowers in the (DTI, credit_util) plane
  02_scaling.png            — StandardScaler effect, before vs after
  03_split.png              — stratified train / test split
  04_kernel_matrix.png      — Gram matrix K = X Xᵀ (sorted by class)
  05_alpha_solution.png     — αᵢ from the dual QP, sorted, with box bounds
  06_support_vectors.png    — training points with SVs highlighted
  07_weight_vector.png      — recovered w as a per-feature bar chart
  08_decision_boundary.png  — f = -1, 0, +1 contours in the 2-D slice
  09_primal_vs_dual.png     — f_primal == f_dual scatter (should be y = x)
  10_scratch_vs_sklearn.png — f_scratch vs f_sklearn scatter (≈ y = x)
  11_roc_and_margins.png    — ROC overlay + margin-score distribution

All plots land in ./svm_visualize/ (created if missing). Runtime ~30 s.
"""

import os
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ----------------------------------------------------------------------------
# Constants and reproducibility
# ----------------------------------------------------------------------------
PLOT_DIR     = "svm_visualize"
RANDOM_STATE = 42
C_VALUE      = 10.0
COLOR_REPAY  = "#4c72b0"   # steelblue
COLOR_DEFAULT = "#c44e52"  # indianred
COLOR_FREE_SV    = "#55a868"  # green
COLOR_BOUNDED_SV = "#dd8452"  # orange

os.makedirs(PLOT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)


def save_plot(filename: str) -> None:
    """Save current figure to PLOT_DIR with consistent dpi and close it."""
    plt.tight_layout()
    out = f"{PLOT_DIR}/{filename}"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  → {out}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ============================================================================
# 1. Build the same synthetic credit dataset
# ============================================================================
section("1. Loading the dataset")

n_samples          = 500
debt_to_income     = rng.beta(2, 5, n_samples) * 1.2
credit_utilization = rng.beta(2, 4, n_samples)
missed_payments    = rng.poisson(0.6, n_samples)
savings_ratio      = rng.beta(2, 5, n_samples) * 0.5
income_volatility  = np.abs(rng.normal(0.15, 0.1, n_samples))

X_dataframe = pd.DataFrame({
    "debt_to_income":     debt_to_income,
    "credit_utilization": credit_utilization,
    "missed_payments":    missed_payments,
    "savings_ratio":      savings_ratio,
    "income_volatility":  income_volatility,
})
feature_names = list(X_dataframe.columns)

# Strong-signal logistic DGP (matches svm_from_scratch.py).
logit = (
    -4.0
    + 8.0 * debt_to_income
    + 5.0 * credit_utilization
    + 2.0 * missed_payments
    - 8.0 * savings_ratio
    + 5.0 * income_volatility
)
p_default = 1.0 / (1.0 + np.exp(-logit))
y_01      = (rng.uniform(size=n_samples) < p_default).astype(int)
y_signed  = 2 * y_01 - 1

print(f"  n = {n_samples}, default rate = {y_01.mean():.2%}")


# ----------------------------------------------------------------------------
# Plot 01 — raw dataset on (DTI, credit_utilization)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(debt_to_income[y_01 == 0], credit_utilization[y_01 == 0],
           s=18, c=COLOR_REPAY, alpha=0.7, label="repay")
ax.scatter(debt_to_income[y_01 == 1], credit_utilization[y_01 == 1],
           s=18, c=COLOR_DEFAULT, alpha=0.7, label="default")
ax.set_xlabel("debt_to_income")
ax.set_ylabel("credit_utilization")
ax.set_title("Step 1 — raw borrowers in (DTI, credit_util) space\n"
             f"n = {n_samples}, default rate = {y_01.mean():.1%}")
ax.legend()
save_plot("01_dataset.png")


# ============================================================================
# 2. Stratified train/test split + scaling
# ============================================================================
section("2. Train/test split and scaling")

(X_train_raw, X_test_raw,
 y_train_01, y_test_01,
 y_train, y_test) = train_test_split(
    X_dataframe.values, y_01, y_signed,
    test_size=0.25, stratify=y_01, random_state=RANDOM_STATE,
)
n_train    = X_train_raw.shape[0]
n_features = X_train_raw.shape[1]
print(f"  train: {n_train} rows   test: {len(X_test_raw)} rows")
print(f"  default rate — train: {y_train_01.mean():.2%}   test: {y_test_01.mean():.2%}")

scaler         = StandardScaler().fit(X_train_raw)
X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)


# ----------------------------------------------------------------------------
# Plot 02 — scaling effect (before vs after) on (DTI, credit_util)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_train_raw[y_train_01 == 0, 0], X_train_raw[y_train_01 == 0, 1],
                s=14, c=COLOR_REPAY, alpha=0.6, label="repay")
axes[0].scatter(X_train_raw[y_train_01 == 1, 0], X_train_raw[y_train_01 == 1, 1],
                s=14, c=COLOR_DEFAULT, alpha=0.6, label="default")
axes[0].set_xlabel("debt_to_income (raw)")
axes[0].set_ylabel("credit_utilization (raw)")
axes[0].set_title(f"Before scaling\n"
                  f"means = ({X_train_raw[:,0].mean():.2f}, {X_train_raw[:,1].mean():.2f}), "
                  f"stds = ({X_train_raw[:,0].std():.2f}, {X_train_raw[:,1].std():.2f})")
axes[0].legend()

axes[1].scatter(X_train_scaled[y_train_01 == 0, 0], X_train_scaled[y_train_01 == 0, 1],
                s=14, c=COLOR_REPAY, alpha=0.6, label="repay")
axes[1].scatter(X_train_scaled[y_train_01 == 1, 0], X_train_scaled[y_train_01 == 1, 1],
                s=14, c=COLOR_DEFAULT, alpha=0.6, label="default")
axes[1].axhline(0, color="k", lw=0.5, alpha=0.4)
axes[1].axvline(0, color="k", lw=0.5, alpha=0.4)
axes[1].set_xlabel("debt_to_income (scaled)")
axes[1].set_ylabel("credit_utilization (scaled)")
axes[1].set_title(f"After StandardScaler\n"
                  f"means ≈ (0, 0), stds ≈ (1, 1)")
axes[1].legend()

plt.suptitle("Step 2 — feature scaling: same shapes, common units", y=1.02)
save_plot("02_scaling.png")


# ----------------------------------------------------------------------------
# Plot 03 — train/test split (stratified)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

axes[0].scatter(X_train_scaled[y_train_01 == 0, 0], X_train_scaled[y_train_01 == 0, 1],
                s=12, c=COLOR_REPAY, alpha=0.6, label="repay")
axes[0].scatter(X_train_scaled[y_train_01 == 1, 0], X_train_scaled[y_train_01 == 1, 1],
                s=12, c=COLOR_DEFAULT, alpha=0.6, label="default")
axes[0].set_title(f"Train ({n_train} rows, "
                  f"default rate = {y_train_01.mean():.1%})")
axes[0].set_xlabel("DTI (scaled)")
axes[0].set_ylabel("credit_util (scaled)")
axes[0].legend()

axes[1].scatter(X_test_scaled[y_test_01 == 0, 0], X_test_scaled[y_test_01 == 0, 1],
                s=12, c=COLOR_REPAY, alpha=0.6, label="repay")
axes[1].scatter(X_test_scaled[y_test_01 == 1, 0], X_test_scaled[y_test_01 == 1, 1],
                s=12, c=COLOR_DEFAULT, alpha=0.6, label="default")
axes[1].set_title(f"Test ({len(X_test_raw)} rows, "
                  f"default rate = {y_test_01.mean():.1%})")
axes[1].set_xlabel("DTI (scaled)")
axes[1].legend()

plt.suptitle("Step 3 — stratified split preserves class balance in both halves",
             y=1.02)
save_plot("03_split.png")


# ============================================================================
# 3. Build the linear kernel matrix and solve the dual QP (5-D model)
# ============================================================================
section("3. Solving the Lagrangian dual (5-D linear SVM)")

K_train = X_train_scaled @ X_train_scaled.T

alpha    = cp.Variable(n_train, nonneg=True)
alpha_y  = cp.multiply(alpha, y_train)
objective = cp.Maximize(
    cp.sum(alpha) - 0.5 * cp.quad_form(alpha_y, cp.psd_wrap(K_train))
)
constraints = [alpha <= C_VALUE, cp.sum(alpha_y) == 0]
cp.Problem(objective, constraints).solve()
alpha_value = np.asarray(alpha.value).ravel()
print(f"  dual objective = {(cp.sum(alpha).value - 0.5 * (alpha_value * y_train) @ K_train @ (alpha_value * y_train)):.3f}")

# Classify SVs
TOL          = 1e-5
sv_mask      = alpha_value > TOL
sv_indices   = np.where(sv_mask)[0]
free_mask    = (alpha_value > TOL) & (alpha_value < C_VALUE - TOL)
free_indices = np.where(free_mask)[0]
bounded_indices = np.where(alpha_value >= C_VALUE - TOL)[0]

# Recover w and b
alpha_y_train = alpha_value * y_train
w_scratch = alpha_y_train @ X_train_scaled
b_per_free_sv = (y_train[free_indices]
                 - X_train_scaled[free_indices] @ w_scratch)
b_scratch = float(np.mean(b_per_free_sv))

print(f"  n_SV={sv_indices.size}  free={free_indices.size}  bounded={bounded_indices.size}")
print(f"  b = {b_scratch:+.4f}   w = {w_scratch.round(3)}")


# ----------------------------------------------------------------------------
# Plot 04 — kernel matrix K = X Xᵀ, rows sorted by class for block structure
# ----------------------------------------------------------------------------
sort_order = np.argsort(-y_train)             # defaults first (+1), then repays (-1)
K_sorted   = K_train[sort_order][:, sort_order]
n_default_train = int((y_train == +1).sum())

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(K_sorted, cmap="coolwarm", vmin=-K_sorted.max(), vmax=K_sorted.max())
ax.axhline(n_default_train, color="black", lw=1.0)
ax.axvline(n_default_train, color="black", lw=1.0)
ax.set_title(f"Step 4 — Gram matrix K[i, j] = ⟨xᵢ, xⱼ⟩  ({n_train}×{n_train})\n"
             "rows/cols sorted so defaults come first, then repays")
ax.set_xlabel("j")
ax.set_ylabel("i")
fig.colorbar(im, ax=ax, label="K[i, j]")
save_plot("04_kernel_matrix.png")


# ----------------------------------------------------------------------------
# Plot 05 — dual solution αᵢ sorted descending, colour-coded by SV type
# ----------------------------------------------------------------------------
order = np.argsort(-alpha_value)
alpha_sorted = alpha_value[order]
colours = np.where(alpha_sorted >= C_VALUE - TOL, COLOR_BOUNDED_SV,
            np.where(alpha_sorted > TOL,           COLOR_FREE_SV, "lightgrey"))

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(range(n_train), alpha_sorted, color=colours, width=1.0)
ax.axhline(C_VALUE, color="black", linestyle="--", linewidth=1,
           label=f"upper box: C = {C_VALUE}")
ax.axhline(0,       color="black", linestyle="--", linewidth=1,
           label="lower box: 0")
ax.set_xlabel("training point (sorted by α descending)")
ax.set_ylabel(r"$\alpha_i$")
ax.set_title("Step 5 — dual solution α from the Lagrangian QP\n"
             f"bounded SVs (α=C): {bounded_indices.size}   "
             f"free SVs (0<α<C): {free_indices.size}   "
             f"non-SV (α=0): {n_train - sv_indices.size}")
legend_handles = [
    Patch(facecolor=COLOR_BOUNDED_SV, label=f"bounded SV (α = C)"),
    Patch(facecolor=COLOR_FREE_SV,    label=f"free SV (0 < α < C)"),
    Patch(facecolor="lightgrey",      label=f"not SV (α = 0)"),
]
ax.legend(handles=legend_handles, loc="upper right")
save_plot("05_alpha_solution.png")


# ----------------------------------------------------------------------------
# Plot 06 — recovered weight vector w as a per-feature bar chart
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
bar_colours = ["#c44e52" if v > 0 else "#4c72b0" for v in w_scratch]
ax.barh(feature_names, w_scratch, color=bar_colours)
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel(r"$w_k$  (recovered from $w = \sum_i \alpha_i y_i x_i$)")
ax.set_title("Step 6 — weight vector w (linear SVM, scaled features)\n"
             "positive (red) = pushes toward default;  negative (blue) = pushes toward repay")
save_plot("06_weight_vector.png")


# ============================================================================
# 4. 2-D companion model on (DTI, credit_util) for boundary visualisations
# ============================================================================
section("4. Fitting a parallel 2-D model for boundary plots")

X2_train = X_train_scaled[:, :2]              # only DTI + credit_util
K2_train = X2_train @ X2_train.T

alpha2   = cp.Variable(n_train, nonneg=True)
alpha2_y = cp.multiply(alpha2, y_train)
cp.Problem(
    cp.Maximize(cp.sum(alpha2) - 0.5 * cp.quad_form(alpha2_y, cp.psd_wrap(K2_train))),
    [alpha2 <= C_VALUE, cp.sum(alpha2_y) == 0],
).solve()
alpha2_value = np.asarray(alpha2.value).ravel()

# SVs of the 2-D model
sv2_mask    = alpha2_value > TOL
sv2_indices = np.where(sv2_mask)[0]
free2_mask  = (alpha2_value > TOL) & (alpha2_value < C_VALUE - TOL)
free2_indices = np.where(free2_mask)[0]
bounded2_indices = np.where(alpha2_value >= C_VALUE - TOL)[0]

# Recover w and b for the 2-D model
ay2_train  = alpha2_value * y_train
w2_scratch = ay2_train @ X2_train
b2_per_free_sv = (y_train[free2_indices]
                  - X2_train[free2_indices] @ w2_scratch)
b2_scratch = float(np.mean(b2_per_free_sv)) if free2_indices.size else 0.0

print(f"  2-D model  n_SV={sv2_indices.size}  w={w2_scratch.round(3)}  b={b2_scratch:+.3f}")


# ----------------------------------------------------------------------------
# Plot 07 — support vectors (2-D model)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X2_train[y_train == -1, 0], X2_train[y_train == -1, 1],
           s=12, c=COLOR_REPAY, alpha=0.5, label="repay")
ax.scatter(X2_train[y_train == +1, 0], X2_train[y_train == +1, 1],
           s=12, c=COLOR_DEFAULT, alpha=0.5, label="default")
ax.scatter(X2_train[free2_indices, 0], X2_train[free2_indices, 1],
           s=80, facecolors="none", edgecolors=COLOR_FREE_SV, linewidths=1.5,
           label=f"free SV ({free2_indices.size})")
ax.scatter(X2_train[bounded2_indices, 0], X2_train[bounded2_indices, 1],
           s=80, facecolors="none", edgecolors=COLOR_BOUNDED_SV, linewidths=1.5,
           label=f"bounded SV ({bounded2_indices.size})")
ax.set_xlabel("DTI (scaled)")
ax.set_ylabel("credit_util (scaled)")
ax.set_title("Step 7 — support vectors of the 2-D linear SVM\n"
             "free SVs lie ON the margin (αᵢ < C); bounded SVs are inside (αᵢ = C)")
ax.legend(loc="upper left")
save_plot("07_support_vectors.png")


# ----------------------------------------------------------------------------
# Plot 08 — decision boundary f = 0 with margin lines f = ±1
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

pad   = 0.5
x_lo  = X2_train[:, 0].min() - pad
x_hi  = X2_train[:, 0].max() + pad
y_lo  = X2_train[:, 1].min() - pad
y_hi  = X2_train[:, 1].max() + pad
xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 300),
                     np.linspace(y_lo, y_hi, 300))
Z = w2_scratch[0] * xx + w2_scratch[1] * yy + b2_scratch

# Fill predicted-default / predicted-repay regions
ax.contourf(xx, yy, Z, levels=[-1e3, 0, 1e3],
            colors=["#cfe2f3", "#f4cccc"], alpha=0.55)
# Contour lines at f = -1, 0, +1
cs = ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                colors="black", linestyles=["--", "-", "--"], linewidths=1.0)
ax.clabel(cs, fmt={-1: "f=-1", 0: "f=0", 1: "f=+1"}, fontsize=8)

# Training points + SVs
ax.scatter(X2_train[y_train == -1, 0], X2_train[y_train == -1, 1],
           s=10, c=COLOR_REPAY, alpha=0.55, label="repay")
ax.scatter(X2_train[y_train == +1, 0], X2_train[y_train == +1, 1],
           s=10, c=COLOR_DEFAULT, alpha=0.55, label="default")
ax.scatter(X2_train[sv2_indices, 0], X2_train[sv2_indices, 1],
           s=45, facecolors="none", edgecolors="black", linewidths=0.7,
           label=f"SVs ({sv2_indices.size})")

ax.set_xlabel("DTI (scaled)")
ax.set_ylabel("credit_util (scaled)")
ax.set_title("Step 8 — decision boundary f(x) = 0 with margin lines f = ±1\n"
             "(2-D linear SVM solved from scratch via the Lagrangian dual)")
ax.legend(loc="upper left")
save_plot("08_decision_boundary.png")


# ============================================================================
# 5. Validations (5-D model)
# ============================================================================
section("5. Sanity-check plots for the 5-D model")

# Primal vs dual decision function on the test set.
f_primal = X_test_scaled @ w_scratch + b_scratch
K_test_sv = X_test_scaled @ X_train_scaled[sv_indices].T
f_dual   = K_test_sv @ alpha_y_train[sv_indices] + b_scratch

# sklearn comparison
svc = SVC(kernel="linear", C=C_VALUE).fit(X_train_scaled, y_train)
f_sklearn = svc.decision_function(X_test_scaled)

print(f"  max |f_primal - f_dual|    = {np.abs(f_primal - f_dual).max():.2e}")
print(f"  max |f_scratch - f_sklearn| = {np.abs(f_dual - f_sklearn).max():.2e}")


# ----------------------------------------------------------------------------
# Plot 09 — primal vs dual decision function (should lie on y = x exactly)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(f_primal, f_dual, s=18, alpha=0.7, color="#4c72b0")
lo, hi = min(f_primal.min(), f_dual.min()), max(f_primal.max(), f_dual.max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
ax.set_xlabel("f_primal(x) = ⟨w, x⟩ + b")
ax.set_ylabel("f_dual(x) = Σᵢ αᵢ yᵢ ⟨xᵢ, x⟩ + b")
ax.set_title("Step 9 — primal vs dual decision function\n"
             f"max |Δ| = {np.abs(f_primal - f_dual).max():.1e}  →  identical to machine precision")
ax.legend(loc="upper left")
save_plot("09_primal_vs_dual.png")


# ----------------------------------------------------------------------------
# Plot 10 — from-scratch vs sklearn decision function on test points
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(f_dual, f_sklearn, s=18, alpha=0.7, color="#55a868")
lo, hi = min(f_dual.min(), f_sklearn.min()), max(f_dual.max(), f_sklearn.max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
ax.set_xlabel("f_scratch(x)  (cvxpy CLARABEL)")
ax.set_ylabel("f_sklearn(x)  (libsvm SMO)")
ax.set_title("Step 10 — same Lagrangian, two solvers\n"
             f"max |Δ| = {np.abs(f_dual - f_sklearn).max():.2e}  →  solver-tolerance noise")
ax.legend(loc="upper left")
save_plot("10_scratch_vs_sklearn.png")


# ----------------------------------------------------------------------------
# Plot 11 — ROC overlay + margin-score distribution by true class
# ----------------------------------------------------------------------------
fpr_s, tpr_s, _ = roc_curve(y_test, f_dual)
fpr_k, tpr_k, _ = roc_curve(y_test, f_sklearn)
auc_s = roc_auc_score(y_test, f_dual)
auc_k = roc_auc_score(y_test, f_sklearn)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC overlay
axes[0].plot(fpr_s, tpr_s, color="#55a868", lw=2,
             label=f"from-scratch  (AUC = {auc_s:.3f})")
axes[0].plot(fpr_k, tpr_k, color="#4c72b0", lw=2, ls="--",
             label=f"sklearn       (AUC = {auc_k:.3f})")
axes[0].plot([0, 1], [0, 1], "k:", lw=1, label="random")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curves — both solvers, both rankings")
axes[0].legend(loc="lower right")

# Margin-score distribution
axes[1].hist(f_dual[y_test == -1], bins=25, alpha=0.6,
             color=COLOR_REPAY, label="repay")
axes[1].hist(f_dual[y_test == +1], bins=25, alpha=0.6,
             color=COLOR_DEFAULT, label="default")
axes[1].axvline(0, color="k", lw=1, ls="--")
axes[1].set_xlabel("f(x) on test set  (signed margin)")
axes[1].set_ylabel("count")
axes[1].set_title("Margin score by true class — defaults sit right of f = 0")
axes[1].legend()

plt.suptitle("Step 11 — model evaluation", y=1.02)
save_plot("11_roc_and_margins.png")


# ============================================================================
# Done
# ============================================================================
print(f"\nAll plots written to ./{PLOT_DIR}/")
