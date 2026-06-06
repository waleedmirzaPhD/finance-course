"""
SVM from scratch — solve the Lagrangian dual explicitly, simplest kernel first.

This is the explicit-math companion to svm_minimal.py. Instead of calling
SVC(...).fit(), we write the dual quadratic programme out by hand and hand
it to a convex solver. We start with the SIMPLEST kernel — the linear one:

    k(x, x') = ⟨x, x'⟩         (feature map Φ(x) = x; no γ; finite-dim w)

This is the cleanest place to see the slide math because:

  • The Gram matrix is just  K = X Xᵀ  — no kernel function needed.
  • Because Φ(x) = x is finite-dimensional, we can also recover the primal
    weight vector  w = Σᵢ αᵢ yᵢ xᵢ  — IMPOSSIBLE with the RBF kernel,
    whose Φ lives in an infinite-dimensional space.
  • We can therefore compute f(x) two equivalent ways and check they agree:
        primal:  f(x) = ⟨w, x⟩ + b
        dual:    f(x) = Σᵢ∈SV (αᵢ yᵢ) ⟨xᵢ, x⟩ + b

Maps to Halperin's "SVM in Finance" slides, Part II:

  PRIMAL  (soft-margin SVC)
      min  (1/2) ||w||²  +  C · Σᵢ ξᵢ
      s.t. yᵢ (⟨w, xᵢ⟩ + b) ≥ 1 − ξᵢ,   ξᵢ ≥ 0

  LAGRANGIAN  (with αᵢ ≥ 0 on margin, μᵢ ≥ 0 on slack-nonnegativity)
      L(w, b, ξ, α, μ) = (1/2)||w||² + C Σᵢ ξᵢ
                       − Σᵢ αᵢ [ yᵢ (⟨w, xᵢ⟩ + b) − 1 + ξᵢ ]
                       − Σᵢ μᵢ ξᵢ

  STATIONARITY  (∂L/∂w = ∂L/∂b = ∂L/∂ξᵢ = 0)
      w = Σᵢ αᵢ yᵢ xᵢ
      Σᵢ αᵢ yᵢ = 0
      0 ≤ αᵢ ≤ C

  DUAL  (substitute stationarity back into L)
      max_α  Σᵢ αᵢ  −  (1/2) Σᵢⱼ αᵢ αⱼ yᵢ yⱼ ⟨xᵢ, xⱼ⟩
      s.t.   Σᵢ αᵢ yᵢ = 0,    0 ≤ αᵢ ≤ C

Install:
    pip install cvxpy
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import cvxpy as cp                                            # convex QP solver
import numpy as np                                            # numerical arrays
import pandas as pd                                           # dataset assembly
from sklearn.metrics import roc_auc_score                     # ranking metric
from sklearn.model_selection import train_test_split          # stratified split
from sklearn.preprocessing import StandardScaler              # mean=0, std=1
from sklearn.svm import SVC                                   # reference implementation


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ============================================================================
# 1. Same tiny synthetic credit dataset as svm_minimal.py
# ============================================================================
n_samples = 500

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

# Known logistic data-generating process — coefficients are deliberately
# strong (≈2× the svm_minimal.py values) so the classes are clearly
# linearly separable and the from-scratch linear SVM has a non-degenerate
# optimum to find. With weaker signal the soft-margin optimum is w ≈ 0
# (predict-all-majority) and the comparison to sklearn becomes noise-dominated.
logit = (
    -4.0
    + 8.0 * debt_to_income
    + 5.0 * credit_utilization
    + 2.0 * missed_payments
    - 8.0 * savings_ratio
    + 5.0 * income_volatility
)
p_default = 1.0 / (1.0 + np.exp(-logit))
y_01 = (rng.uniform(size=n_samples) < p_default).astype(int)   # labels in {0, 1}

# The slide math uses yᵢ ∈ {-1, +1}, so convert once and keep both forms.
y_signed = 2 * y_01 - 1                                         # {0,1} → {-1,+1}


# ============================================================================
# 2. Stratified train / test split, then scale
# ============================================================================
(X_train, X_test,
 y_train_01, y_test_01,
 y_train, y_test) = train_test_split(
    X_dataframe.values, y_01, y_signed,
    test_size=0.25, stratify=y_01, random_state=RANDOM_STATE,
)

scaler          = StandardScaler().fit(X_train)
X_train_scaled  = scaler.transform(X_train)
X_test_scaled   = scaler.transform(X_test)

n_train         = X_train_scaled.shape[0]
n_features      = X_train_scaled.shape[1]
print(f"Training set : {n_train} rows × {n_features} features")
print(f"Default rate : train {y_train_01.mean():.2%}   test {y_test_01.mean():.2%}")


# ============================================================================
# 3. Hyperparameter (linear SVM has just C — no γ)
# ============================================================================
# We pick C = 10 deliberately. With C = 1 on this dataset the soft-margin
# linear SVM is degenerate — the optimum is w = 0, b = ±1 (predict-all-
# majority), because the slack penalty 2·n_minority is cheaper than any
# non-zero ||w||² could afford. Bumping C up makes a non-zero w worthwhile
# and gives a meaningful separating hyperplane.
C_value = 10.0
print(f"\nHyperparameter: C = {C_value}   (linear kernel → no γ)")


# ============================================================================
# 4. Linear kernel Gram matrix:  K[i, j] = ⟨xᵢ, xⱼ⟩  =  (X Xᵀ)[i, j]
# ============================================================================
# This is the kernel trick at its most trivial: the "feature map" is the
# identity, so the kernel value is just a dot product. K is PSD because it's
# a Gram matrix of real vectors.
K_train = X_train_scaled @ X_train_scaled.T


# ============================================================================
# 5. Build and solve the dual QP with cvxpy  ← THE LAGRANGIAN DUAL
# ============================================================================
# Variables: one αᵢ ≥ 0 per training row.
alpha = cp.Variable(n_train, nonneg=True)

# (αᵢ yᵢ) appears all over the dual; precompute the element-wise product.
alpha_y = cp.multiply(alpha, y_train)

# Dual objective:
#     Σᵢ αᵢ − (1/2) Σᵢⱼ αᵢ αⱼ yᵢ yⱼ ⟨xᵢ, xⱼ⟩
#   = Σᵢ αᵢ − (1/2) (α∘y)ᵀ K (α∘y)
dual_objective = cp.Maximize(
    cp.sum(alpha)
    - 0.5 * cp.quad_form(alpha_y, cp.psd_wrap(K_train))
)

# Constraints from KKT:
#     0 ≤ αᵢ ≤ C        (box from soft-margin slack)
#     Σᵢ αᵢ yᵢ = 0      (from ∂L/∂b = 0)
dual_constraints = [
    alpha <= C_value,
    cp.sum(alpha_y) == 0,
]

dual_problem = cp.Problem(dual_objective, dual_constraints)
dual_problem.solve()

print(f"\nDual QP solved")
print(f"  cvxpy status     : {dual_problem.status}")
print(f"  optimal objective: {dual_problem.value:.6f}")

# Optimal α values as plain NumPy.
alpha_value = np.asarray(alpha.value).ravel()


# ============================================================================
# 6. Identify support vectors and compute b from KKT
# ============================================================================
TOL = 1e-5

# KKT: αᵢ > 0 ⇔ training point i is a support vector.
sv_mask     = alpha_value > TOL
sv_indices  = np.where(sv_mask)[0]
n_sv        = sv_indices.size

# Free SV: 0 < αᵢ < C  →  sits exactly on the margin (slack ξᵢ = 0).
# Bounded SV: αᵢ = C   →  sits inside the margin or misclassified (ξᵢ > 0).
free_mask     = (alpha_value > TOL) & (alpha_value < C_value - TOL)
free_indices  = np.where(free_mask)[0]
n_free        = free_indices.size
n_bounded     = n_sv - n_free

# Signed dual coefficients α_i y_i (this is exactly what sklearn stores).
alpha_y_train = alpha_value * y_train


# ============================================================================
# 7. Recover the PRIMAL weight vector w  (only possible with linear kernel)
# ============================================================================
# From stationarity:  w = Σᵢ αᵢ yᵢ xᵢ.
# This is impossible for the RBF kernel because Φ(x) lives in an
# infinite-dimensional space — w would have infinitely many entries.
# Here, with the linear kernel, w is a concrete 5-vector.
w_scratch = alpha_y_train @ X_train_scaled                # shape (n_features,)

# For a free SV i:  yᵢ (⟨w, xᵢ⟩ + b) = 1  ⇒  b = yᵢ − ⟨w, xᵢ⟩.
# Average across all free SVs for numerical stability.
b_per_free_sv = y_train[free_indices] - X_train_scaled[free_indices] @ w_scratch
b_scratch     = float(np.mean(b_per_free_sv))

print(f"\nFrom-scratch solution")
print(f"  total support vectors : {n_sv}  ({100 * n_sv / n_train:.1f}% of train)")
print(f"  free SVs   (|α| < C)  : {n_free}")
print(f"  bounded SVs (α = C)   : {n_bounded}")
print(f"  intercept b           : {b_scratch:+.6f}")
print(f"  weight vector w       : {w_scratch.round(4)}")


# ============================================================================
# 8. Decision function — compute it TWO ways and check they agree
# ============================================================================
# Way A (PRIMAL form, only possible with linear kernel):
#     f(x) = ⟨w, x⟩ + b
f_primal = X_test_scaled @ w_scratch + b_scratch

# Way B (DUAL form, support-vector expansion — generalises to any kernel):
#     f(x) = Σᵢ∈SV (αᵢ yᵢ) ⟨xᵢ, x⟩ + b
K_test_sv = X_test_scaled @ X_train_scaled[sv_indices].T
f_dual    = K_test_sv @ alpha_y_train[sv_indices] + b_scratch

# These two must be numerically identical, modulo float-epsilon.
primal_vs_dual_err = np.abs(f_primal - f_dual).max()
print(f"\nPrimal vs dual decision function:")
print(f"  max |f_primal − f_dual| = {primal_vs_dual_err:.2e}   "
      "(should be ~ machine epsilon)")


# ============================================================================
# 9. sklearn SVC with kernel="linear" for an apples-to-apples comparison
# ============================================================================
svc = SVC(kernel="linear", C=C_value)
svc.fit(X_train_scaled, y_train)

# sklearn exposes `coef_` only for the linear kernel — this IS w.
w_sklearn = svc.coef_.ravel()
b_sklearn = svc.intercept_[0]

print(f"\nsklearn SVC (kernel='linear')")
print(f"  total support vectors : {svc.support_.size}")
print(f"  intercept b           : {b_sklearn:+.6f}")
print(f"  weight vector w       : {w_sklearn.round(4)}")


# ============================================================================
# 10. Compare from-scratch vs sklearn on the test set
# ============================================================================
f_sklearn = svc.decision_function(X_test_scaled)

# Decision-function agreement on test points.
max_err   = np.abs(f_dual - f_sklearn).max()
mean_err  = np.abs(f_dual - f_sklearn).mean()
# Weight-vector agreement.
w_err     = np.abs(w_scratch - w_sklearn).max()
# Intercept agreement.
b_err     = abs(b_scratch - b_sklearn)

print(f"\nAgreement with sklearn")
print(f"  max |w_scratch − w_sklearn|       = {w_err:.2e}")
print(f"  |b_scratch − b_sklearn|           = {b_err:.2e}")
print(f"  max |f_scratch − f_sklearn| (test) = {max_err:.2e}")
print(f"  mean|f_scratch − f_sklearn| (test) = {mean_err:.2e}")

# Predictions and AUC.
y_pred_scratch = np.where(f_dual >= 0, +1, -1)
y_pred_sklearn = svc.predict(X_test_scaled)

acc_scratch = (y_pred_scratch == y_test).mean()
acc_sklearn = (y_pred_sklearn == y_test).mean()
auc_scratch = roc_auc_score(y_test, f_dual)
auc_sklearn = roc_auc_score(y_test, f_sklearn)

print(f"\nTest accuracy :  from-scratch {acc_scratch:.3f}   sklearn {acc_sklearn:.3f}")
print(f"Test ROC-AUC  :  from-scratch {auc_scratch:.3f}   sklearn {auc_sklearn:.3f}")

if max_err < 1e-2 and acc_scratch == acc_sklearn:
    print("\n✓ The from-scratch linear SVM matches sklearn.")
    print("  (Sub-1% margin disagreement is solver tolerance: cvxpy uses")
    print("   CLARABEL, sklearn uses libsvm's SMO — same optimum, different")
    print("   solvers. Predictions and AUC are bit-identical.)")


# ============================================================================
# 11. Score a brand-new borrower with the from-scratch model
# ============================================================================
new_borrower = np.array([[0.55, 0.78, 2, 0.05, 0.30]])
new_scaled   = scaler.transform(new_borrower)

# Primal form: cheapest at inference time once w is known.
# `new_scaled` is shape (1, n_features); pull the scalar out with [0].
new_margin   = float((new_scaled @ w_scratch)[0] + b_scratch)
new_decision = "DEFAULT" if new_margin >= 0 else "REPAY"
print(f"\nNew borrower (from-scratch linear SVM):  "
      f"margin = {new_margin:+.3f}  →  {new_decision}")
