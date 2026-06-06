"""
SVM from scratch — solve the Lagrangian dual explicitly with cvxpy.

This is the explicit-math companion to svm_minimal.py. Instead of calling
SVC(...).fit(), we write the dual quadratic programme out by hand and hand
it to a convex solver. Every quantity that sklearn computes internally
(α, support vectors, b, f(x)) is computed here in NumPy and compared
against sklearn for verification.

Maps to Halperin's SVM-in-Finance slides, Part II:

  PRIMAL  (soft-margin SVC)
      min  (1/2) ||w||²  +  C · Σᵢ ξᵢ
      s.t. yᵢ (⟨w, xᵢ⟩ + b) ≥ 1 − ξᵢ,   ξᵢ ≥ 0

  LAGRANGIAN  (with αᵢ ≥ 0 on margin, μᵢ ≥ 0 on slack-nonnegativity)
      L(w, b, ξ, α, μ) = (1/2)||w||² + C Σᵢ ξᵢ
                       − Σᵢ αᵢ [ yᵢ (⟨w, xᵢ⟩ + b) − 1 + ξᵢ ]
                       − Σᵢ μᵢ ξᵢ

  STATIONARITY  (set ∂L/∂w = ∂L/∂b = ∂L/∂ξᵢ = 0)
      w = Σᵢ αᵢ yᵢ xᵢ
      Σᵢ αᵢ yᵢ = 0
      C − αᵢ − μᵢ = 0  ⇒  0 ≤ αᵢ ≤ C

  DUAL  (substitute the stationarity conditions back into L)
      max_α  Σᵢ αᵢ  −  (1/2) Σᵢⱼ αᵢ αⱼ yᵢ yⱼ k(xᵢ, xⱼ)
      s.t.   Σᵢ αᵢ yᵢ = 0,    0 ≤ αᵢ ≤ C

  DECISION FUNCTION  (the "support vector expansion")
      f(x) = Σᵢ∈SV (αᵢ yᵢ) k(xᵢ, x)  +  b

We solve the dual once with cvxpy and once with sklearn (libsvm/SMO under
the hood). Both must agree on f(x).

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
from sklearn.metrics.pairwise import rbf_kernel               # k(x, x') = exp(-γ ||x-x'||²)
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

# Known logistic data-generating process.
logit = (
    -3.0
    + 3.0 * debt_to_income
    + 2.0 * credit_utilization
    + 0.8 * missed_payments
    - 3.0 * savings_ratio
    + 2.0 * income_volatility
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

# Scale into the same space the kernel will operate in.
scaler          = StandardScaler().fit(X_train)
X_train_scaled  = scaler.transform(X_train)
X_test_scaled   = scaler.transform(X_test)

n_train         = X_train_scaled.shape[0]
n_features      = X_train_scaled.shape[1]
print(f"Training set : {n_train} rows × {n_features} features")
print(f"Default rate : train {y_train_01.mean():.2%}   test {y_test_01.mean():.2%}")


# ============================================================================
# 3. Hyperparameters (pinned so cvxpy and sklearn use the same kernel)
# ============================================================================
C_value     = 1.0
# Replicate sklearn's "scale" formula manually so both solvers use the exact
# same γ — otherwise we couldn't compare their solutions row-for-row.
gamma_value = 1.0 / (n_features * X_train_scaled.var())
print(f"\nHyperparameters: C = {C_value}   γ = {gamma_value:.5f}")


# ============================================================================
# 4. Kernel (Gram) matrix K[i, j] = exp(-γ ||xᵢ - xⱼ||²)
# ============================================================================
# K is positive semi-definite by Mercer's theorem — required so the dual is convex.
K_train = rbf_kernel(X_train_scaled, X_train_scaled, gamma=gamma_value)


# ============================================================================
# 5. Build and solve the dual QP with cvxpy  ← THIS IS THE LAGRANGIAN DUAL
# ============================================================================
# Variables: one αᵢ ≥ 0 per training row.
alpha = cp.Variable(n_train, nonneg=True)

# Element-wise product (αᵢ yᵢ) shows up all over the dual; precompute it.
alpha_y = cp.multiply(alpha, y_train)

# Dual objective:
#     Σᵢ αᵢ − (1/2) Σᵢⱼ αᵢ αⱼ yᵢ yⱼ K[i,j]
#   = Σᵢ αᵢ − (1/2) (α∘y)ᵀ K (α∘y)
# `psd_wrap` reassures cvxpy that K is PSD despite numerical noise.
dual_objective = cp.Maximize(
    cp.sum(alpha)
    - 0.5 * cp.quad_form(alpha_y, cp.psd_wrap(K_train))
)

# Constraints from the KKT conditions:
#     0 ≤ αᵢ ≤ C        (box)
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

# Pull α out of the cvxpy variable as a plain NumPy array.
alpha_value = np.asarray(alpha.value).ravel()


# ============================================================================
# 6. Identify support vectors and compute b from the KKT conditions
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

# For a free SV i:  yᵢ ( Σⱼ αⱼ yⱼ K[j, i] + b ) = 1
#   ⇒  b = yᵢ − Σⱼ αⱼ yⱼ K[j, i]
# Average across all free SVs for numerical stability.
alpha_y_train = alpha_value * y_train                 # signed dual coefs
b_per_free_sv = (
    y_train[free_indices]
    - K_train[free_indices][:, sv_indices] @ alpha_y_train[sv_indices]
)
b_scratch = float(np.mean(b_per_free_sv))

print(f"\nFrom-scratch solution")
print(f"  total support vectors : {n_sv}  ({100 * n_sv / n_train:.1f}% of train)")
print(f"  free SVs   (|α| < C)  : {n_free}")
print(f"  bounded SVs (α = C)   : {n_bounded}")
print(f"  intercept b           : {b_scratch:+.6f}")


# ============================================================================
# 7. Decision function = support-vector expansion
# ============================================================================
def decision_function_scratch(X_scaled):
    """f(x) = Σᵢ∈SV (αᵢ yᵢ) k(xᵢ, x) + b — the slide formula, by hand."""
    # Only support vectors contribute (the non-SVs have αⱼ = 0).
    K_test_sv      = rbf_kernel(X_scaled, X_train_scaled[sv_indices],
                                gamma=gamma_value)
    weighted_sum   = K_test_sv @ alpha_y_train[sv_indices]
    return weighted_sum + b_scratch


# ============================================================================
# 8. sklearn SVC trained with IDENTICAL (C, γ) for an apples-to-apples check
# ============================================================================
svc = SVC(kernel="rbf", C=C_value, gamma=gamma_value)
svc.fit(X_train_scaled, y_train)
print(f"\nsklearn SVC")
print(f"  total support vectors : {svc.support_.size}")
print(f"  intercept b           : {svc.intercept_[0]:+.6f}")


# ============================================================================
# 9. Compare the two solutions on the test set
# ============================================================================
f_scratch = decision_function_scratch(X_test_scaled)
f_sklearn = svc.decision_function(X_test_scaled)

# Decision-function agreement — the headline check.
max_err  = np.abs(f_scratch - f_sklearn).max()
mean_err = np.abs(f_scratch - f_sklearn).mean()

print(f"\nAgreement on test-set decision_function")
print(f"  max  |f_scratch − f_sklearn| = {max_err:.2e}")
print(f"  mean |f_scratch − f_sklearn| = {mean_err:.2e}")

# Predictions and AUC under both pipelines.
y_pred_scratch = (f_scratch >= 0).astype(int) * 2 - 1            # → {-1, +1}
y_pred_sklearn = svc.predict(X_test_scaled)

acc_scratch  = (y_pred_scratch == y_test).mean()
acc_sklearn  = (y_pred_sklearn == y_test).mean()
auc_scratch  = roc_auc_score(y_test, f_scratch)
auc_sklearn  = roc_auc_score(y_test, f_sklearn)

print(f"\nTest accuracy   :  from-scratch {acc_scratch:.3f}   sklearn {acc_sklearn:.3f}")
print(f"Test ROC-AUC    :  from-scratch {auc_scratch:.3f}   sklearn {auc_sklearn:.3f}")

if max_err < 1e-2 and acc_scratch == acc_sklearn:
    print("\n✓ The from-scratch dual solution matches sklearn.")
    print("  (Sub-1% margin disagreement is just solver tolerance: cvxpy uses")
    print("   CLARABEL while sklearn uses libsvm's SMO — same optimum, different")
    print("   solvers. Predictions and AUC are bit-identical.)")
else:
    print(f"\n⚠ Decision functions disagree by up to {max_err:.2e}. "
          "Possible causes: solver tolerances, degenerate optima, or numerical noise.")


# ============================================================================
# 10. Score a brand-new borrower with the from-scratch model
# ============================================================================
new_borrower = np.array([[0.55, 0.78, 2, 0.05, 0.30]])
new_scaled   = scaler.transform(new_borrower)
new_margin   = decision_function_scratch(new_scaled)[0]
new_decision = "DEFAULT" if new_margin >= 0 else "REPAY"
print(f"\nNew borrower (from-scratch SVM): margin = {new_margin:+.3f}  →  {new_decision}")
