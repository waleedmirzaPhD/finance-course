"""
Minimalistic SVM for loan default prediction — every slide concept, expanded.

Maps to Halperin's "SVM in Finance" slides:
  Part I   — soft-margin SVC with hinge-loss + C trade-off
  Part II  — convex dual QP, support-vector expansion
             f(x) = Σᵢ αᵢ yᵢ k(xᵢ, x) + b
  Part III — kernel trick ⟨Φ(x), Φ(x')⟩ = k(x, x'), Gaussian RBF kernel

Each step below is split into one named intermediate per line so the data
flow stays readable end-to-end.
"""

# ----------------------------------------------------------------------------
# Imports — each on its own line with a one-line note on its role.
# ----------------------------------------------------------------------------
import numpy as np                                            # numerical arrays
import pandas as pd                                           # tabular display
from sklearn.metrics import confusion_matrix                  # 2x2 actual vs predicted
from sklearn.metrics import roc_auc_score                     # ranking-based eval metric
from sklearn.model_selection import GridSearchCV              # exhaustive hyperparam search
from sklearn.model_selection import StratifiedKFold           # class-balanced CV folds
from sklearn.model_selection import train_test_split          # single-shot split
from sklearn.pipeline import Pipeline                         # chain scaler + estimator
from sklearn.preprocessing import StandardScaler              # mean=0, std=1 per column
from sklearn.svm import SVC                                   # soft-margin SVC


# ----------------------------------------------------------------------------
# Reproducibility — fix every random seed via a single constant.
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ============================================================================
# 1. Build a tiny synthetic borrower book (500 rows, 5 risk factors)
# ============================================================================
# Choose the sample size up front so it's easy to scale.
n_samples = 500

# Draw each risk factor on its own line so the distribution is visible.
debt_to_income     = rng.beta(2, 5, n_samples) * 1.2
credit_utilization = rng.beta(2, 4, n_samples)
missed_payments    = rng.poisson(0.6, n_samples)
savings_ratio      = rng.beta(2, 5, n_samples) * 0.5
income_volatility  = np.abs(rng.normal(0.15, 0.1, n_samples))

# Assemble the risk factors into a DataFrame for clean labelling later.
X_dataframe = pd.DataFrame({
    "debt_to_income":     debt_to_income,
    "credit_utilization": credit_utilization,
    "missed_payments":    missed_payments,
    "savings_ratio":      savings_ratio,
    "income_volatility":  income_volatility,
})

# Build the log-odds of default term by term so each coefficient is visible.
intercept_term            = -3.0
contribution_dti          =  3.0 * debt_to_income
contribution_credit_util  =  2.0 * credit_utilization
contribution_missed_pmt   =  0.8 * missed_payments
contribution_savings      = -3.0 * savings_ratio
contribution_income_vol   =  2.0 * income_volatility

# Sum all contributions — this is the log-odds (logit) of default per borrower.
logit = (
    intercept_term
    + contribution_dti
    + contribution_credit_util
    + contribution_missed_pmt
    + contribution_savings
    + contribution_income_vol
)

# Map log-odds → probability of default via the logistic CDF.
exp_neg_logit = np.exp(-logit)
denom         = 1.0 + exp_neg_logit
p_default     = 1.0 / denom

# Bernoulli draw: borrower defaults iff a uniform draw falls under p_default.
uniform_draws = rng.uniform(size=n_samples)
y_bool        = uniform_draws < p_default       # NumPy bool array
y             = y_bool.astype(int)              # final 1-D array of {0, 1} labels


# ============================================================================
# 2. Stratified train / test split
# ============================================================================
# Convert the DataFrame to a plain NumPy array (sklearn-friendly).
X_array = X_dataframe.values

# Stratify on y so train and test halves keep the same default rate.
X_train, X_test, y_train, y_test = train_test_split(
    X_array,
    y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE,
)


# ============================================================================
# 3. Pipeline + (C, γ) grid search
# ============================================================================
# Why scale: the RBF kernel exp(−γ ||x − x'||²) is distance-based; without
# scaling, large-range features dominate the kernel value.
# Why convex (Part II): the SVM dual is a QP with linear constraints, so
# the solver finds a unique global minimum (unlike neural networks).

# Build each pipeline step as its own object — easier to inspect later.
scaler  = StandardScaler()
svm_clf = SVC(
    kernel="rbf",
    class_weight="balanced",
)
pipe = Pipeline([
    ("scaler", scaler),
    ("svm",    svm_clf),
])

# Hyperparameter grid: C controls margin softness (Part I); γ controls
# the RBF kernel bandwidth (Part III). Three values each = 9 combos.
C_values     = [0.1, 1.0, 10.0]
gamma_values = [0.01, 0.1, 1.0]
param_grid = {
    "svm__C":     C_values,
    "svm__gamma": gamma_values,
}

# 5-fold stratified cross-validation → 9 × 5 = 45 SVM fits in parallel.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

# Pull the refit best estimator and its hyperparameters out of the grid.
model       = grid.best_estimator_
best_params = grid.best_params_
best_cv_auc = grid.best_score_

print(f"Best params : {best_params}")
print(f"CV AUC      : {best_cv_auc:.3f}")


# ============================================================================
# 4. Evaluation on the held-out test set
# ============================================================================
# Raw signed margin score f(x) — this is what the SVM actually optimises.
test_scores = model.decision_function(X_test)

# Hard {0, 1} predictions at the f(x) = 0 threshold.
test_predict = model.predict(X_test)

# Rank-based AUC operates directly on the continuous margin score.
test_auc = roc_auc_score(y_test, test_scores)

# 2 × 2 confusion matrix (rows = actual, cols = predicted).
test_confusion = confusion_matrix(y_test, test_predict)

print(f"Test AUC    : {test_auc:.3f}")
print("Confusion   : (rows = actual, cols = predicted)")
print(test_confusion)


# ============================================================================
# 5. Support-vector count (Part II: complexity ∝ #SVs, not feature dim)
# ============================================================================
# Pull the SVC out of the pipeline so we can read its dual attributes.
svm_step = model.named_steps["svm"]

# `support_` lists the training-row indices that became support vectors.
sv_indices = svm_step.support_
n_sv       = sv_indices.size

# Total number of training rows, for the SV-fraction calculation.
n_train     = len(X_train)
sv_fraction = 100.0 * n_sv / n_train

print(f"Support vectors: {n_sv} of {n_train} training rows ({sv_fraction:.0f}%)")


# ============================================================================
# 6. Kernel trick (Part III)
# ============================================================================
# Slide claim: for the quadratic feature map
#     Φ(x) = (x₁², √2 x₁ x₂, x₂²)
# the feature-space inner product equals the squared raw dot product:
#     ⟨Φ(a), Φ(b)⟩ = ⟨a, b⟩².

def phi(z):
    """Explicit quadratic feature map from the slides: R² → R³."""
    # Component 1: squared first coordinate.
    component_1 = z[0] ** 2
    # Component 2: cross term weighted by √2 (this is what makes the identity work).
    component_2 = np.sqrt(2.0) * z[0] * z[1]
    # Component 3: squared second coordinate.
    component_3 = z[1] ** 2
    return np.array([component_1, component_2, component_3])


# Two arbitrary 2-D points to compare.
point_a = rng.standard_normal(2)
point_b = rng.standard_normal(2)

# LHS — explicit feature map followed by a 3-D dot product.
phi_a = phi(point_a)
phi_b = phi(point_b)
lhs_dot_product = np.dot(phi_a, phi_b)

# RHS — square of the raw 2-D dot product (no feature map needed).
raw_dot_product = np.dot(point_a, point_b)
rhs_squared     = raw_dot_product ** 2

print(f"\nKernel trick: ⟨Φ(a), Φ(b)⟩ = {lhs_dot_product:+.4f}")
print(f"             ⟨a, b⟩²       = {rhs_squared:+.4f}   (equal → trick holds)")


# ============================================================================
# 7. Support-vector expansion (Part II)
# ============================================================================
# Slide formula:    f(x) = Σᵢ (αᵢ yᵢ) k(xᵢ, x) + b
# sklearn storage:
#     αᵢ yᵢ                  →  svm.dual_coef_[0]
#     xᵢ (in scaled space)   →  svm.support_vectors_
#     b                      →  svm.intercept_[0]
#     γ (numeric value)      →  svm._gamma

# Pull the scaler and the SVM out of the pipeline.
scaler_step = model.named_steps["scaler"]
svm_step    = model.named_steps["svm"]

# Take only the first 3 test rows for a compact comparison.
n_to_verify     = 3
X_test_subset   = X_test[:n_to_verify]

# Bring the test rows into the same scaled space the SVs live in.
X_test_scaled = scaler_step.transform(X_test_subset)

# The trained dual primitives.
support_vectors = svm_step.support_vectors_      # shape (n_SV, d)
dual_coef       = svm_step.dual_coef_[0]         # shape (n_SV,)  values = αᵢ yᵢ
intercept       = svm_step.intercept_[0]         # scalar bias b
gamma_value     = svm_step._gamma                # numeric γ resolved from "scale"

# Build pairwise differences SV_i - x_j by NumPy broadcasting.
# Reshape SVs to (1, n_SV, d) and tests to (n_test, 1, d).
support_vectors_3d = support_vectors[None, :, :]
X_test_scaled_3d   = X_test_scaled[:, None, :]
pairwise_diff      = support_vectors_3d - X_test_scaled_3d        # (n_test, n_SV, d)

# Square element-wise, then sum along the feature axis = ||·||².
pairwise_squared = pairwise_diff ** 2
pairwise_sqdist  = np.sum(pairwise_squared, axis=-1)              # (n_test, n_SV)

# Apply the Gaussian RBF kernel element-wise.
neg_gamma_sqdist = -gamma_value * pairwise_sqdist
kernel_matrix    = np.exp(neg_gamma_sqdist)                       # (n_test, n_SV)

# Weighted sum of kernel values → the support-vector expansion.
weighted_sum = kernel_matrix @ dual_coef
f_manual     = weighted_sum + intercept

# Compare against sklearn's own decision_function on the same rows.
f_sklearn = model.decision_function(X_test_subset)

print(f"\nManual SV expansion : {f_manual.round(6)}")
print(f"sklearn decision_fn : {f_sklearn.round(6)}   (identical → expansion verified)")


# ============================================================================
# 8. Score a brand-new borrower (inference example)
# ============================================================================
# A single applicant profile; columns are in the same order as X_dataframe.
new_borrower_features = np.array([[
    0.55,    # debt_to_income      — 55% of income goes to debt service
    0.78,    # credit_utilization  — 78% of revolving limit used
    2.0,     # missed_payments     — 2 missed payments in last 12 months
    0.05,    # savings_ratio       — only 5% of income saved
    0.30,    # income_volatility   — high CV of monthly income
]])

# Get the raw signed margin score under the tuned model.
new_margin = model.decision_function(new_borrower_features)[0]

# Decision rule: positive margin → predicted default.
if new_margin >= 0:
    new_decision = "DEFAULT"
else:
    new_decision = "REPAY"

print(f"\nNew borrower margin = {new_margin:+.3f}  →  {new_decision}")
