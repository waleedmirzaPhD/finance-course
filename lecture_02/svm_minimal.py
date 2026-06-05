"""
Minimalistic SVM for loan default prediction — every slide concept in ~80 lines.

Maps to Halperin's "SVM in Finance" slides:
  Part I   — soft-margin SVC with hinge-loss + C trade-off (every fit below)
  Part II  — convex dual QP, support-vector expansion f(x) = Σᵢ αᵢyᵢ k(xᵢ,x) + b
  Part III — kernel trick ⟨Φ(x),Φ(x')⟩ = k(x,x'),  Gaussian RBF kernel
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

rng = np.random.default_rng(42)

# 1. Tiny synthetic borrower book (500 rows, 5 risk factors) ------------------
n = 500
X = pd.DataFrame({
    "debt_to_income":     rng.beta(2, 5, n) * 1.2,
    "credit_utilization": rng.beta(2, 4, n),
    "missed_payments":    rng.poisson(0.6, n),
    "savings_ratio":      rng.beta(2, 5, n) * 0.5,
    "income_volatility":  np.abs(rng.normal(0.15, 0.1, n)),
})
# Known logistic data-generating process → ~25% default rate.
logit = (-3.0 + 3.0 * X["debt_to_income"] + 2.0 * X["credit_utilization"]
         + 0.8 * X["missed_payments"] - 3.0 * X["savings_ratio"]
         + 2.0 * X["income_volatility"])
y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int).values

# 2. Stratified train / test split --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.25, stratify=y, random_state=42,
)

# 3. Pipeline + (C, γ) grid search (Part I: C, Part III: γ) -------------------
# StandardScaler is mandatory — the RBF kernel exp(-γ ||x-x'||²) is
# distance-based, so unscaled features with different magnitudes would
# dominate the kernel. The whole training problem is a convex QP (Part II)
# so the solution is the unique global minimum.
pipe = Pipeline([("scaler", StandardScaler()),
                 ("svm",    SVC(kernel="rbf", class_weight="balanced"))])
grid = GridSearchCV(
    pipe,
    param_grid={"svm__C": [0.1, 1, 10], "svm__gamma": [0.01, 0.1, 1]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc", n_jobs=-1,
).fit(X_train, y_train)
model = grid.best_estimator_
print(f"Best params : {grid.best_params_}   CV AUC = {grid.best_score_:.3f}")

# 4. Evaluation on the held-out test set --------------------------------------
score = model.decision_function(X_test)                 # signed margin f(x)
pred  = model.predict(X_test)
print(f"Test AUC    : {roc_auc_score(y_test, score):.3f}")
print(f"Confusion   : (rows = actual, cols = predicted)\n{confusion_matrix(y_test, pred)}")

# 5. Number of support vectors (Part II: model complexity ∝ #SVs, not d) ------
svm  = model.named_steps["svm"]
n_sv = svm.support_.size
print(f"Support vectors: {n_sv}/{len(X_train)} ({100 * n_sv / len(X_train):.0f}% of train)")

# 6. Kernel trick (Part III): ⟨Φ(a), Φ(b)⟩ = ⟨a, b⟩²  for Φ(x)=(x₁², √2 x₁x₂, x₂²)
a, b = rng.standard_normal(2), rng.standard_normal(2)
phi  = lambda z: np.array([z[0] ** 2, np.sqrt(2) * z[0] * z[1], z[1] ** 2])
print(f"\nKernel trick: ⟨Φ(a),Φ(b)⟩ = {phi(a) @ phi(b):+.4f}   "
      f"⟨a,b⟩² = {(a @ b) ** 2:+.4f}   (equal → trick holds)")

# 7. Support-vector expansion (Part II): f(x) = Σᵢ (αᵢyᵢ) k(xᵢ,x) + b ---------
# sklearn stores α_i y_i in `dual_coef_[0]` and the scaled SV inputs x_i in
# `support_vectors_`. We rebuild f(x) by hand and confirm it matches.
Xs   = model.named_steps["scaler"].transform(X_test[:3])
K    = np.exp(-svm._gamma * np.sum(
    (svm.support_vectors_[None, :, :] - Xs[:, None, :]) ** 2, axis=-1))
f_manual  = K @ svm.dual_coef_[0] + svm.intercept_[0]
f_sklearn = model.decision_function(X_test[:3])
print(f"\nManual SV expansion : {f_manual.round(6)}")
print(f"sklearn decision_fn : {f_sklearn.round(6)}   (identical → expansion verified)")

# 8. Score a brand-new borrower -----------------------------------------------
new = np.array([[0.55, 0.78, 2, 0.05, 0.30]])           # DTI, CU, MP, SR, IV
margin = model.decision_function(new)[0]
print(f"\nNew borrower margin = {margin:+.3f}  "
      f"→ {'DEFAULT' if margin >= 0 else 'REPAY'}")
