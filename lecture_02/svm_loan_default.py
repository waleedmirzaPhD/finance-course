"""
Support Vector Machines for Finance — a deep dive aligned with Halperin's
"Supervised, Unsupervised and Reinforcement Learning in Finance" lecture
(NYU Tandon, 2017), Week 1 SVM slides Parts I, II and III.

The script demonstrates every concept from those slides on a realistic
retail-credit dataset and adds the practical scaffolding you would build
around an SVM in production:

  PART I   (linear SVR / classification, flatness, slack, ε-insensitive loss)
    - Build a synthetic loan dataset and a continuous-spread dataset.
    - Train SVMs in a Pipeline(scaler, SVC) — the scaler is essential
      because the kernel is distance-based.

  PART II  (KKT, the dual, support-vector expansion, convex optimization)
    - Inspect dual_coef_, support_vectors_, free vs bounded SVs.
    - Reconstruct f(x) = Σ_i (α_i y_i) k(x_i, x) + b by hand and check it
      matches sklearn.decision_function — the slide's "SV expansion".
    - Train with multiple random seeds → identical (α, b): the dual QP is
      convex and has a unique global minimum (unlike neural networks).

  PART III (the kernel trick: features Φ(x) replaced by a kernel k(x, x'))
    - Compare linear / polynomial / sigmoid / Gaussian-RBF kernels.
    - Numerically verify <Φ(x), Φ(x')> = <x, x'>² for the slide's
      quadratic feature map Φ(x) = (x_1², √2 x_1 x_2, x_2²).
    - Sweep (C, γ) on a CV-AUC heatmap.

  APPLIED LAYER (what a credit-risk analyst actually does next)
    - Calibrate SVM scores → probabilities (Platt vs isotonic).
    - Permutation feature importance (RBF SVMs have no `coef_`).
    - Threshold tuning for asymmetric default cost.
    - Learning curve.
    - 2-D decision-boundary plot on two intuitive features.
    - Score a brand-new borrower.

  SVR DEMO  (slides cover both SVC and SVR; we show SVR too)
    - Predict a continuous "fair credit spread" with ε-SVR and ν-SVR.
    - Illustrate the ε-tube and the sparse SV expansion in regression form.

Run:
    python svm_loan_default.py

All plots are written to ./svm_plots/. Total runtime ~1–2 minutes.
"""

# ============================================================================
# MATHEMATICAL RECAP — direct map to Halperin's SVM-in-Finance slides
# ============================================================================
#
# ── Part I: Primal SVR  (slides "Sketch of SVM math") ────────────────────────
#
#   Predict  f(x) = <w, x> + b  with deviation at most ε from y_i.
#   "Flatness" means a small w, leading to the quadratic programme
#       min  (1/2) ||w||²
#       s.t.  y_i − <w, x_i> − b ≤ ε
#             <w, x_i> + b − y_i ≤ ε
#
#   Real data needs slack. Introduce ξ_i, ξ_i* ≥ 0 (one per tube side):
#       min  (1/2) ||w||²  +  C · Σ_i (ξ_i + ξ_i*)
#       s.t.  y_i − <w, x_i> − b ≤ ε + ξ_i
#             <w, x_i> + b − y_i ≤ ε + ξ_i*
#             ξ_i, ξ_i* ≥ 0
#
#   C controls the trade-off between flatness and tolerance for deviations
#   larger than ε. The ε-insensitive loss = max(0, |y − f(x)| − ε):
#     · zero penalty inside the ε-tube
#     · linear penalty outside.
#
#   For binary classification (y_i ∈ {-1,+1}) the same machinery becomes
#   the familiar hinge-loss soft-margin SVM (this script uses it):
#       min  (1/2) ||w||²  +  C · Σ_i ξ_i
#       s.t.  y_i (<w, x_i> + b) ≥ 1 − ξ_i,   ξ_i ≥ 0
#
# ── Part II: KKT conditions and the dual  ("the math of SVMs") ──────────────
#
#   For a constrained problem  min f(x)  s.t.  g(x) ≥ 0  the KKT conditions are
#       L(x, λ) = f(x) − λ g(x)
#       g(x) ≥ 0,   λ ≥ 0,   λ g(x) = 0          (complementary slackness)
#
#     Inactive constraint:  g(x) > 0,  λ = 0  → constraint plays no role.
#     Active   constraint:  g(x) = 0,  λ > 0  → solution pinned at boundary.
#
#   Introducing multipliers (α_i, α_i*) ≥ 0 on the two SVR tube constraints
#   leads to the dual quadratic programme
#       min_{α,α*}  (1/2) Σ_{i,j} (α_i − α_i*)(α_j − α_j*) <x_i, x_j>
#                 + ε Σ_i (α_i + α_i*) − Σ_i y_i (α_i − α_i*)
#       s.t.  Σ_i (α_i − α_i*) = 0,   0 ≤ α_i, α_i* ≤ C
#
#   This is a convex QP → a UNIQUE global minimum (no local minima problem
#   like NNs). The solution is the "support vector expansion"
#       f(x) = Σ_i (α_i − α_i*) <x_i, x> + b              (regression)
#       f(x) = Σ_i  α_i  y_i    <x_i, x> + b              (classification)
#
#   By complementary slackness, points strictly inside the ε-tube (or
#   strictly outside the margin in the classification case) have
#   α_i = α_i* = 0 — they DROP from the sum. Only support vectors remain.
#   Hence "the complexity of the SVM representation is set by the number
#   of SVs, not by the dimension of x" (slides Part II control question 6).
#
#   In scikit-learn:
#       svm.dual_coef_[0]    ==  α_i y_i      (classification, signed)
#                            ==  α_i − α_i*   (regression,    signed)
#       svm.support_         ==  indices of support vectors
#       svm.support_vectors_ ==  the SV inputs x_i (in scaled space here)
#       svm.intercept_       ==  b
#
# ── Part III: The kernel trick  ("Support Vector Machines part III") ────────
#
#   Both the dual and f(x) depend on x only through dot products <x_i, x_j>.
#   Replace each dot product by a kernel
#
#       k(x, x') = <Φ(x), Φ(x')>
#
#   for some (possibly infinite-dimensional) feature map Φ. We never have
#   to materialize Φ explicitly — only k.
#
#   Slide example (quadratic features on R²):
#       Φ(x) = (x_1², √2 x_1 x_2, x_2²) ∈ R³
#       <Φ(x), Φ(x')> = (x_1 x_1' + x_2 x_2')² = <x, x'>²
#   We verify this numerically in kernel_trick_demo() below.
#
#   The four standard Mercer-admissible kernels (slide "Examples of kernels"):
#       linear     k(x, x') = <x, x'>
#       polynomial k(x, x') = (γ <x, x'> + r)^d
#       sigmoid    k(x, x') = tanh(γ <x, x'> + r)
#       Gaussian   k(x, x') = exp(-γ ||x − x'||²)      ← infinite-dim Φ
#
#   γ, r, d join (C, ε) as hyperparameters tuned by cross-validation.
#   ν-SVR replaces ε with ν ∈ (0, 1] that bounds the fraction of training
#   errors AND lower-bounds the fraction of support vectors — often easier
#   to tune than ε directly.
# ============================================================================

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, mean_absolute_error, precision_recall_curve, r2_score,
    roc_auc_score, roc_curve,
)
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, learning_curve, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, NuSVR


PLOT_DIR = "svm_plots"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Synthetic datasets (classification target + continuous spread target)
# ---------------------------------------------------------------------------
def make_loan_dataset(n_samples: int = 3000,
                      random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Realistic-shape borrowers with a known default-generating process."""
    rng = np.random.default_rng(random_state)
    debt_to_income    = rng.beta(2, 5, n_samples) * 1.2
    credit_util       = rng.beta(2, 4, n_samples)
    missed_payments   = rng.poisson(lam=0.6, size=n_samples)
    savings_ratio     = rng.beta(2, 5, n_samples) * 0.5
    income_volatility = np.abs(rng.normal(0.15, 0.1, n_samples))
    loan_to_income    = rng.gamma(2.0, 0.3, n_samples)
    credit_age_years  = np.clip(rng.normal(8, 5, n_samples), 0.5, 40)

    logit = (
        -4.0
        + 2.8 * debt_to_income
        + 2.0 * credit_util
        + 0.9 * missed_payments
        - 3.0 * savings_ratio
        + 2.5 * income_volatility
        + 1.2 * loan_to_income
        - 0.05 * credit_age_years
    )
    p_default = 1.0 / (1.0 + np.exp(-logit))
    default = (rng.uniform(size=n_samples) < p_default).astype(int)

    return pd.DataFrame({
        "debt_to_income":     debt_to_income,
        "credit_utilization": credit_util,
        "missed_payments":    missed_payments,
        "savings_ratio":      savings_ratio,
        "income_volatility":  income_volatility,
        "loan_to_income":     loan_to_income,
        "credit_age_years":   credit_age_years,
        "default":            default,
    })


def make_spread_dataset(n_samples: int = 1500,
                        random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Continuous regression target: a borrower-level 'fair credit spread'.

    Used to illustrate SVR (slides Part I / III). The spread (annualized %)
    is a smooth function of the same risk factors plus irreducible noise.
    """
    rng = np.random.default_rng(random_state)
    dti = rng.beta(2, 5, n_samples) * 1.2
    cu  = rng.beta(2, 4, n_samples)
    mp  = rng.poisson(0.6, n_samples)
    sr  = rng.beta(2, 5, n_samples) * 0.5
    iv  = np.abs(rng.normal(0.15, 0.1, n_samples))
    lti = rng.gamma(2.0, 0.3, n_samples)
    cay = np.clip(rng.normal(8, 5, n_samples), 0.5, 40)

    spread = (
        0.02
        + 0.05  * dti
        + 0.04  * cu
        + 0.015 * mp
        - 0.06  * sr
        + 0.07  * iv
        + 0.02  * lti
        - 0.001 * cay
        + rng.normal(0, 0.008, n_samples)   # irreducible noise
    )
    return pd.DataFrame({
        "debt_to_income":     dti,
        "credit_utilization": cu,
        "missed_payments":    mp,
        "savings_ratio":      sr,
        "income_volatility":  iv,
        "loan_to_income":     lti,
        "credit_age_years":   cay,
        "spread":             spread,
    })


# ---------------------------------------------------------------------------
# 2. Pipeline helper + utilities
# ---------------------------------------------------------------------------
# WHY SCALE (Part III): the RBF kernel exp(−γ ||x − x'||²) is distance-based.
# Without scaling, features on large numeric ranges (credit_age in years)
# would dominate ||·||² and crush features on small ranges (ratios in [0,1]).
# StandardScaler puts every column on mean 0, std 1 so all features
# contribute comparably to the kernel.
def make_pipeline(**svc_kwargs) -> Pipeline:
    svc_kwargs.setdefault("class_weight", "balanced")
    svc_kwargs.setdefault("random_state", RANDOM_STATE)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(**svc_kwargs)),
    ])


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# 3. Kernel comparison (slides Part III)
# ---------------------------------------------------------------------------
def kernel_comparison(X_train, y_train, X_test, y_test) -> None:
    """One SVM per kernel from the slides; compare AUC and SV count.

    Slide formulas (Part III):
      linear     k(x, x') = <x, x'>
      polynomial k(x, x') = (γ <x, x'> + r)^d
      sigmoid    k(x, x') = tanh(γ <x, x'> + r)
      RBF        k(x, x') = exp(-γ ||x - x'||²)        ← infinite-dim Φ
    """
    section("PART III — Kernel comparison")
    kernels = [
        ("linear",  dict(kernel="linear",  C=1.0)),
        ("poly-d2", dict(kernel="poly",    C=1.0, degree=2, gamma="scale", coef0=1.0)),
        ("poly-d3", dict(kernel="poly",    C=1.0, degree=3, gamma="scale", coef0=1.0)),
        ("rbf",     dict(kernel="rbf",     C=1.0, gamma="scale")),
        ("sigmoid", dict(kernel="sigmoid", C=1.0, gamma="scale", coef0=0.0)),
    ]
    rows = []
    for name, params in kernels:
        pipe = make_pipeline(**params)
        pipe.fit(X_train, y_train)
        score = pipe.decision_function(X_test)
        rows.append({
            "kernel": name,
            "test_AUC": roc_auc_score(y_test, score),
            "n_support_vectors": pipe.named_steps["svm"].support_.size,
            "%SV": 100 * pipe.named_steps["svm"].support_.size / len(X_train),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format="%.4f"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(table["kernel"], table["test_AUC"], color="steelblue")
    ax1.set_ylim(0.5, 1.0); ax1.set_ylabel("Test ROC-AUC")
    ax1.set_title("Kernel choice vs. test AUC")
    ax2.bar(table["kernel"], table["%SV"], color="indianred")
    ax2.set_ylabel("% of training points used as SV")
    ax2.set_title("Model complexity (lower = simpler)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/kernel_comparison.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 4. (C, γ) heatmap
# ---------------------------------------------------------------------------
def cgamma_heatmap(X_train, y_train) -> tuple[Pipeline, dict]:
    """Grid-search RBF-SVM over (C, γ) and visualize the CV-AUC surface.

    Intuition (Parts I + III):
      C small  → wide margin, many slack violations, simpler model (high bias)
      C large  → narrow margin, few violations,     complex model  (high variance)
      γ small  → wide kernel, smooth boundary,                    high bias
      γ large  → narrow kernel, wiggly boundary,                  high variance
    """
    section("PART III — (C, γ) hyperparameter sweep")
    C_grid     = [0.01, 0.1, 1.0, 10.0, 100.0]
    gamma_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    pipe = make_pipeline(kernel="rbf")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipe,
        param_grid={"svm__C": C_grid, "svm__gamma": gamma_grid},
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Best params : {grid.best_params_}")
    print(f"Best CV AUC : {grid.best_score_:.4f}")

    auc_grid = (
        pd.DataFrame(grid.cv_results_)
        .pivot_table(index="param_svm__gamma", columns="param_svm__C",
                     values="mean_test_score")
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(auc_grid.values, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(C_grid)));     ax.set_xticklabels(C_grid)
    ax.set_yticks(range(len(gamma_grid))); ax.set_yticklabels(gamma_grid)
    ax.set_xlabel("C  (margin softness)")
    ax.set_ylabel(r"$\gamma$  (kernel bandwidth)")
    ax.set_title("5-fold CV ROC-AUC over (C, $\\gamma$)")
    for i in range(auc_grid.shape[0]):
        for j in range(auc_grid.shape[1]):
            ax.text(j, i, f"{auc_grid.values[i, j]:.3f}",
                    ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="CV AUC")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/hyperparameter_heatmap.png", dpi=120)
    plt.close()
    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------------------
# 5. Support-vector / dual-variable analysis (slides Part II)
# ---------------------------------------------------------------------------
def support_vector_analysis(model: Pipeline, X_train) -> None:
    """Inspect (α_i y_i) from the dual solution.

    Slide notation map:
        sklearn `svm.dual_coef_[0, j]`  ==  α_i y_i      (signed)
        sklearn `svm.support_[j]`       ==  index of the j-th SV
        |α_i| ∈ (0, C) → "free" SV: sits ON the margin    (slack ξ_i = 0)
        |α_i| = C      → "bounded" SV: inside margin or wrong side
                                       (slack ξ_i > 0)
        α_i = 0        → not a support vector             (KKT: α g = 0)
    """
    section("PART II — Support-vector / dual-variable analysis")
    svm: SVC = model.named_steps["svm"]
    C = svm.C
    dual_coef = svm.dual_coef_.ravel()
    alphas = np.abs(dual_coef)
    n_sv = alphas.size
    tol = 1e-6 * C
    n_free    = int(np.sum(alphas < C - tol))
    n_bounded = int(np.sum(alphas >= C - tol))

    print(f"Total support vectors      : {n_sv}  "
          f"({100 * n_sv / len(X_train):.1f}% of training set)")
    print(f"  per class (negative, positive) : {tuple(svm.n_support_)}")
    print(f"Free SVs (|α| < C, on margin)   : {n_free}")
    print(f"Bounded SVs (|α| = C, slack>0)  : {n_bounded}")
    print(f"Intercept b                    : {svm.intercept_[0]:+.4f}")
    print("Slide claim verified: the model's complexity is set by the "
          f"{n_sv} SVs, not by the {len(X_train)} training points.")


# ---------------------------------------------------------------------------
# 6. Manual reconstruction of the support-vector expansion (slides Part II)
# ---------------------------------------------------------------------------
def manual_decision_function(model: Pipeline, X_test) -> None:
    """Recompute f(x) = Σ_i (α_i y_i) k(x_i, x) + b by hand.

    The slides state the SV expansion as
        f(x) = Σ_i (α_i − α_i*) k(x_i, x) + b           (regression)
        f(x) = Σ_i  α_i y_i      k(x_i, x) + b           (classification)
    sklearn stores α_i y_i in `dual_coef_[0]` and x_i in `support_vectors_`
    (the latter already in the *scaled* feature space). We rebuild f(x) from
    these primitives and confirm it matches `model.decision_function`.
    """
    section("PART II — Manual SV expansion vs. sklearn.decision_function")
    scaler = model.named_steps["scaler"]
    svm:  SVC = model.named_steps["svm"]

    # Scale into the same space as the SVs.
    Xs = scaler.transform(X_test[:5])
    SV   = svm.support_vectors_   # (n_SV, d)  in scaled space
    dual = svm.dual_coef_[0]      # (n_SV,)    = α_i y_i
    b    = svm.intercept_[0]
    gamma = svm._gamma            # resolved numeric γ (handles "scale")

    # RBF: k(x_i, x) = exp(-γ ||x_i - x||²)
    diff   = SV[None, :, :] - Xs[:, None, :]
    sqdist = np.sum(diff ** 2, axis=-1)
    K      = np.exp(-gamma * sqdist)
    f_manual  = K @ dual + b
    f_sklearn = model.decision_function(X_test[:5])

    print("  test_idx   manual_f(x)        sklearn.decision_function(x)")
    for i, (m, s) in enumerate(zip(f_manual, f_sklearn)):
        print(f"     {i:3d}      {m:+.8f}          {s:+.8f}")
    print(f"\nMax |manual − sklearn| = {np.abs(f_manual - f_sklearn).max():.2e}")
    print("→ The SV expansion from the slides is what sklearn computes.")


# ---------------------------------------------------------------------------
# 7. Kernel-trick verification (slides Part III)
# ---------------------------------------------------------------------------
def kernel_trick_demo() -> None:
    """Verify  <Φ(x), Φ(x')> = <x, x'>²  for the slide's quadratic feature map.

    Slide example:
        Φ : R² → R³,  Φ(x) = (x_1², √2 x_1 x_2, x_2²)
        <Φ(x), Φ(x')> = (x_1 x_1' + x_2 x_2')² = <x, x'>²

    This is exactly the polynomial kernel with γ = 1, r = 0, d = 2.
    The slides also note that the Gaussian RBF kernel corresponds to an
    INFINITE-dimensional Φ — we cannot write it down, but we can still
    evaluate k(x, x') = exp(-γ ||x − x'||²).
    """
    section("PART III — Kernel trick: verify <Φ(x), Φ(x')> = <x, x'>²")
    rng = np.random.default_rng(0)
    x, xp = rng.standard_normal(2), rng.standard_normal(2)

    def phi(z):  # explicit feature map from the slide
        return np.array([z[0] ** 2, np.sqrt(2) * z[0] * z[1], z[1] ** 2])

    lhs = float(np.dot(phi(x), phi(xp)))
    rhs = float(np.dot(x, xp) ** 2)
    print(f"  x   = {x}")
    print(f"  x'  = {xp}")
    print(f"  explicit <Φ(x), Φ(x')> = {lhs:+.6f}")
    print(f"  shortcut  <x, x'>²    = {rhs:+.6f}")
    print(f"  |difference|          = {abs(lhs - rhs):.2e}    ← kernel trick holds")

    k_poly = polynomial_kernel([x], [xp], degree=2, gamma=1, coef0=0)[0, 0]
    k_rbf  = rbf_kernel([x], [xp], gamma=1.0)[0, 0]
    print(f"  sklearn polynomial_kernel(degree=2, γ=1, r=0) = {k_poly:+.6f}")
    print(f"  sklearn rbf_kernel(γ=1) = {k_rbf:.6f}   "
          f"(features are INFINITE-dimensional — slide note)")


# ---------------------------------------------------------------------------
# 8. Convex optimization → unique global minimum (slides Part II)
# ---------------------------------------------------------------------------
def convexity_demo(X_train, y_train) -> None:
    """Slides Part II: 'SVM training amounts to convex optimization ⇒ unique
    minimum (unlike NNs where local minima are a real problem).'

    We train the same SVM with several `random_state` values and confirm
    the dual solution (n_SV, intercept b, Σ α_i y_i) is identical.
    """
    section("PART II — Convex objective: identical solutions across seeds")
    print("  seed   n_SV       b           Σ α_i y_i        status")
    ref = None
    for seed in [0, 1, 7, 42, 999]:
        pipe = make_pipeline(kernel="rbf", C=1.0, gamma="scale",
                             random_state=seed)
        pipe.fit(X_train, y_train)
        svm = pipe.named_steps["svm"]
        sig = (svm.support_.size,
               round(float(svm.intercept_[0]), 8),
               round(float(svm.dual_coef_.sum()), 8))
        if ref is None:
            ref = sig
        status = "match" if sig == ref else "DIFFER"
        print(f"  {seed:4d}   {sig[0]:4d}  {sig[1]:+.6f}   {sig[2]:+.6e}    {status}")
    print("→ Dual QP is convex; the SVM has a unique global optimum.")


# ---------------------------------------------------------------------------
# 9. Diagnostics: decision function, calibration, ROC, PR
# ---------------------------------------------------------------------------
def diagnostics(model: Pipeline, X_train, y_train, X_test, y_test) -> None:
    """Raw SVM scores are NOT probabilities — calibrate them.

    SVC(probability=True) uses Platt scaling (fit a logistic on the margin
    score via internal CV). CalibratedClassifierCV(method="isotonic") is
    non-parametric — more flexible, needs more data. Both are post-hoc.
    """
    section("Calibration: SVM margin → probability (Platt vs. isotonic)")
    base = model.named_steps["svm"].get_params()

    platt_params = dict(base, probability=True)
    platt_pipe = Pipeline([("scaler", StandardScaler()),
                           ("svm",    SVC(**platt_params))])
    platt_pipe.fit(X_train, y_train)

    iso_pipe = CalibratedClassifierCV(
        Pipeline([("scaler", StandardScaler()),
                  ("svm",    SVC(**dict(base, probability=False)))]),
        method="isotonic", cv=5,
    )
    iso_pipe.fit(X_train, y_train)

    score_test  = model.decision_function(X_test)
    proba_platt = platt_pipe.predict_proba(X_test)[:, 1]
    proba_iso   = iso_pipe.predict_proba(X_test)[:, 1]

    print(f"Raw decision_function range : [{score_test.min():+.3f}, "
          f"{score_test.max():+.3f}]   (this is a signed margin, not a probability)")
    print(f"Platt   probability range   : [{proba_platt.min():.3f}, "
          f"{proba_platt.max():.3f}]")
    print(f"Isotonic probability range  : [{proba_iso.min():.3f}, "
          f"{proba_iso.max():.3f}]")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.hist(score_test[y_test == 0], bins=40, alpha=0.6, label="repaid",
            color="steelblue")
    ax.hist(score_test[y_test == 1], bins=40, alpha=0.6, label="default",
            color="indianred")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("decision_function(x)  (raw SVM margin score)")
    ax.set_ylabel("count")
    ax.set_title("Raw SVM margin score by class"); ax.legend()

    ax = axes[0, 1]
    for proba, name in [(proba_platt, "Platt (SVC built-in)"),
                        (proba_iso,   "Isotonic")]:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("Predicted P(default)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration curves"); ax.legend(loc="upper left")

    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(y_test, score_test)
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, score_test):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend()

    ax = axes[1, 1]
    prec, rec, _ = precision_recall_curve(y_test, score_test)
    ax.plot(rec, prec,
            label=f"AP = {average_precision_score(y_test, score_test):.3f}")
    ax.axhline(y_test.mean(), color="k", ls="--", lw=1,
               label=f"base rate = {y_test.mean():.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve"); ax.legend()

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/diagnostics.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 10. Permutation importance, threshold tuning, learning curve, 2-D boundary
# ---------------------------------------------------------------------------
def permutation_importances(model: Pipeline, X_test, y_test,
                            feature_names: list[str]) -> None:
    """RBF SVMs lack `coef_`. Permutation importance is the standard tool."""
    section("Permutation feature importance")
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)
    sorted_names = [feature_names[i] for i in order]
    for name, m, s in zip(reversed(sorted_names),
                          reversed(result.importances_mean[order]),
                          reversed(result.importances_std[order])):
        print(f"  {name:20s}  drop in AUC = {m:+.4f}  ± {s:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(sorted_names, result.importances_mean[order],
            xerr=result.importances_std[order], color="seagreen")
    ax.set_xlabel("Drop in test AUC when feature is shuffled")
    ax.set_title("Permutation feature importance (RBF SVM)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/permutation_importance.png", dpi=120)
    plt.close()


def threshold_tuning(model: Pipeline, X_test, y_test,
                     cost_fn: float = 5.0, cost_fp: float = 1.0) -> None:
    """Find the threshold that minimizes expected loss, not accuracy."""
    section("Threshold tuning under asymmetric mis-classification cost")
    score = model.decision_function(X_test)
    rescaled = (score - score.min()) / (score.max() - score.min())
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        y_pred = (rescaled >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        costs.append(cost_fn * fn + cost_fp * fp)
    costs = np.array(costs)
    best = int(np.argmin(costs))
    print(f"FN cost = {cost_fn}, FP cost = {cost_fp}")
    print(f"Best threshold (rescaled)      : {thresholds[best]:.2f}")
    print(f"Expected loss @ best threshold : {costs[best]:.1f}")
    print(f"Expected loss @ threshold 0.50 : "
          f"{costs[np.argmin(np.abs(thresholds - 0.5))]:.1f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, costs, color="purple")
    ax.axvline(thresholds[best], color="green", ls="--",
               label=f"min @ t={thresholds[best]:.2f}")
    ax.axvline(0.5, color="grey", ls=":", label="default t=0.5")
    ax.set_xlabel("Decision threshold (rescaled score)")
    ax.set_ylabel(f"Expected loss  ({cost_fn}·FN + {cost_fp}·FP)")
    ax.set_title("Threshold optimization under asymmetric cost"); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/threshold_cost.png", dpi=120)
    plt.close()


def learning_curve_plot(model: Pipeline, X, y) -> None:
    section("Learning curve")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sizes, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=np.linspace(0.1, 1.0, 6),
            cv=StratifiedKFold(4, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE,
        )
    train_mean, train_std = train_scores.mean(1), train_scores.std(1)
    val_mean,   val_std   = val_scores.mean(1),   val_scores.std(1)
    print("size   train AUC    val AUC")
    for n, t, v in zip(sizes, train_mean, val_mean):
        print(f"{n:5d}     {t:.3f}      {v:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, train_mean, "o-", label="train", color="steelblue")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    color="steelblue", alpha=0.2)
    ax.plot(sizes, val_mean, "o-", label="validation", color="indianred")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    color="indianred", alpha=0.2)
    ax.set_xlabel("Training set size"); ax.set_ylabel("ROC AUC")
    ax.set_title("Learning curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/learning_curve.png", dpi=120)
    plt.close()


def decision_boundary_2d(df: pd.DataFrame,
                         feature_x: str = "debt_to_income",
                         feature_y: str = "credit_utilization") -> None:
    """Refit an RBF SVC on two features so the boundary can be drawn.

    Contours show f(x) = 0 (decision boundary) and f(x) = ±1 (margin band).
    Circled points are support vectors — exactly the borrowers contributing
    to the support-vector expansion from slides Part II.
    """
    section("2-D decision-boundary visualization")
    X2 = df[[feature_x, feature_y]].values
    y2 = df["default"].values
    pipe = make_pipeline(kernel="rbf", C=10.0, gamma="scale")
    pipe.fit(X2, y2)
    svm = pipe.named_steps["svm"]

    pad_x = 0.05 * (X2[:, 0].max() - X2[:, 0].min())
    pad_y = 0.05 * (X2[:, 1].max() - X2[:, 1].min())
    xx, yy = np.meshgrid(
        np.linspace(X2[:, 0].min() - pad_x, X2[:, 0].max() + pad_x, 250),
        np.linspace(X2[:, 1].min() - pad_y, X2[:, 1].max() + pad_y, 250),
    )
    Z = pipe.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, levels=[-1e3, 0, 1e3],
                colors=["#cfe2f3", "#f4cccc"], alpha=0.55)
    cs = ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                    colors="black", linestyles=["--", "-", "--"])
    ax.clabel(cs, fmt={-1: "f=-1", 0: "f=0", 1: "f=+1"}, fontsize=8)

    rng = np.random.default_rng(RANDOM_STATE)
    idx_repay = rng.choice(np.where(y2 == 0)[0], size=400, replace=False)
    idx_default = rng.choice(np.where(y2 == 1)[0],
                             size=min(400, int((y2 == 1).sum())), replace=False)
    ax.scatter(X2[idx_repay, 0], X2[idx_repay, 1], s=8, c="steelblue",
               alpha=0.7, label="repaid")
    ax.scatter(X2[idx_default, 0], X2[idx_default, 1], s=8, c="indianred",
               alpha=0.7, label="default")
    sv_idx = svm.support_
    ax.scatter(X2[sv_idx, 0], X2[sv_idx, 1], s=40, facecolors="none",
               edgecolors="black", linewidths=0.6,
               label=f"support vectors ({len(sv_idx)})")
    ax.set_xlabel(feature_x); ax.set_ylabel(feature_y)
    ax.set_title("RBF-SVM decision boundary on two features\n"
                 "(solid: f=0, dashed: ±1 margin from the slides)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/decision_boundary_2d.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 11. Support Vector Regression (slides Part I / III)
# ---------------------------------------------------------------------------
def svr_demo() -> None:
    """ε-SVR and ν-SVR on a continuous fair-credit-spread target.

    Demonstrates the slide concepts that don't show up in classification:
      · The ε-insensitive loss: predictions inside the ε-tube get zero
        penalty; outside, penalty grows linearly.
      · The sparse SV expansion: borrowers inside the tube have α_i = α_i* = 0
        and DROP from f(x) = Σ_i (α_i − α_i*) k(x_i, x) + b.
      · ν-SVR: reparameterizes ε via ν ∈ (0, 1]. ν bounds the fraction of
        margin errors AND lower-bounds the fraction of SVs. Often easier
        to tune than ε directly.
    """
    section("PART I/III — ε-SVR and ν-SVR on credit spreads")
    df = make_spread_dataset()
    feat = [c for c in df.columns if c != "spread"]
    X, y = df[feat].values, df["spread"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    eps_value = 0.01
    eps_pipe = Pipeline([("scaler", StandardScaler()),
                         ("svr", SVR(kernel="rbf", C=1.0, gamma="scale",
                                     epsilon=eps_value))])
    nu_pipe = Pipeline([("scaler", StandardScaler()),
                        ("svr", NuSVR(kernel="rbf", C=1.0, gamma="scale",
                                      nu=0.5))])
    eps_pipe.fit(X_train, y_train)
    nu_pipe.fit(X_train, y_train)

    # Empirical ε for NuSVR: median residual at the FREE support vectors,
    # i.e. SVs with |α_i − α_i*| < C — these lie exactly on the ε-tube edge.
    nu_svr = nu_pipe.named_steps["svr"]
    sv_idx = nu_svr.support_
    if sv_idx.size:
        sv_resid = np.abs(y_train[sv_idx] - nu_pipe.predict(X_train[sv_idx]))
        free_mask = np.abs(nu_svr.dual_coef_.ravel()) < nu_svr.C - 1e-6
        nu_eps_emp = (float(np.median(sv_resid[free_mask]))
                      if free_mask.any() else float("nan"))
    else:
        nu_eps_emp = float("nan")

    print(f"{'model':16s} {'MAE':>8s} {'R²':>7s} {'n_SV':>8s} {'%SV':>7s} {'ε (used)':>12s}")
    for name, p, eps_used in [
        ("ε-SVR (ε=0.01)", eps_pipe, eps_value),
        ("ν-SVR (ν=0.5)",  nu_pipe,  nu_eps_emp),
    ]:
        y_hat = p.predict(X_test)
        mae = mean_absolute_error(y_test, y_hat)
        r2  = r2_score(y_test, y_hat)
        n_sv = p.named_steps["svr"].support_.size
        eps_str = f"{eps_used:.4f}" if not np.isnan(eps_used) else "—"
        print(f"{name:16s} {mae:8.4f} {r2:7.4f} {n_sv:8d} "
              f"{100*n_sv/len(X_train):6.1f}% {eps_str:>12s}")
    print("ν is an UPPER bound on the fraction of margin errors and a LOWER "
          "bound on the SV fraction (Schölkopf et al., 2000). With noisy data "
          "the SV fraction can exceed ν substantially.")

    # Plot predicted-vs-actual with ε-tube.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, p, name, eps in [
        (axes[0], eps_pipe, f"ε-SVR  (fixed ε = {eps_value})", eps_value),
        (axes[1], nu_pipe,
         f"ν-SVR  (ν = 0.5, empirical ε ≈ {nu_eps_emp:.3f})", nu_eps_emp),
    ]:
        y_hat = p.predict(X_test)
        ax.scatter(y_test, y_hat, s=10, alpha=0.6, color="steelblue",
                   label="test predictions")
        lo = min(y_test.min(), y_hat.min())
        hi = max(y_test.max(), y_hat.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ŷ = y")
        if not np.isnan(eps):
            ax.fill_between([lo, hi], [lo - eps, hi - eps],
                            [lo + eps, hi + eps],
                            color="red", alpha=0.15, label=f"±ε tube")
        ax.set_xlabel("Actual spread"); ax.set_ylabel("Predicted spread")
        ax.set_title(name); ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/svr_demo.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 12. Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    # PART I dataset
    df = make_loan_dataset()
    print(f"Dataset      : {len(df)} borrowers")
    print(f"Default rate : {df['default'].mean():.2%}")

    feature_cols = [c for c in df.columns if c != "default"]
    X = df[feature_cols].values
    y = df["default"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # ---- PART III: kernel head-to-head, then tune RBF (C, γ) ----
    kernel_comparison(X_train, y_train, X_test, y_test)
    best_model, best_params = cgamma_heatmap(X_train, y_train)

    # ---- Test-set evaluation of the tuned RBF SVM ----
    section("Tuned RBF SVM — test-set evaluation")
    y_pred  = best_model.predict(X_test)
    y_score = best_model.decision_function(X_test)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, y_score):.4f}")
    print(f"PR  AP   : {average_precision_score(y_test, y_score):.4f}")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["actual_repaid", "actual_default"],
        columns=["pred_repaid", "pred_default"],
    ))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["repaid", "default"]))

    # Logistic-regression baseline so we know the kernel SVM earns its keep.
    lr = Pipeline([("scaler", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=2000,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE))])
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    print(f"\nBaseline logistic-regression AUC : {lr_auc:.4f}    "
          f"(RBF SVM AUC: {roc_auc_score(y_test, y_score):.4f})")

    # ---- PART II demos ----
    support_vector_analysis(best_model, X_train)
    manual_decision_function(best_model, X_test)
    convexity_demo(X_train, y_train)

    # ---- PART III demos ----
    kernel_trick_demo()

    # ---- Applied layer ----
    diagnostics(best_model, X_train, y_train, X_test, y_test)
    permutation_importances(best_model, X_test, y_test, feature_cols)
    threshold_tuning(best_model, X_test, y_test, cost_fn=5.0, cost_fp=1.0)
    learning_curve_plot(best_model, X, y)
    decision_boundary_2d(df)

    # ---- SVR demo ----
    svr_demo()

    # ---- Score a brand-new borrower ----
    section("Scoring a brand-new borrower")
    new_borrower = pd.DataFrame([{
        "debt_to_income":     0.55,
        "credit_utilization": 0.78,
        "missed_payments":    2,
        "savings_ratio":      0.05,
        "income_volatility":  0.30,
        "loan_to_income":     0.9,
        "credit_age_years":   3.0,
    }])
    margin = best_model.decision_function(new_borrower[feature_cols].values)[0]
    decision = "DEFAULT" if margin >= 0 else "REPAY"
    print(f"New borrower profile:\n{new_borrower.T}")
    print(f"\nRaw SVM margin score : {margin:+.3f}")
    print(f"Predicted class      : {decision}")
    print("Positive margin → the borrower lies on the 'default' side of f(x)=0.")
    print(f"All plots written to ./{PLOT_DIR}/")


if __name__ == "__main__":
    main()
