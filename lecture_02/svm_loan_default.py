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

# ---------------------------------------------------------------------------
# Imports — every dependency is annotated with the SVM concept it supports.
# ---------------------------------------------------------------------------
# `from __future__ import annotations` makes type hints strings at runtime,
# so we can use Python ≥3.10 syntax like `tuple[Pipeline, dict]` on older
# Python versions without a syntax error at import time.
from __future__ import annotations

# `os` is used only to create the output directory for plots (os.makedirs).
import os
# `warnings` lets us silence noisy sklearn warnings inside the learning_curve
# call (some sklearn versions warn when SVC encounters ties).
import warnings

# NumPy is the numerical workhorse: arrays, random-number generation,
# vectorised math, and the linear-algebra used in the manual kernel demo.
import numpy as np
# pandas is used for tabular display (DataFrames printed to stdout) and for
# the synthetic-borrower dataset.
import pandas as pd
# matplotlib must be configured BEFORE we import pyplot — "Agg" is a
# non-interactive backend that writes PNGs without needing a display.
# This makes the script run on CI / headless servers.
import matplotlib
matplotlib.use("Agg")
# pyplot is the plotting API we actually call to draw figures.
import matplotlib.pyplot as plt

# `CalibratedClassifierCV` wraps any classifier and produces probabilities
# via Platt (sigmoid) or isotonic regression — needed because raw SVM
# scores are signed margins, NOT probabilities. `calibration_curve` is the
# reliability-diagram helper used in the diagnostics panel.
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
# Permutation importance is the standard model-agnostic feature attribution.
# It's the only sensible importance metric for an RBF-kernel SVM since there
# is no `coef_` in input space (the model lives in feature space Φ).
from sklearn.inspection import permutation_importance
# Logistic regression is included as a baseline so we can see whether the
# RBF kernel buys anything over a linear model.
from sklearn.linear_model import LogisticRegression
# Classification metrics:
#   accuracy_score           — fraction of correct labels.
#   average_precision_score  — area under the PR curve (good for imbalance).
#   classification_report    — precision / recall / F1 per class.
#   confusion_matrix         — 2×2 count matrix; (TN, FP, FN, TP) when raveled.
#   mean_absolute_error      — used by the SVR demo.
#   precision_recall_curve   — point-cloud of PR pairs for plotting.
#   r2_score                 — coefficient of determination for SVR.
#   roc_auc_score / roc_curve — area under ROC and the curve itself.
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, mean_absolute_error, precision_recall_curve, r2_score,
    roc_auc_score, roc_curve,
)
# `polynomial_kernel` and `rbf_kernel` are direct callable forms of the
# slide formulas — we use them to double-check the kernel-trick demo.
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
# Model-selection utilities:
#   GridSearchCV          — exhaustive search over a hyperparameter grid.
#   StratifiedKFold       — CV split that preserves the class proportions.
#   learning_curve        — train/val scores at multiple training sizes.
#   train_test_split      — single-shot train/test split with stratification.
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, learning_curve, train_test_split,
)
# `Pipeline` chains transformations (scaler) with an estimator (SVM) so the
# fit/predict interfaces work on a single object — and CV folds correctly
# refit the scaler on each fold (avoids train/test leakage).
from sklearn.pipeline import Pipeline
# `StandardScaler` z-scores each feature column to mean 0, std 1.
# Critical for distance-based kernels — explained at length in make_pipeline.
from sklearn.preprocessing import StandardScaler
# The three SVM estimators we use:
#   SVC   — Support Vector Classifier (binary classification).
#   SVR   — ε-SVR (regression with a fixed ε-insensitive tube).
#   NuSVR — ν-SVR (Schölkopf reparametrization; tunes ε via ν).
from sklearn.svm import SVC, SVR, NuSVR


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Output directory for every PNG this script writes. Relative path → created
# next to the script when run.
PLOT_DIR = "svm_plots"
# Fix every random seed to make the run fully reproducible.
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Synthetic datasets (classification target + continuous spread target)
# ---------------------------------------------------------------------------
def make_loan_dataset(n_samples: int = 3000,
                      random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Realistic-shape borrowers with a known default-generating process."""
    # Use the modern NumPy Generator API for reproducibility.
    rng = np.random.default_rng(random_state)

    # ---- Risk-factor distributions chosen to mimic a real retail book ----
    # Debt-to-income: Beta-shaped, mostly in [0, 0.4], stretched to [0, 1.2].
    debt_to_income    = rng.beta(2, 5, n_samples) * 1.2
    # Credit utilisation: Beta in [0, 1] — fraction of revolving limit used.
    credit_util       = rng.beta(2, 4, n_samples)
    # Missed payments in the last 12 months: Poisson with low mean (most
    # borrowers have 0; a long tail with 1–3+).
    missed_payments   = rng.poisson(lam=0.6, size=n_samples)
    # Savings ratio (savings / income): Beta in [0, 0.5].
    savings_ratio     = rng.beta(2, 5, n_samples) * 0.5
    # Income volatility: absolute Gaussian (so always ≥ 0).
    income_volatility = np.abs(rng.normal(0.15, 0.1, n_samples))
    # Loan-to-income at origination: Gamma → right-skewed, occasionally > 1.
    loan_to_income    = rng.gamma(2.0, 0.3, n_samples)
    # Credit-file age (years): truncated Gaussian centred at ~8 years.
    credit_age_years  = np.clip(rng.normal(8, 5, n_samples), 0.5, 40)

    # ---- True default-generating process (a known logistic model) ----
    # Intercept −4.0 sets the base rate at ~18% (realistic for unsecured
    # consumer credit). Coefficients reflect the direction a credit-risk
    # analyst would expect: higher DTI/credit-util/missed-payments push
    # default UP, more savings or older credit file push it DOWN.
    logit = (
        -4.0                                  # base log-odds of default
        + 2.8  * debt_to_income               # more debt → riskier
        + 2.0  * credit_util                  # high revolving use → riskier
        + 0.9  * missed_payments              # past delinquencies are predictive
        - 3.0  * savings_ratio                # savings cushion → safer
        + 2.5  * income_volatility            # unstable income → riskier
        + 1.2  * loan_to_income               # large loan relative to income
        - 0.05 * credit_age_years             # longer credit history → safer
    )
    # Convert log-odds → probability via the logistic CDF.
    p_default = 1.0 / (1.0 + np.exp(-logit))
    # Bernoulli draw: each borrower defaults with prob p_default.
    default = (rng.uniform(size=n_samples) < p_default).astype(int)

    # Return one row per borrower; column order is preserved for downstream code.
    return pd.DataFrame({
        "debt_to_income":     debt_to_income,
        "credit_utilization": credit_util,
        "missed_payments":    missed_payments,
        "savings_ratio":      savings_ratio,
        "income_volatility":  income_volatility,
        "loan_to_income":     loan_to_income,
        "credit_age_years":   credit_age_years,
        "default":            default,           # the binary target {0, 1}
    })


def make_spread_dataset(n_samples: int = 1500,
                        random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Continuous regression target: a borrower-level 'fair credit spread'.

    Used to illustrate SVR (slides Part I / III). The spread (annualized %)
    is a smooth function of the same risk factors plus irreducible noise.
    """
    # Independent RNG so the regression sample is statistically separate
    # from the classification one.
    rng = np.random.default_rng(random_state)
    # Same factor distributions as the classification dataset, abbreviated names.
    dti = rng.beta(2, 5, n_samples) * 1.2
    cu  = rng.beta(2, 4, n_samples)
    mp  = rng.poisson(0.6, n_samples)
    sr  = rng.beta(2, 5, n_samples) * 0.5
    iv  = np.abs(rng.normal(0.15, 0.1, n_samples))
    lti = rng.gamma(2.0, 0.3, n_samples)
    cay = np.clip(rng.normal(8, 5, n_samples), 0.5, 40)

    # Linear "fair pricing" model:  spread(%) = base + Σ coeff·factor + noise.
    # Coefficients on roughly the same order of magnitude as historical
    # consumer-credit spread sensitivities.
    spread = (
        0.02                                    # base risk-free + opex spread
        + 0.05  * dti                           # +5% per unit DTI
        + 0.04  * cu                            # +4% per unit utilisation
        + 0.015 * mp                            # +1.5% per missed payment
        - 0.06  * sr                            # −6% per unit savings ratio
        + 0.07  * iv                            # +7% per unit income vol
        + 0.02  * lti                           # +2% per unit loan-to-income
        - 0.001 * cay                           # −0.1% per credit-file year
        + rng.normal(0, 0.008, n_samples)       # irreducible noise σ ≈ 0.8%
    )
    return pd.DataFrame({
        "debt_to_income":     dti,
        "credit_utilization": cu,
        "missed_payments":    mp,
        "savings_ratio":      sr,
        "income_volatility":  iv,
        "loan_to_income":     lti,
        "credit_age_years":   cay,
        "spread":             spread,           # continuous target [%]
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
    # `class_weight="balanced"` re-weights samples inversely to class freq.
    # Defaults can be overridden by the caller via **kwargs.
    svc_kwargs.setdefault("class_weight", "balanced")
    # Pin a default seed (used only for tie-breaking inside SVC — see slides
    # Part II: the dual QP itself is deterministic given hyperparameters).
    svc_kwargs.setdefault("random_state", RANDOM_STATE)
    # Two-step pipeline: scale first (fit-on-train, applied in fold), then SVM.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(**svc_kwargs)),
    ])


def section(title: str) -> None:
    # Pretty-print a banner so long stdout output is navigable.
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
    # Print the section banner.
    section("PART III — Kernel comparison")
    # Each tuple is (display name, kwargs that map directly to a slide formula).
    # `gamma="scale"` makes γ = 1 / (n_features · X.var()) — sklearn's default,
    # works well across features of different scale once StandardScaler is on.
    kernels = [
        ("linear",  dict(kernel="linear",  C=1.0)),                                            # k = <x, x'>
        ("poly-d2", dict(kernel="poly",    C=1.0, degree=2, gamma="scale", coef0=1.0)),        # (γ<x,x'>+1)²
        ("poly-d3", dict(kernel="poly",    C=1.0, degree=3, gamma="scale", coef0=1.0)),        # (γ<x,x'>+1)³
        ("rbf",     dict(kernel="rbf",     C=1.0, gamma="scale")),                             # exp(-γ ||x-x'||²)
        ("sigmoid", dict(kernel="sigmoid", C=1.0, gamma="scale", coef0=0.0)),                  # tanh(γ<x,x'>+0)
    ]
    # Accumulate one summary row per kernel.
    rows = []
    for name, params in kernels:
        # Build a scaler+SVM pipeline for this kernel.
        pipe = make_pipeline(**params)
        # Fit on training data — both scaler and SVM are fit jointly.
        pipe.fit(X_train, y_train)
        # `decision_function` returns the raw signed margin f(x), not a probability.
        score = pipe.decision_function(X_test)
        # ROC-AUC is rank-based so it works directly on the margin score.
        rows.append({
            "kernel": name,
            "test_AUC": roc_auc_score(y_test, score),
            # `support_` is the array of training-row indices that ended up as SVs.
            "n_support_vectors": pipe.named_steps["svm"].support_.size,
            "%SV": 100 * pipe.named_steps["svm"].support_.size / len(X_train),
        })
    # Tabulate and print.
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format="%.4f"))

    # ---- Two-panel bar chart: AUC vs. model complexity ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    # Left panel: test AUC per kernel.
    ax1.bar(table["kernel"], table["test_AUC"], color="steelblue")
    ax1.set_ylim(0.5, 1.0)                    # 0.5 = random; cap at 1.0
    ax1.set_ylabel("Test ROC-AUC")
    ax1.set_title("Kernel choice vs. test AUC")
    # Right panel: % of training set used as support vectors (complexity proxy).
    ax2.bar(table["kernel"], table["%SV"], color="indianred")
    ax2.set_ylabel("% of training points used as SV")
    ax2.set_title("Model complexity (lower = simpler)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    # Save and close (no GUI window — Agg backend).
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
    # Log-spaced grids — standard SVM playbook because the effect of C and γ
    # is multiplicative, not additive.
    C_grid     = [0.01, 0.1, 1.0, 10.0, 100.0]
    gamma_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    # Fresh pipeline; GridSearchCV will substitute the SVC params at each cell.
    pipe = make_pipeline(kernel="rbf")
    # Stratified 5-fold keeps the default rate identical in each fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # `svm__C` is "step name + double-underscore + param" — Pipeline parameter syntax.
    grid = GridSearchCV(
        pipe,
        param_grid={"svm__C": C_grid, "svm__gamma": gamma_grid},
        cv=cv,
        scoring="roc_auc",         # AUC — robust under class imbalance
        n_jobs=-1,                 # parallelise across all CPU cores
    )
    # 5 × 5 = 25 hyperparam settings × 5 folds = 125 SVM fits.
    grid.fit(X_train, y_train)
    print(f"Best params : {grid.best_params_}")
    print(f"Best CV AUC : {grid.best_score_:.4f}")

    # Pivot the long-form cv_results_ table into a (γ rows × C cols) matrix.
    auc_grid = (
        pd.DataFrame(grid.cv_results_)
        .pivot_table(index="param_svm__gamma", columns="param_svm__C",
                     values="mean_test_score")
        .astype(float)
    )

    # ---- Heatmap of CV-AUC over the (C, γ) grid ----
    fig, ax = plt.subplots(figsize=(7, 5))
    # `origin="lower"` so γ axis grows upward.
    im = ax.imshow(auc_grid.values, cmap="viridis", aspect="auto", origin="lower")
    # Custom tick labels = the actual C and γ values.
    ax.set_xticks(range(len(C_grid)));     ax.set_xticklabels(C_grid)
    ax.set_yticks(range(len(gamma_grid))); ax.set_yticklabels(gamma_grid)
    ax.set_xlabel("C  (margin softness)")
    ax.set_ylabel(r"$\gamma$  (kernel bandwidth)")
    ax.set_title("5-fold CV ROC-AUC over (C, $\\gamma$)")
    # Annotate every cell with its CV-AUC value for easy reading.
    for i in range(auc_grid.shape[0]):
        for j in range(auc_grid.shape[1]):
            ax.text(j, i, f"{auc_grid.values[i, j]:.3f}",
                    ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="CV AUC")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/hyperparameter_heatmap.png", dpi=120)
    plt.close()
    # Hand back the refit best estimator + which params won.
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
    # Pull the SVC out of the pipeline so we can read its dual attributes.
    svm: SVC = model.named_steps["svm"]
    # The box constraint upper bound from the primal C·Σξ term.
    C = svm.C
    # `dual_coef_` has shape (1, n_SV) for binary classification.
    # Values are α_i · y_i (signed); .ravel() flattens to 1-D.
    dual_coef = svm.dual_coef_.ravel()
    # |α_i y_i| = α_i since y_i ∈ {-1, +1}.
    alphas = np.abs(dual_coef)
    # Total number of support vectors (rows with non-zero α).
    n_sv = alphas.size
    # Numerical tolerance to call α "equal to C" given floating-point noise.
    tol = 1e-6 * C
    # Free SVs sit exactly on the margin with strictly positive but < C alpha.
    n_free    = int(np.sum(alphas < C - tol))
    # Bounded SVs are inside the margin or misclassified — slack > 0.
    n_bounded = int(np.sum(alphas >= C - tol))

    print(f"Total support vectors      : {n_sv}  "
          f"({100 * n_sv / len(X_train):.1f}% of training set)")
    # `n_support_` gives the SV count per class (negative class, positive class).
    print(f"  per class (negative, positive) : {tuple(svm.n_support_)}")
    print(f"Free SVs (|α| < C, on margin)   : {n_free}")
    print(f"Bounded SVs (|α| = C, slack>0)  : {n_bounded}")
    # `intercept_` is a 1-element array; [0] extracts the scalar bias b.
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
    # Two pipeline steps: scaler (fitted on train data) and the SVM itself.
    scaler = model.named_steps["scaler"]
    svm:  SVC = model.named_steps["svm"]

    # The SVs live in *scaled* feature space, so we must scale the test
    # inputs first before computing distances against them.
    Xs = scaler.transform(X_test[:5])           # only 5 points to verify
    # The training inputs that became support vectors, in scaled space.
    SV   = svm.support_vectors_                 # shape (n_SV, d)
    # Signed dual coefficients α_i y_i (one per SV).
    dual = svm.dual_coef_[0]                    # shape (n_SV,)
    # Scalar bias from the dual problem.
    b    = svm.intercept_[0]
    # The actual numeric γ used by libsvm. "_gamma" handles the "scale"/"auto"
    # string-to-float resolution that sklearn does internally.
    gamma = svm._gamma

    # ---- RBF kernel evaluated pairwise: K_{i,j} = exp(-γ ||SV_i - x_j||²) ----
    # `SV[None, :, :]` adds a leading axis → shape (1, n_SV, d).
    # `Xs[:, None, :]`  adds a middle  axis → shape (n_test, 1, d).
    # Broadcasting → (n_test, n_SV, d) pairwise differences.
    diff   = SV[None, :, :] - Xs[:, None, :]
    # Squared L2 distance per (test point, SV) pair → (n_test, n_SV).
    sqdist = np.sum(diff ** 2, axis=-1)
    # Apply the Gaussian kernel element-wise.
    K      = np.exp(-gamma * sqdist)
    # Manual SV expansion: matmul over SVs + bias.
    f_manual  = K @ dual + b
    # The same quantity computed by sklearn — should match to ~1e-12.
    f_sklearn = model.decision_function(X_test[:5])

    # Side-by-side printout.
    print("  test_idx   manual_f(x)        sklearn.decision_function(x)")
    for i, (m, s) in enumerate(zip(f_manual, f_sklearn)):
        print(f"     {i:3d}      {m:+.8f}          {s:+.8f}")
    # Worst-case absolute error — should be machine-epsilon small.
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
    # Local RNG, seed 0 → reproducible demo numbers.
    rng = np.random.default_rng(0)
    # Two random 2-D points to compare.
    x, xp = rng.standard_normal(2), rng.standard_normal(2)

    def phi(z):
        # The slide's explicit feature map: a 2-vector → a 3-vector.
        # The √2 on the cross term is what makes <Φ, Φ> equal <·,·>².
        return np.array([z[0] ** 2, np.sqrt(2) * z[0] * z[1], z[1] ** 2])

    # LHS: build Φ(x), Φ(x') explicitly, then dot-product in R³.
    lhs = float(np.dot(phi(x), phi(xp)))
    # RHS: kernel shortcut — square the input-space dot product.
    rhs = float(np.dot(x, xp) ** 2)
    print(f"  x   = {x}")
    print(f"  x'  = {xp}")
    print(f"  explicit <Φ(x), Φ(x')> = {lhs:+.6f}")
    print(f"  shortcut  <x, x'>²    = {rhs:+.6f}")
    print(f"  |difference|          = {abs(lhs - rhs):.2e}    ← kernel trick holds")

    # Confirm against sklearn's vectorised polynomial kernel.
    # `polynomial_kernel` takes 2-D inputs (matrix of rows), hence [x] / [xp].
    k_poly = polynomial_kernel([x], [xp], degree=2, gamma=1, coef0=0)[0, 0]
    # The Gaussian RBF kernel — infinite-dim feature map, but easy to evaluate.
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
    # `ref` is the signature of the first solution; later seeds compare to it.
    ref = None
    for seed in [0, 1, 7, 42, 999]:
        # Same hyperparameters, different libsvm tie-breaking seed.
        pipe = make_pipeline(kernel="rbf", C=1.0, gamma="scale",
                             random_state=seed)
        pipe.fit(X_train, y_train)
        svm = pipe.named_steps["svm"]
        # Signature = (n_SV, bias, sum-of-signed-α). If the QP is convex, this
        # tuple should be identical across seeds — modulo numerical noise we
        # absorb by rounding to 8 digits.
        sig = (svm.support_.size,
               round(float(svm.intercept_[0]), 8),
               round(float(svm.dual_coef_.sum()), 8))
        # First iteration stores the reference; subsequent ones must match.
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
    # Capture the tuned SVM's hyperparameters so we can clone them with
    # additional knobs (probability=True / False) below.
    base = model.named_steps["svm"].get_params()

    # ---- Platt-scaling pipeline (SVC's built-in path) ----
    # `probability=True` triggers an internal 5-fold CV that fits a logistic
    # to the decision_function values. Slower to train but `predict_proba`
    # then works directly.
    platt_params = dict(base, probability=True)
    platt_pipe = Pipeline([("scaler", StandardScaler()),
                           ("svm",    SVC(**platt_params))])
    platt_pipe.fit(X_train, y_train)

    # ---- Isotonic-calibration pipeline (sklearn's generic wrapper) ----
    # CalibratedClassifierCV fits the SVM under CV folds and then a
    # non-parametric isotonic regression maps decision_function → probability.
    iso_pipe = CalibratedClassifierCV(
        Pipeline([("scaler", StandardScaler()),
                  ("svm",    SVC(**dict(base, probability=False)))]),
        method="isotonic", cv=5,
    )
    iso_pipe.fit(X_train, y_train)

    # ---- Evaluate the three score sources on the test set ----
    # Raw signed margin (the "f(x)" from the slides).
    score_test  = model.decision_function(X_test)
    # Platt-scaled probabilities (the [:, 1] column = P(class = default)).
    proba_platt = platt_pipe.predict_proba(X_test)[:, 1]
    # Isotonic-calibrated probabilities.
    proba_iso   = iso_pipe.predict_proba(X_test)[:, 1]

    # ---- Console summary ----
    print(f"Raw decision_function range : [{score_test.min():+.3f}, "
          f"{score_test.max():+.3f}]   (this is a signed margin, not a probability)")
    print(f"Platt   probability range   : [{proba_platt.min():.3f}, "
          f"{proba_platt.max():.3f}]")
    print(f"Isotonic probability range  : [{proba_iso.min():.3f}, "
          f"{proba_iso.max():.3f}]")

    # ---- 2×2 panel of diagnostic plots ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (a) Distribution of the raw margin score by true class.
    ax = axes[0, 0]
    ax.hist(score_test[y_test == 0], bins=40, alpha=0.6, label="repaid",
            color="steelblue")
    ax.hist(score_test[y_test == 1], bins=40, alpha=0.6, label="default",
            color="indianred")
    # f(x)=0 is the decision boundary — verify with the eye that defaults
    # mostly fall right of it.
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("decision_function(x)  (raw SVM margin score)")
    ax.set_ylabel("count")
    ax.set_title("Raw SVM margin score by class"); ax.legend()

    # (b) Calibration curves: predicted P vs observed default rate.
    ax = axes[0, 1]
    for proba, name in [(proba_platt, "Platt (SVC built-in)"),
                        (proba_iso,   "Isotonic")]:
        # Bin predictions into 10 equal-width buckets; plot mean(pred) vs
        # fraction-positive.
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", label=name)
    # Perfect calibration would lie on the diagonal y = x.
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("Predicted P(default)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration curves"); ax.legend(loc="upper left")

    # (c) ROC curve — area under it is a rank-based summary.
    ax = axes[1, 0]
    fpr, tpr, _ = roc_curve(y_test, score_test)
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, score_test):.3f}")
    # Random-classifier baseline: y = x line.
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend()

    # (d) Precision-Recall curve — better for imbalanced classes than ROC.
    ax = axes[1, 1]
    prec, rec, _ = precision_recall_curve(y_test, score_test)
    ax.plot(rec, prec,
            label=f"AP = {average_precision_score(y_test, score_test):.3f}")
    # Horizontal baseline = base rate (a constant predictor at this rate
    # would give precision == base rate).
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
    # For each feature j: shuffle column j of X_test, re-score, measure AUC drop.
    # Repeat n_repeats times to get a mean ± std.
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    # `importances_mean[j]` = mean AUC drop when feature j is shuffled.
    # Sort ascending so the most important feature ends up at the top of the plot.
    order = np.argsort(result.importances_mean)
    sorted_names = [feature_names[i] for i in order]
    # Print most → least important in the console.
    for name, m, s in zip(reversed(sorted_names),
                          reversed(result.importances_mean[order]),
                          reversed(result.importances_std[order])):
        print(f"  {name:20s}  drop in AUC = {m:+.4f}  ± {s:.4f}")

    # ---- Horizontal bar chart with error bars ----
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
    # Raw margin score (continuous).
    score = model.decision_function(X_test)
    # Min-max rescale to [0, 1] so the threshold-sweep axis is interpretable.
    rescaled = (score - score.min()) / (score.max() - score.min())
    # 99 candidate thresholds along the rescaled-score axis.
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        # Predict default ↔ rescaled score ≥ t.
        y_pred = (rescaled >= t).astype(int)
        # `.ravel()` gives (TN, FP, FN, TP) in that order.
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # Expected loss = cost_fn × #missed defaults + cost_fp × #wrongly rejected.
        costs.append(cost_fn * fn + cost_fp * fp)
    # Convert to numpy for vector ops.
    costs = np.array(costs)
    # The index of the cheapest threshold.
    best = int(np.argmin(costs))
    print(f"FN cost = {cost_fn}, FP cost = {cost_fp}")
    print(f"Best threshold (rescaled)      : {thresholds[best]:.2f}")
    print(f"Expected loss @ best threshold : {costs[best]:.1f}")
    # Compare against the naive 0.5 cutoff.
    print(f"Expected loss @ threshold 0.50 : "
          f"{costs[np.argmin(np.abs(thresholds - 0.5))]:.1f}")

    # ---- Plot: expected loss as a function of threshold ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, costs, color="purple")
    # Mark the cost-minimising threshold and the default 0.5.
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
    # Some sklearn versions emit a UserWarning about ties; suppress for clarity.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # train_sizes = 6 fractions from 10% to 100% of the data.
        # For each size, do a stratified 4-fold CV; record train AUC and
        # validation AUC at every fold.
        sizes, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=np.linspace(0.1, 1.0, 6),
            cv=StratifiedKFold(4, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE,
        )
    # Mean / std across folds for plotting confidence bands.
    train_mean, train_std = train_scores.mean(1), train_scores.std(1)
    val_mean,   val_std   = val_scores.mean(1),   val_scores.std(1)
    print("size   train AUC    val AUC")
    for n, t, v in zip(sizes, train_mean, val_mean):
        print(f"{n:5d}     {t:.3f}      {v:.3f}")

    # ---- Plot two curves with shaded ±1σ bands ----
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
    # Restrict to the two chosen features so the model lives in R² and the
    # decision surface can be drawn.
    X2 = df[[feature_x, feature_y]].values
    y2 = df["default"].values
    # Slightly higher C → tighter margin → cleaner contour for the visual.
    pipe = make_pipeline(kernel="rbf", C=10.0, gamma="scale")
    pipe.fit(X2, y2)
    svm = pipe.named_steps["svm"]

    # 5% padding around the data range so the boundary doesn't clip points.
    pad_x = 0.05 * (X2[:, 0].max() - X2[:, 0].min())
    pad_y = 0.05 * (X2[:, 1].max() - X2[:, 1].min())
    # Dense 250×250 grid over the (x, y) plane.
    xx, yy = np.meshgrid(
        np.linspace(X2[:, 0].min() - pad_x, X2[:, 0].max() + pad_x, 250),
        np.linspace(X2[:, 1].min() - pad_y, X2[:, 1].max() + pad_y, 250),
    )
    # Score every grid point; reshape back to the meshgrid shape.
    Z = pipe.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # ---- Draw: filled regions for sign(f), contour lines for f = -1, 0, +1 ----
    fig, ax = plt.subplots(figsize=(8, 6))
    # Two colour regions: blue for f<0 (predicted repaid), red for f>0.
    ax.contourf(xx, yy, Z, levels=[-1e3, 0, 1e3],
                colors=["#cfe2f3", "#f4cccc"], alpha=0.55)
    # Solid line at f=0 (decision boundary), dashed at f=±1 (margin).
    cs = ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                    colors="black", linestyles=["--", "-", "--"])
    ax.clabel(cs, fmt={-1: "f=-1", 0: "f=0", 1: "f=+1"}, fontsize=8)

    # Random subsample for visual clarity (full point cloud is too dense).
    rng = np.random.default_rng(RANDOM_STATE)
    idx_repay = rng.choice(np.where(y2 == 0)[0], size=400, replace=False)
    idx_default = rng.choice(np.where(y2 == 1)[0],
                             size=min(400, int((y2 == 1).sum())), replace=False)
    ax.scatter(X2[idx_repay, 0], X2[idx_repay, 1], s=8, c="steelblue",
               alpha=0.7, label="repaid")
    ax.scatter(X2[idx_default, 0], X2[idx_default, 1], s=8, c="indianred",
               alpha=0.7, label="default")
    # Highlight the actual support vectors with hollow circles.
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
    # Build the continuous-target dataset and split into train/test.
    df = make_spread_dataset()
    feat = [c for c in df.columns if c != "spread"]
    X, y = df[feat].values, df["spread"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    # ---- ε-SVR with a fixed insensitive-tube half-width ----
    eps_value = 0.01     # 1% tube — predictions within ±1% incur no loss
    eps_pipe = Pipeline([("scaler", StandardScaler()),
                         ("svr", SVR(kernel="rbf", C=1.0, gamma="scale",
                                     epsilon=eps_value))])
    # ---- ν-SVR with ν=0.5: at most 50% of points outside the tube, ----
    # ---- at least ~50% used as support vectors                       ----
    nu_pipe = Pipeline([("scaler", StandardScaler()),
                        ("svr", NuSVR(kernel="rbf", C=1.0, gamma="scale",
                                      nu=0.5))])
    eps_pipe.fit(X_train, y_train)
    nu_pipe.fit(X_train, y_train)

    # Empirical ε for NuSVR: median residual at the FREE support vectors,
    # i.e. SVs with |α_i − α_i*| < C — these lie exactly on the ε-tube edge.
    nu_svr = nu_pipe.named_steps["svr"]
    # Indices (in training set) of the support vectors.
    sv_idx = nu_svr.support_
    if sv_idx.size:
        # Absolute residuals only at the SV rows.
        sv_resid = np.abs(y_train[sv_idx] - nu_pipe.predict(X_train[sv_idx]))
        # "Free" SVs: their dual coef has strict |α| < C, so they sit ON the tube.
        free_mask = np.abs(nu_svr.dual_coef_.ravel()) < nu_svr.C - 1e-6
        # Median free-SV residual ≈ empirical ε.
        nu_eps_emp = (float(np.median(sv_resid[free_mask]))
                      if free_mask.any() else float("nan"))
    else:
        nu_eps_emp = float("nan")

    # ---- Tabulate the two models side by side ----
    print(f"{'model':16s} {'MAE':>8s} {'R²':>7s} {'n_SV':>8s} {'%SV':>7s} {'ε (used)':>12s}")
    for name, p, eps_used in [
        ("ε-SVR (ε=0.01)", eps_pipe, eps_value),
        ("ν-SVR (ν=0.5)",  nu_pipe,  nu_eps_emp),
    ]:
        # Test-set predictions and metrics.
        y_hat = p.predict(X_test)
        mae = mean_absolute_error(y_test, y_hat)
        r2  = r2_score(y_test, y_hat)
        n_sv = p.named_steps["svr"].support_.size
        # Format ε; show "—" if undefined.
        eps_str = f"{eps_used:.4f}" if not np.isnan(eps_used) else "—"
        print(f"{name:16s} {mae:8.4f} {r2:7.4f} {n_sv:8d} "
              f"{100*n_sv/len(X_train):6.1f}% {eps_str:>12s}")
    print("ν is an UPPER bound on the fraction of margin errors and a LOWER "
          "bound on the SV fraction (Schölkopf et al., 2000). With noisy data "
          "the SV fraction can exceed ν substantially.")

    # ---- Plot predicted-vs-actual with the ε-tube shaded ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, p, name, eps in [
        (axes[0], eps_pipe, f"ε-SVR  (fixed ε = {eps_value})", eps_value),
        (axes[1], nu_pipe,
         f"ν-SVR  (ν = 0.5, empirical ε ≈ {nu_eps_emp:.3f})", nu_eps_emp),
    ]:
        y_hat = p.predict(X_test)
        # Scatter predicted vs actual spread.
        ax.scatter(y_test, y_hat, s=10, alpha=0.6, color="steelblue",
                   label="test predictions")
        # Diagonal y=ŷ reference line.
        lo = min(y_test.min(), y_hat.min())
        hi = max(y_test.max(), y_hat.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ŷ = y")
        # Shade ±ε band — borrowers inside this band don't contribute to f(x).
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
    # Create the output folder for PNGs (no error if it already exists).
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ---- PART I: load the classification dataset ----
    df = make_loan_dataset()
    print(f"Dataset      : {len(df)} borrowers")
    print(f"Default rate : {df['default'].mean():.2%}")

    # Split into feature matrix X (all columns but `default`) and target y.
    feature_cols = [c for c in df.columns if c != "default"]
    X = df[feature_cols].values
    y = df["default"].values

    # Stratified train/test split → preserves the ~18% default rate in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # ---- PART III: kernel head-to-head, then tune RBF (C, γ) ----
    kernel_comparison(X_train, y_train, X_test, y_test)
    best_model, best_params = cgamma_heatmap(X_train, y_train)

    # ---- Test-set evaluation of the tuned RBF SVM ----
    section("Tuned RBF SVM — test-set evaluation")
    # Hard predictions and raw margin scores on the held-out test set.
    y_pred  = best_model.predict(X_test)
    y_score = best_model.decision_function(X_test)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, y_score):.4f}")
    print(f"PR  AP   : {average_precision_score(y_test, y_score):.4f}")
    # Confusion matrix wrapped in a labelled DataFrame for readability.
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["actual_repaid", "actual_default"],
        columns=["pred_repaid", "pred_default"],
    ))
    # Per-class precision / recall / F1.
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["repaid", "default"]))

    # ---- Logistic-regression baseline ----
    # If a linear logistic match the RBF SVM's AUC, the kernel isn't earning
    # its keep on this dataset.
    lr = Pipeline([("scaler", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=2000,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE))])
    lr.fit(X_train, y_train)
    # `predict_proba(...)[:, 1]` = probability of the positive class.
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    print(f"\nBaseline logistic-regression AUC : {lr_auc:.4f}    "
          f"(RBF SVM AUC: {roc_auc_score(y_test, y_score):.4f})")

    # ---- PART II demos ----
    support_vector_analysis(best_model, X_train)        # inspect dual_coef_
    manual_decision_function(best_model, X_test)        # rebuild f(x) by hand
    convexity_demo(X_train, y_train)                    # unique global min

    # ---- PART III demos ----
    kernel_trick_demo()                                 # Φ(x) and <x,x'>²

    # ---- Applied layer (calibration, importance, threshold, learning, boundary) ----
    diagnostics(best_model, X_train, y_train, X_test, y_test)
    permutation_importances(best_model, X_test, y_test, feature_cols)
    threshold_tuning(best_model, X_test, y_test, cost_fn=5.0, cost_fp=1.0)
    learning_curve_plot(best_model, X, y)
    decision_boundary_2d(df)

    # ---- SVR demo (ε-SVR vs ν-SVR on credit spreads) ----
    svr_demo()

    # ---- Score a brand-new borrower (inference example) ----
    section("Scoring a brand-new borrower")
    # A single-row DataFrame: a realistically risky applicant profile.
    new_borrower = pd.DataFrame([{
        "debt_to_income":     0.55,             # 55% of income → debt service
        "credit_utilization": 0.78,             # 78% of revolving limit used
        "missed_payments":    2,                # 2 missed payments in 12 mo
        "savings_ratio":      0.05,             # only 5% saved
        "income_volatility":  0.30,             # gig-worker style swings
        "loan_to_income":     0.9,              # loan = 90% of annual income
        "credit_age_years":   3.0,              # short credit history
    }])
    # Raw signed margin under the tuned model.
    margin = best_model.decision_function(new_borrower[feature_cols].values)[0]
    # Decision rule: positive margin → predicted default.
    decision = "DEFAULT" if margin >= 0 else "REPAY"
    print(f"New borrower profile:\n{new_borrower.T}")
    print(f"\nRaw SVM margin score : {margin:+.3f}")
    print(f"Predicted class      : {decision}")
    print("Positive margin → the borrower lies on the 'default' side of f(x)=0.")
    print(f"All plots written to ./{PLOT_DIR}/")


# Standard Python idiom: only run `main()` when this file is executed directly,
# not when it's imported (e.g. by a test suite).
if __name__ == "__main__":
    main()
