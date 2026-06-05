# Lecture 02 — Code Structure

Flow-chart view of [`svm_loan_default.py`](svm_loan_default.py).
Diagrams use [Mermaid](https://mermaid.js.org/) — GitHub renders them inline.

The script is organised in three layers, mirroring Halperin's *SVM in
Finance* slides (NYU Tandon, 2017):

* **Part I** — primal SVR / SVC, ε-insensitive loss, soft-margin `C`, scaling.
* **Part II** — KKT conditions, the dual, the support-vector expansion,
  convex optimisation.
* **Part III** — the kernel trick: features `Φ(x)` replaced by a kernel `k(x, x')`.

---

## 1. Top-level execution flow (`main()`)

The orchestration of every analysis step, in the order `main()` runs them.

```mermaid
flowchart TD
    Start([python svm_loan_default.py]) --> Mkdir[os.makedirs svm_plots/]
    Mkdir --> Data["make_loan_dataset — 3000 borrowers, default ~18%"]
    Data --> Split["train_test_split — stratified 75/25"]

    Split --> Kernels["kernel_comparison — 5 kernels<br/>→ kernel_comparison.png"]
    Kernels --> Grid["cgamma_heatmap — 5×5 grid, 5-fold CV<br/>→ hyperparameter_heatmap.png"]
    Grid --> Eval["Tuned RBF eval — accuracy, AUC, AP, CM, report"]
    Eval --> LR["LogisticRegression baseline — sanity-check vs kernel SVM"]

    LR --> SV["support_vector_analysis — dual_coef_, free vs bounded SVs"]
    SV --> Manual["manual_decision_function — rebuild f(x) by hand"]
    Manual --> Convex["convexity_demo — 5 seeds → identical α, b"]

    Convex --> Trick["kernel_trick_demo — verify ⟨Φ, Φ'⟩ = ⟨x, x'⟩²"]

    Trick --> Diag["diagnostics — margin hist + calibration + ROC + PR<br/>→ diagnostics.png"]
    Diag --> Perm["permutation_importances<br/>→ permutation_importance.png"]
    Perm --> Thr["threshold_tuning — FN cost = 5, FP cost = 1<br/>→ threshold_cost.png"]
    Thr --> LC["learning_curve_plot<br/>→ learning_curve.png"]
    LC --> Bdy["decision_boundary_2d — RBF contour with SVs<br/>→ decision_boundary_2d.png"]

    Bdy --> SVR["svr_demo — ε-SVR + ν-SVR on credit spreads<br/>→ svr_demo.png"]
    SVR --> New["Score a brand-new borrower"]
    New --> End([8 plots written to svm_plots/])
```

---

## 2. Slide concept → function map

Which slide concept is demonstrated in which function.

```mermaid
flowchart LR
    subgraph PartI ["Part I — Primal SVR / SVC"]
        I1["ε-insensitive loss + ξ, ξ* slack"]
        I2["Soft-margin C trade-off"]
        I3["Feature scaling rationale"]
    end
    subgraph PartII ["Part II — KKT and the dual"]
        II1["Dual multipliers α_i, α_i*"]
        II2["SV expansion f(x) = Σ α_i y_i k + b"]
        II3["Convex QP → unique minimum"]
        II4["KKT: α = 0 inside the tube"]
    end
    subgraph PartIII ["Part III — Kernel trick"]
        III1["k(x, x') = ⟨Φ(x), Φ(x')⟩"]
        III2["4 Mercer kernels"]
        III3["γ hyperparameter"]
    end

    I1 --> svr_demo
    I2 --> make_pipeline
    I3 --> make_pipeline

    II1 --> support_vector_analysis
    II2 --> manual_decision_function
    II3 --> convexity_demo
    II4 --> support_vector_analysis

    III1 --> kernel_trick_demo
    III2 --> kernel_comparison
    III3 --> cgamma_heatmap
```

---

## 3. Dataset generation (`make_loan_dataset`)

Each borrower's default label comes from a *known* logit model so we can
diagnose the SVM against ground truth.

```mermaid
flowchart LR
    rng["np.random.default_rng(42)"] --> beta1["debt_to_income — Beta(2,5) × 1.2"]
    rng --> beta2["credit_utilization — Beta(2,4)"]
    rng --> pois["missed_payments — Poisson(0.6)"]
    rng --> beta3["savings_ratio — Beta(2,5) × 0.5"]
    rng --> norm1["income_volatility — abs Normal(0.15, 0.1)"]
    rng --> gam["loan_to_income — Gamma(2.0, 0.3)"]
    rng --> norm2["credit_age_years — clip Normal(8, 5)"]

    beta1 --> logit
    beta2 --> logit
    pois --> logit
    beta3 --> logit
    norm1 --> logit
    gam --> logit
    norm2 --> logit
    logit["logit = -4.0 + 2.8·DTI + 2.0·CU + 0.9·MP<br/>     − 3.0·SR + 2.5·IV + 1.2·LTI − 0.05·CAY"]
    logit --> sigmoid["p_default = σ(logit)"]
    sigmoid --> bern["Bernoulli draw: default = 1 if U &lt; p_default"]
    bern --> df["pd.DataFrame — 8 columns"]
```

---

## 4. Kernel comparison (`kernel_comparison`)

```mermaid
flowchart TD
    Kernels["kernels = [linear, poly-d2, poly-d3, rbf, sigmoid]"] --> Loop{for name, params in kernels}
    Loop --> Pipe["make_pipeline(**params)"]
    Pipe --> Fit["pipe.fit(X_train, y_train)"]
    Fit --> Score["pipe.decision_function(X_test)"]
    Score --> Metrics["roc_auc_score + n_SV"]
    Metrics --> Append["rows.append(...)"]
    Append --> Loop
    Loop -->|done| Table["pd.DataFrame → print"]
    Table --> Plot["2-panel bar chart: AUC and %SV"]
    Plot --> Out[/kernel_comparison.png/]
```

---

## 5. (C, γ) grid search (`cgamma_heatmap`)

```mermaid
flowchart TD
    Pipe["Pipeline(scaler, SVC(kernel=rbf))"] --> Grid["GridSearchCV<br/>C ∈ 5 vals, γ ∈ 5 vals"]
    Grid --> CV["StratifiedKFold(k=5)"]
    CV --> Loop["25 settings × 5 folds = 125 fits<br/>n_jobs = -1 parallel"]
    Loop --> Pick["Best by mean ROC-AUC"]
    Pick --> Pivot["Pivot cv_results_ → γ × C matrix"]
    Pivot --> Heat["plt.imshow + annotated cells"]
    Heat --> Out[/hyperparameter_heatmap.png/]
    Pick --> Return["return best_estimator_, best_params_"]
```

---

## 6. Manual SV expansion (`manual_decision_function`)

Reconstructs the slide formula `f(x) = Σᵢ (αᵢ yᵢ) k(xᵢ, x) + b` from
sklearn's stored primitives and checks it matches `decision_function`.

```mermaid
flowchart TD
    Test["X_test (first 5 rows)"] --> Scale["scaler.transform → Xs"]

    Model["Trained SVC"] --> SV["support_vectors_<br/>scaled-space SVs, shape (n_SV, d)"]
    Model --> Dual["dual_coef_[0]<br/>α_i × y_i, shape (n_SV,)"]
    Model --> Bias["intercept_[0] = b"]
    Model --> Gam["_gamma — resolved float"]

    Scale --> Diff["Pairwise difference SV minus Xs<br/>broadcast to shape n_test, n_SV, d"]
    Diff --> SqDist["sum diff squared along last axis"]
    SqDist --> Kernel["K = exp(−γ × sq_dist)"]
    Gam --> Kernel

    Kernel --> Mat["f_manual = K @ dual_coef + b"]
    Dual --> Mat
    Bias --> Mat

    Test --> Sklearn["model.decision_function on 5 rows"]
    Mat --> Compare{within 1e-10 ?}
    Sklearn --> Compare
    Compare -->|yes| OK["SV expansion verified"]
```

---

## 7. Kernel-trick demo (`kernel_trick_demo`)

Numerically verifies `⟨Φ(x), Φ(x')⟩ = ⟨x, x'⟩²` for the slide's
quadratic feature map `Φ(x) = (x₁², √2 x₁ x₂, x₂²)`.

```mermaid
flowchart LR
    rng["np.random.default_rng(0)"] --> x["x ∈ R²"]
    rng --> xp["x' ∈ R²"]

    x --> Phi1["Φ(x) = (x₁², √2 x₁x₂, x₂²)"]
    xp --> Phi2["Φ(x') analogous"]
    Phi1 --> LHS["⟨Φ(x), Φ(x')⟩ — R³ dot"]
    Phi2 --> LHS

    x --> Dot["⟨x, x'⟩"]
    xp --> Dot
    Dot --> RHS["⟨x, x'⟩²"]

    LHS --> Cmp{equal?}
    RHS --> Cmp
    Cmp -->|yes| Trick["Kernel trick verified ✓"]

    x --> Poly["sklearn polynomial_kernel<br/>(d=2, γ=1, r=0)"]
    xp --> Poly
    Poly --> Same["same numeric value"]

    x --> Rbf["sklearn rbf_kernel(γ=1)"]
    xp --> Rbf
    Rbf --> Inf["infinite-dim Φ — value still computable"]
```

---

## 8. Convexity demo (`convexity_demo`)

```mermaid
flowchart TD
    Seeds["seeds = [0, 1, 7, 42, 999]"] --> Loop{for seed in seeds}
    Loop --> Pipe["make_pipeline(kernel=rbf, C=1, γ=scale, random_state=seed)"]
    Pipe --> Fit["pipe.fit(X_train, y_train)"]
    Fit --> Sig["sig = (n_SV, b, Σ α_i y_i)"]
    Sig --> First{first seed?}
    First -->|yes| Save["ref = sig"]
    First -->|no| Check{sig == ref ?}
    Check -->|yes| Print["print 'match'"]
    Check -->|no| Print2["print 'DIFFER'"]
    Save --> Loop
    Print --> Loop
    Print2 --> Loop
    Loop -->|done| Done["All 5 match → convex QP has unique optimum ✓"]
```

---

## 9. Diagnostics (`diagnostics`)

```mermaid
flowchart TD
    Best["Tuned RBF SVC"] --> RawScore["decision_function(X_test)"]
    Best --> Params["Clone hyperparameters"]

    Params --> Platt["SVC(probability=True)<br/>internal CV + sigmoid"]
    Platt --> ProbaP["predict_proba — Platt"]

    Params --> Iso["CalibratedClassifierCV(method=isotonic, cv=5)"]
    Iso --> ProbaI["predict_proba — isotonic"]

    RawScore --> H1["Histogram of margin by class"]
    ProbaP --> CalP["calibration_curve (Platt)"]
    ProbaI --> CalI["calibration_curve (isotonic)"]
    RawScore --> ROC["roc_curve + AUC"]
    RawScore --> PR["precision_recall_curve + AP"]

    H1 --> Out[/diagnostics.png — 4-panel/]
    CalP --> Out
    CalI --> Out
    ROC --> Out
    PR --> Out
```

---

## 10. SVR demo (`svr_demo`)

```mermaid
flowchart TD
    Data["make_spread_dataset(1500)"] --> Split["train_test_split 75/25"]

    Split --> EpsPipe["ε-SVR pipeline<br/>scaler + SVR(ε=0.01)"]
    Split --> NuPipe["ν-SVR pipeline<br/>scaler + NuSVR(ν=0.5)"]

    EpsPipe --> EpsFit["fit(X_train, y_train)"]
    NuPipe --> NuFit["fit(X_train, y_train)"]

    NuFit --> NuEps["Empirical ε for NuSVR:<br/>median absolute residual at FREE SVs<br/>where dual_coef magnitude is below C"]

    EpsFit --> Eval["For each model report:<br/>MAE, R², n_SV, %SV, ε used"]
    NuFit --> Eval
    NuEps --> Eval

    Eval --> Plot["Predicted vs actual scatter<br/>with ±ε tube shaded"]
    Plot --> Out[/svr_demo.png/]
```

---

## 11. Output artifacts

```
lecture_02/svm_plots/
├── kernel_comparison.png       — AUC + %SV bar charts per kernel
├── hyperparameter_heatmap.png  — 5×5 CV-AUC heatmap over (C, γ)
├── diagnostics.png             — 4-panel: margin hist, calibration, ROC, PR
├── permutation_importance.png  — horizontal bars of AUC drop per feature
├── threshold_cost.png          — expected loss vs threshold
├── learning_curve.png          — train/val AUC vs training size
├── decision_boundary_2d.png    — RBF contour on (DTI, credit_util)
└── svr_demo.png                — predicted vs actual + ε-tube (ε-SVR & ν-SVR)
```

---

Function names in the diagrams match exactly the function definitions in
[`svm_loan_default.py`](svm_loan_default.py) — jump to any function to
read its implementation.
