"""
ε-SVR for illiquid CDS spread prediction — companion to Halperin's
"SVM in Finance" slide deck "Part IV: SVM for prediction of credit spreads".

Problem statement (verbatim from the slide):
  Task   : Estimate unobserved CDS spreads for companies with illiquid debt,
           based on observed spreads and other characteristics of OTHER,
           liquid companies.
  Inputs : X_i = (rating, industry sector, geographic sector, EDF).
  Output : Y_i = 5-year CDS spread (basis points) for company i.
  Sample : {(X_i, Y_i)}_{i=1..N} for N ≈ 500 liquid US-market CDS.
  Models : Neural net or SVM (we use SVM, specifically ε-SVR).
  Usage  : Counterparty credit risk management, capital calculation.

This script:
  1. Synthesises 500 "liquid" and 150 "illiquid" US-market companies with
     known features and true CDS spreads.
  2. One-hot encodes the three categorical features (rating, industry,
     region) and log-transforms EDF; predicts log(spread) so the model
     covers the 25–1500 bps dynamic range.
  3. Tunes (C, γ, ε) via 5-fold CV grid search and trains the winning
     ε-SVR on the full liquid universe.
  4. Evaluates on the illiquid pool — overall MAE/R²/MAPE plus per-rating
     breakdown.
  5. Reproduces the slide's index-tracking plot by simulating ~60 days of
     drift in EDF and a residual market shock, then aggregating SVM-implied
     vs Actual vs MLP-implied Investment-Grade (IG) and High-Yield (HY)
     index levels.

Run:
    python svm_cds_spreads.py

Outputs to ./svm_cds_plots/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, StandardScaler,
)
from sklearn.svm import SVR


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
PLOT_DIR     = "svm_cds_plots"
RANDOM_STATE = 42
N_LIQUID     = 500
N_ILLIQUID   = 150
N_DAYS       = 60     # simulated time-series window for the index plot


# Rating universe and prior — roughly the IG/HY composition of CDX/iTraxx.
RATINGS      = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_PROBS = [0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05]

# Base 5Y CDS spread (bps) by rating. Calibrated so the IG-weighted average
# lands ≈130 bps and HY ≈600 bps — matching the slide's index ranges.
BASE_BY_RATING = {
    "AAA":   25,
    "AA":    50,
    "A":    100,
    "BBB":  200,
    "BB":   400,
    "B":    700,
    "CCC": 1200,
}

# Industry adjustments (bps). Cyclical / commodity sectors pay more; staples
# and utilities pay less. Magnitudes are deliberately small so the rating
# dominates — as in the real market.
INDUSTRIES   = ["Energy", "Financials", "Healthcare", "Industrials",
                "Materials", "Technology", "ConsumerStaples",
                "ConsumerDiscretionary", "Utilities", "Telecom"]
INDUSTRY_ADJ = {
    "Energy":               +30,
    "Financials":           +20,
    "Healthcare":            -5,
    "Industrials":          +10,
    "Materials":            +25,
    "Technology":             0,
    "ConsumerStaples":      -15,
    "ConsumerDiscretionary":+15,
    "Utilities":            -25,
    "Telecom":              +10,
}

REGIONS    = ["Northeast", "South", "Midwest", "West"]
REGION_ADJ = {"Northeast": -5, "South": +5, "Midwest": 0, "West": -3}

# Base log10(EDF) location by rating — lower-rated companies have higher PD.
EDF_LOC_BY_RATING = {
    "AAA": -7, "AA": -6, "A": -5, "BBB": -4, "BB": -3, "B": -2, "CCC": -1,
}


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
os.makedirs(PLOT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)


# ============================================================================
# 1. Synthesise the US CDS universe
# ============================================================================
def generate_companies(n: int, seed_offset: int = 0) -> pd.DataFrame:
    """Generate a population of companies with their static features."""
    local_rng = np.random.default_rng(RANDOM_STATE + seed_offset)
    ratings    = local_rng.choice(RATINGS, size=n, p=RATING_PROBS)
    industries = local_rng.choice(INDUSTRIES, size=n)
    regions    = local_rng.choice(REGIONS, size=n)
    # EDF (Expected Default Frequency): log-normally distributed around a
    # rating-specific location. Realistic PDs in [1e-4, 0.3].
    edf_loc = np.array([EDF_LOC_BY_RATING[r] for r in ratings])
    edf     = np.exp(edf_loc + local_rng.normal(0, 0.6, size=n))
    return pd.DataFrame({
        "rating":   ratings,
        "industry": industries,
        "region":   regions,
        "edf":      edf,
    })


def true_spread(df: pd.DataFrame,
                market_shift_bps: float = 0.0,
                noise_sd: float = 15.0,
                rng_local: np.random.Generator | None = None) -> np.ndarray:
    """Return the true CDS spread (bps) for each company.

    True spread = base(rating) + industry_adj + region_adj
                + 80·log10(EDF·1e6) − 100   (nonlinear EDF kicker)
                + market_shift_bps          (system-wide credit cycle)
                + Gaussian noise.
    Clipped to [5, 3000] bps.
    """
    r = rng_local if rng_local is not None else rng
    base = np.array([BASE_BY_RATING[x] for x in df["rating"]])
    indu = np.array([INDUSTRY_ADJ[x]   for x in df["industry"]])
    reg  = np.array([REGION_ADJ[x]     for x in df["region"]])
    edf_term = 80 * np.log10(df["edf"].values * 1e6) - 100
    noise = r.normal(0, noise_sd, size=len(df))
    return np.clip(base + indu + reg + edf_term + market_shift_bps + noise,
                   5, 3000)


# Generate the day-0 universe.
liquid_df   = generate_companies(N_LIQUID,   seed_offset=0)
illiquid_df = generate_companies(N_ILLIQUID, seed_offset=1)
liquid_df["spread"]   = true_spread(liquid_df)
illiquid_df["spread"] = true_spread(illiquid_df)

print(f"Liquid universe   : {len(liquid_df):>3d} companies, "
      f"avg spread = {liquid_df['spread'].mean():.0f} bps   "
      f"range [{liquid_df['spread'].min():.0f}, {liquid_df['spread'].max():.0f}]")
print(f"Illiquid pool     : {len(illiquid_df):>3d} companies, "
      f"avg spread = {illiquid_df['spread'].mean():.0f} bps   "
      f"range [{illiquid_df['spread'].min():.0f}, {illiquid_df['spread'].max():.0f}]")

FEATURE_COLS = ["rating", "industry", "region", "edf"]


# ============================================================================
# 2. Preprocessing pipeline
# ============================================================================
def make_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals; log-transform + standardise EDF."""
    return ColumnTransformer([
        ("cats", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                  ["rating", "industry", "region"]),
        ("edf", Pipeline([
            ("log", FunctionTransformer(np.log10)),
            ("std", StandardScaler()),
        ]), ["edf"]),
    ])


def make_svr() -> Pipeline:
    """ε-SVR with RBF kernel on the preprocessed features."""
    return Pipeline([
        ("pre", make_preprocessor()),
        ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale")),
    ])


def make_mlp() -> Pipeline:
    """Small MLP baseline (one hidden layer)."""
    return Pipeline([
        ("pre", make_preprocessor()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(16,), max_iter=2000,
                             learning_rate_init=0.01, alpha=1e-3,
                             random_state=RANDOM_STATE)),
    ])


# ============================================================================
# 3. Hyperparameter tuning (5-fold CV grid search) — predict log10(spread)
# ============================================================================
print("\nTuning (C, γ, ε) for ε-SVR …")
y_liquid_log = np.log10(liquid_df["spread"].values)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(
    make_svr(),
    param_grid={
        "svr__C":       [1.0, 10.0, 100.0],
        "svr__gamma":   ["scale", 0.05, 0.5],
        "svr__epsilon": [0.02, 0.05, 0.1],
    },
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
grid.fit(liquid_df[FEATURE_COLS], y_liquid_log)
print(f"  best params   : {grid.best_params_}")
print(f"  best CV MAE   : {-grid.best_score_:.4f}  (log10 bps)")
svr_model = grid.best_estimator_


# ============================================================================
# 4. Evaluate on the illiquid pool
# ============================================================================
y_ill_true = illiquid_df["spread"].values
y_ill_pred_log = svr_model.predict(illiquid_df[FEATURE_COLS])
y_ill_pred     = 10 ** y_ill_pred_log

overall_mae  = mean_absolute_error(y_ill_true, y_ill_pred)
overall_r2   = r2_score(y_ill_true, y_ill_pred)
overall_mape = np.mean(np.abs(y_ill_pred - y_ill_true) / y_ill_true) * 100

print(f"\nIlliquid pool predictions (n={len(illiquid_df)}):")
print(f"  MAE  : {overall_mae:7.1f} bps")
print(f"  R²   : {overall_r2:7.3f}")
print(f"  MAPE : {overall_mape:7.1f} %")

# Per-rating breakdown
print(f"\n  rating  count  avg actual  avg pred   MAE (bps)")
per_rating = []
for r in RATINGS:
    mask = illiquid_df["rating"].values == r
    if mask.sum() == 0:
        continue
    a = y_ill_true[mask]
    p = y_ill_pred[mask]
    mae_r = mean_absolute_error(a, p)
    print(f"  {r:>5s}   {mask.sum():>4d}    {a.mean():7.0f}    "
          f"{p.mean():7.0f}    {mae_r:7.1f}")
    per_rating.append((r, mask.sum(), a.mean(), p.mean(), mae_r))


# ----------------------------------------------------------------------------
# Plot 1 — predicted vs actual scatter, log-log for the dynamic range
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
colors = {"IG": "#4c72b0", "HY": "#c44e52"}
ig_mask_ill = illiquid_df["rating"].isin(["AAA", "AA", "A", "BBB"]).values
ax.scatter(y_ill_true[ig_mask_ill],  y_ill_pred[ig_mask_ill],
           s=20, c=colors["IG"], alpha=0.7, label="IG  (AAA … BBB)")
ax.scatter(y_ill_true[~ig_mask_ill], y_ill_pred[~ig_mask_ill],
           s=20, c=colors["HY"], alpha=0.7, label="HY  (BB … CCC)")
lo, hi = 10, 3000
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal $\\hat y = y$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
ax.set_xlabel("Actual 5Y CDS spread (bps)")
ax.set_ylabel("SVR-predicted 5Y CDS spread (bps)")
ax.set_title(f"Illiquid pool — actual vs predicted\n"
             f"MAE = {overall_mae:.1f} bps   $R^2$ = {overall_r2:.3f}")
ax.legend(loc="upper left"); ax.grid(alpha=0.2, which="both")
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/01_pred_vs_actual.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# Plot 2 — per-rating MAE bar chart
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
rating_labels = [row[0] for row in per_rating]
maes          = [row[4] for row in per_rating]
counts        = [row[1] for row in per_rating]
bars = ax.bar(rating_labels, maes,
              color=["#4c72b0" if r in ("AAA","AA","A","BBB") else "#c44e52"
                     for r in rating_labels])
for bar, mae, n in zip(bars, maes, counts):
    ax.text(bar.get_x() + bar.get_width()/2,
            mae + max(maes) * 0.02,
            f"n={n}\n{mae:.0f}", ha="center", fontsize=9)
ax.set_ylabel("MAE (bps)")
ax.set_title("Per-rating prediction error on the illiquid pool")
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/02_per_rating_mae.png", dpi=120)
plt.close()


# ============================================================================
# 5. Time-series simulation — reproduce the slide's index plot
# ============================================================================
print("\nSimulating index time series …")


def simulate_market_path(n_days: int):
    """Return (edf_factor, market_shift) over n_days.

    edf_factor[t]    multiplicative drift on every borrower's EDF.
    market_shift[t]  additive bps shift that the SVR does NOT see.
    """
    local_rng = np.random.default_rng(RANDOM_STATE + 99)
    # EDF: low-frequency log-Brownian motion ±25%.
    edf_drift = np.cumsum(local_rng.normal(0, 0.025, n_days))
    edf_factor = np.exp(edf_drift - edf_drift[0])
    # Residual market shock: smooth random walk + a "shock then recovery" hump
    # so the IG index rises then falls (mimicking 2011-Q4 European debt crisis).
    market_shift = np.cumsum(local_rng.normal(0, 1.5, n_days))
    hump = 12 * np.exp(-((np.arange(n_days) - n_days/3) / 8) ** 2)
    market_shift = market_shift - market_shift[0] + hump
    return edf_factor, market_shift


edf_factor, market_shift = simulate_market_path(N_DAYS)

# Train MLP baseline once on day-0 liquid universe.
print("  fitting MLP baseline …")
mlp_model = make_mlp().fit(liquid_df[FEATURE_COLS], y_liquid_log)

# IG = AAA…BBB, HY = BB…CCC. Compute the index on the liquid universe
# (representative basket of generic names per the slide).
ig_mask = liquid_df["rating"].isin(["AAA", "AA", "A", "BBB"]).values
hy_mask = ~ig_mask

ig_actual = np.zeros(N_DAYS); ig_svm = np.zeros(N_DAYS); ig_mlp = np.zeros(N_DAYS)
hy_actual = np.zeros(N_DAYS); hy_svm = np.zeros(N_DAYS); hy_mlp = np.zeros(N_DAYS)

for t in range(N_DAYS):
    df_t = liquid_df.copy()
    df_t["edf"] = df_t["edf"] * edf_factor[t]
    # True spreads on day t (with market shift in the noise term).
    s_actual = true_spread(df_t, market_shift_bps=market_shift[t],
                           noise_sd=8.0,
                           rng_local=np.random.default_rng(t))
    # SVR / MLP predictions — only use the features they were trained on.
    X_t = df_t[FEATURE_COLS]
    s_svm = 10 ** svr_model.predict(X_t)
    s_mlp = 10 ** mlp_model.predict(X_t)
    ig_actual[t] = s_actual[ig_mask].mean()
    ig_svm[t]    = s_svm[ig_mask].mean()
    ig_mlp[t]    = s_mlp[ig_mask].mean()
    hy_actual[t] = s_actual[hy_mask].mean()
    hy_svm[t]    = s_svm[hy_mask].mean()
    hy_mlp[t]    = s_mlp[hy_mask].mean()

print(f"  IG: actual avg {ig_actual.mean():.0f} bps   "
      f"SVM tracking error MAE = {np.abs(ig_actual - ig_svm).mean():.1f} bps   "
      f"MLP tracking error MAE = {np.abs(ig_actual - ig_mlp).mean():.1f} bps")
print(f"  HY: actual avg {hy_actual.mean():.0f} bps   "
      f"SVM tracking error MAE = {np.abs(hy_actual - hy_svm).mean():.1f} bps   "
      f"MLP tracking error MAE = {np.abs(hy_actual - hy_mlp).mean():.1f} bps")


# ----------------------------------------------------------------------------
# Plot 3 — IG and HY index reproductions (mimics the slide's two side panels)
# ----------------------------------------------------------------------------
days = np.arange(N_DAYS)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(days, ig_actual, color="#1f77b4", marker="o", ms=4, lw=1.5,
        label="Actual")
ax.plot(days, ig_svm,    color="#c44e52", marker="s", ms=4, lw=1.0,
        label="SVM Implied")
ax.scatter(days[::3], ig_mlp[::3], color="#55a868", marker="^", s=35,
           label="ANN Implied")
ax.set_xlabel("Trading day")
ax.set_ylabel("5Y CDS spread (bps)")
ax.set_title(f"IG Index (basket of AAA…BBB names)\n"
             f"SVM tracking MAE = {np.abs(ig_actual - ig_svm).mean():.1f} bps")
ax.legend(loc="upper right"); ax.grid(alpha=0.2)

ax = axes[1]
ax.plot(days, hy_actual, color="#1f77b4", marker="o", ms=4, lw=1.5,
        label="Actual")
ax.plot(days, hy_svm,    color="#c44e52", marker="s", ms=4, lw=1.0,
        label="SVM Implied")
ax.scatter(days[::3], hy_mlp[::3], color="#55a868", marker="^", s=35,
           label="ANN Implied")
ax.set_xlabel("Trading day")
ax.set_ylabel("5Y CDS spread (bps)")
ax.set_title(f"HY Index (basket of BB…CCC names)\n"
             f"SVM tracking MAE = {np.abs(hy_actual - hy_svm).mean():.1f} bps")
ax.legend(loc="upper right"); ax.grid(alpha=0.2)

plt.suptitle("Theoretical index spread mimicking CDX / iTraxx — "
             "SVR vs Actual vs MLP", y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_index_timeseries.png", dpi=120)
plt.close()


# ============================================================================
# 6. Predict a brand-new illiquid borrower
# ============================================================================
print("\nScoring a new illiquid borrower …")
new_company = pd.DataFrame([{
    "rating":   "BB",
    "industry": "Energy",
    "region":   "South",
    "edf":      0.03,         # 3% expected default frequency
}])
pred_spread = float(10 ** svr_model.predict(new_company)[0])
print(f"  features  : BB-rated Energy company in the South, EDF = 3%")
print(f"  predicted : {pred_spread:.0f} bps  "
      f"(annual premium on a $10m notional = ${pred_spread * 1000:.0f}/yr)")

print(f"\nAll plots written to ./{PLOT_DIR}/")
