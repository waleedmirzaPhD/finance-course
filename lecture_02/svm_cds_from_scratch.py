"""
epsilon-SVR from scratch for illiquid CDS spread prediction.

Companion to svm_cds_spreads.py (which calls sklearn's SVR) and to
svm_from_scratch.py (which solved the *classification* dual). Here we
write the **epsilon-SVR regression dual** out explicitly as a cvxpy QP,
solve it, recover the support-vector expansion and the bias b by hand,
and verify the result against sklearn's SVR trained with identical
(C, gamma, epsilon).

Problem (Halperin "SVM in Finance", Part IV): estimate the unobserved
5Y CDS spread of an illiquid company from features (rating, industry,
region, EDF) of liquid peers.

--- The epsilon-SVR primal (slides Part I) -------------------------------
    min_{w,b,xi,xi*}  (1/2)||w||^2 + C sum_i (xi_i + xi_i*)
    s.t.  y_i - <w,phi(x_i)> - b <= eps + xi_i        (point above the tube)
          <w,phi(x_i)> + b - y_i <= eps + xi_i*       (point below the tube)
          xi_i, xi_i* >= 0

--- The dual (slides Part III, page 2) -----------------------------------
Introduce multipliers alpha_i (upper constraint) and alpha_i* (lower),
let theta_i := alpha_i - alpha_i*. Then

    min_{alpha,alpha*}
        (1/2) sum_{i,j} theta_i theta_j K(x_i,x_j)
        + eps sum_i (alpha_i + alpha_i*)
        - sum_i y_i theta_i
    s.t.  sum_i theta_i = 0,   0 <= alpha_i, alpha_i* <= C

--- Decision function (support-vector expansion) -------------------------
    f(x) = sum_i theta_i K(x_i, x) + b
The solution depends on x only through the kernel -> kernel trick.

Install:
    pip install cvxpy
"""

import os

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, StandardScaler,
)
from sklearn.svm import SVR


# ----------------------------------------------------------------------------
# Constants  (shared with svm_cds_spreads.py so the two are comparable)
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
N_TRAIN      = 300     # liquid names used to solve the from-scratch QP
N_TEST       = 150     # illiquid names held out for evaluation

RATINGS      = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_PROBS = [0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05]
BASE_BY_RATING = {"AAA": 25, "AA": 50, "A": 100, "BBB": 200,
                  "BB": 400, "B": 700, "CCC": 1200}
INDUSTRIES   = ["Energy", "Financials", "Healthcare", "Industrials",
                "Materials", "Technology", "ConsumerStaples",
                "ConsumerDiscretionary", "Utilities", "Telecom"]
INDUSTRY_ADJ = {"Energy": +30, "Financials": +20, "Healthcare": -5,
                "Industrials": +10, "Materials": +25, "Technology": 0,
                "ConsumerStaples": -15, "ConsumerDiscretionary": +15,
                "Utilities": -25, "Telecom": +10}
REGIONS      = ["Northeast", "South", "Midwest", "West"]
REGION_ADJ   = {"Northeast": -5, "South": +5, "Midwest": 0, "West": -3}
EDF_LOC_BY_RATING = {"AAA": -7, "AA": -6, "A": -5, "BBB": -4,
                     "BB": -3, "B": -2, "CCC": -1}
FEATURE_COLS = ["rating", "industry", "region", "edf"]

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================================
# 1. Synthesise the universe (identical generator to svm_cds_spreads.py)
# ============================================================================
def generate_companies(n, seed_offset=0):
    r = np.random.default_rng(RANDOM_STATE + seed_offset)
    ratings    = r.choice(RATINGS, size=n, p=RATING_PROBS)
    industries = r.choice(INDUSTRIES, size=n)
    regions    = r.choice(REGIONS, size=n)
    edf_loc = np.array([EDF_LOC_BY_RATING[x] for x in ratings])
    edf     = np.exp(edf_loc + r.normal(0, 0.6, size=n))
    return pd.DataFrame({"rating": ratings, "industry": industries,
                         "region": regions, "edf": edf})


def true_spread(df, noise_sd=15.0):
    base = np.array([BASE_BY_RATING[x] for x in df["rating"]])
    indu = np.array([INDUSTRY_ADJ[x]   for x in df["industry"]])
    reg  = np.array([REGION_ADJ[x]     for x in df["region"]])
    edf_term = 80 * np.log10(df["edf"].values * 1e6) - 100
    noise = rng.normal(0, noise_sd, size=len(df))
    return np.clip(base + indu + reg + edf_term + noise, 5, 3000)


liquid_df   = generate_companies(N_TRAIN, seed_offset=0)
illiquid_df = generate_companies(N_TEST,  seed_offset=1)
liquid_df["spread"]   = true_spread(liquid_df)
illiquid_df["spread"] = true_spread(illiquid_df)

print(f"Liquid (train) : {len(liquid_df)} names, "
      f"avg {liquid_df['spread'].mean():.0f} bps")
print(f"Illiquid (test): {len(illiquid_df)} names, "
      f"avg {illiquid_df['spread'].mean():.0f} bps")


# ============================================================================
# 2. Preprocess: one-hot categoricals + log10/scale EDF; target = log10(spread)
# ============================================================================
pre = ColumnTransformer([
    ("cats", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
              ["rating", "industry", "region"]),
    ("edf", Pipeline([("log", FunctionTransformer(np.log10)),
                      ("std", StandardScaler())]), ["edf"]),
]).fit(liquid_df[FEATURE_COLS])

X_train = pre.transform(liquid_df[FEATURE_COLS])
X_test  = pre.transform(illiquid_df[FEATURE_COLS])
y_train = np.log10(liquid_df["spread"].values)        # regression target
y_test  = np.log10(illiquid_df["spread"].values)

n_train, n_features = X_train.shape
print(f"Design matrix  : {n_train} x {n_features} (after one-hot + scaling)")


# ============================================================================
# 3. Hyperparameters (pinned so cvxpy and sklearn build the SAME kernel)
# ============================================================================
C_value   = 10.0
EPS_value = 0.05      # epsilon-insensitive half-width, in log10(bps) units
# Replicate sklearn's gamma="scale" formula so both solvers agree.
gamma_value = 1.0 / (n_features * X_train.var())
print(f"\nHyperparameters: C = {C_value}, eps = {EPS_value}, "
      f"gamma = {gamma_value:.5f}")


# ============================================================================
# 4. Kernel (Gram) matrix  K[i,j] = exp(-gamma ||x_i - x_j||^2)
# ============================================================================
K_train = rbf_kernel(X_train, X_train, gamma=gamma_value)


# ============================================================================
# 5. Build and solve the epsilon-SVR dual QP  <-- THE SVM, written out
# ============================================================================
# Two multiplier vectors: alpha (upper tube) and alpha_star (lower tube).
alpha      = cp.Variable(n_train, nonneg=True)
alpha_star = cp.Variable(n_train, nonneg=True)

# Net dual coefficient theta_i = alpha_i - alpha_star_i.
theta = alpha - alpha_star

# Dual objective (minimisation form):
#   (1/2) theta^T K theta + eps * sum(alpha + alpha*) - y^T theta
objective = cp.Minimize(
    0.5 * cp.quad_form(theta, cp.psd_wrap(K_train))
    + EPS_value * cp.sum(alpha + alpha_star)
    - y_train @ theta
)

# Constraints: balance + box bounds.
constraints = [
    cp.sum(theta) == 0,        # sum_i (alpha_i - alpha_i*) = 0
    alpha      <= C_value,     # 0 <= alpha_i      <= C
    alpha_star <= C_value,     # 0 <= alpha_i*     <= C
]

problem = cp.Problem(objective, constraints)
problem.solve()

print(f"\nDual QP solved : status = {problem.status}")
print(f"  optimal objective = {problem.value:.6f}")

alpha_v      = np.asarray(alpha.value).ravel()
alpha_star_v = np.asarray(alpha_star.value).ravel()
theta_v      = alpha_v - alpha_star_v


# ============================================================================
# 6. Identify support vectors and recover b from the KKT tube conditions
# ============================================================================
TOL = 1e-6
# A training point is a support vector iff theta_i != 0.
sv_mask    = np.abs(theta_v) > TOL
n_sv       = int(sv_mask.sum())

# Free SVs sit exactly on a tube edge: one multiplier strictly in (0, C),
# the other zero. For those, the residual equals +/- eps, which pins b.
#   upper edge: 0 < alpha_i  < C  ->  f(x_i) = y_i - eps
#   lower edge: 0 < alpha_i* < C  ->  f(x_i) = y_i + eps
upper_free = (alpha_v      > TOL) & (alpha_v      < C_value - TOL)
lower_free = (alpha_star_v > TOL) & (alpha_star_v < C_value - TOL)

# Kernel evaluations between every training point and the SVs:
#   raw_f_i = sum_j theta_j K(x_j, x_i)   (the f(x_i) WITHOUT the bias b)
raw_f = K_train @ theta_v

b_estimates = []
b_estimates += list(y_train[upper_free] - EPS_value - raw_f[upper_free])
b_estimates += list(y_train[lower_free] + EPS_value - raw_f[lower_free])
b_scratch = float(np.mean(b_estimates)) if b_estimates else 0.0

print(f"\nFrom-scratch solution")
print(f"  support vectors      : {n_sv} / {n_train} "
      f"({100*n_sv/n_train:.0f}%)")
print(f"  free SVs (tube edge) : {int(upper_free.sum() + lower_free.sum())}")
print(f"  intercept b          : {b_scratch:+.5f}")


# ============================================================================
# 7. Decision function = support-vector expansion
# ============================================================================
def predict_scratch_log(X):
    """f(x) = sum_i theta_i k(x_i, x) + b   (in log10(bps) units)."""
    K = rbf_kernel(X, X_train, gamma=gamma_value)
    return K @ theta_v + b_scratch


# ============================================================================
# 8. sklearn SVR with identical (C, gamma, eps) for an apples-to-apples check
# ============================================================================
svr = SVR(kernel="rbf", C=C_value, gamma=gamma_value, epsilon=EPS_value)
svr.fit(X_train, y_train)

print(f"\nsklearn SVR")
print(f"  support vectors      : {svr.support_.size} / {n_train}")
print(f"  intercept b          : {svr.intercept_[0]:+.5f}")


# ============================================================================
# 9. Compare the two solvers on the held-out illiquid pool
# ============================================================================
f_scratch_log = predict_scratch_log(X_test)
f_sklearn_log = svr.predict(X_test)

# Decision-function agreement (in log space).
max_err  = np.abs(f_scratch_log - f_sklearn_log).max()
mean_err = np.abs(f_scratch_log - f_sklearn_log).mean()
print(f"\nAgreement with sklearn (log10 space)")
print(f"  max  |f_scratch - f_sklearn| = {max_err:.2e}")
print(f"  mean |f_scratch - f_sklearn| = {mean_err:.2e}")

# Back-transform to bps and report business metrics for both.
spread_true     = illiquid_df["spread"].values
spread_scratch  = 10 ** f_scratch_log
spread_sklearn  = 10 ** f_sklearn_log

for name, pred in [("from-scratch", spread_scratch),
                   ("sklearn SVR ", spread_sklearn)]:
    mae  = mean_absolute_error(spread_true, pred)
    r2   = r2_score(spread_true, pred)
    mape = np.mean(np.abs(pred - spread_true) / spread_true) * 100
    print(f"  {name}:  MAE = {mae:6.1f} bps   R2 = {r2:.3f}   "
          f"MAPE = {mape:.1f}%")

if max_err < 1e-2:
    print("\n[OK] The from-scratch epsilon-SVR matches sklearn (sub-1e-2 in "
          "log space -> identical bps predictions; residual is solver "
          "tolerance: cvxpy CLARABEL vs libsvm SMO).")


# ============================================================================
# 10. Score a brand-new illiquid borrower with the from-scratch model
# ============================================================================
new_company = pd.DataFrame([{
    "rating": "BB", "industry": "Energy", "region": "South", "edf": 0.03,
}])
new_X = pre.transform(new_company[FEATURE_COLS])
new_spread = float(10 ** predict_scratch_log(new_X)[0])
print(f"\nNew borrower (BB Energy, South, EDF=3%): "
      f"from-scratch predicts {new_spread:.0f} bps")
