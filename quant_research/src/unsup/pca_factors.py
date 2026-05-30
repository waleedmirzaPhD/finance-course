"""Statistical factors via PCA on the return matrix.

The first PC of the universe's daily returns is empirically a near-perfect
proxy for the market factor (corr ~0.99 with an equal-weighted basket).
Subsequent PCs typically pick up sector / style dispersion.

The return is the (n_obs x n_components) loading matrix in time, plus a
diagnostic that compares each PC to a simple benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_pca_factors(returns: pd.DataFrame, n_components: int = 5
                    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Returns:
        factors      : DataFrame (date x PC_k) of factor returns
        loadings     : DataFrame (ticker x PC_k) of static loadings
        explained_var: np.ndarray of explained variance ratios
    """
    r = returns.dropna(how="any")
    if r.shape[0] < n_components * 5:
        raise ValueError("not enough observations for PCA")

    centered = r - r.mean()
    pca = PCA(n_components=n_components)
    pca.fit(centered.to_numpy())
    factors = pd.DataFrame(
        pca.transform(centered.to_numpy()),
        index=r.index,
        columns=[f"PC{k+1}" for k in range(n_components)],
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=r.columns,
        columns=factors.columns,
    )
    return factors, loadings, pca.explained_variance_ratio_


def market_factor_diagnostic(factors: pd.DataFrame,
                             returns: pd.DataFrame) -> pd.Series:
    """Correlation of each PC with the equal-weighted market basket."""
    market = returns.mean(axis=1).reindex(factors.index)
    return factors.apply(lambda f: f.corr(market))


def static_factor_returns(loadings: pd.DataFrame,
                          returns: pd.DataFrame) -> pd.DataFrame:
    """Apply static loadings cross-sectionally to get factor returns
    in periods outside the PCA training window."""
    common = loadings.index.intersection(returns.columns)
    return returns[common].fillna(0).to_numpy() @ loadings.loc[common].to_numpy() \
        / np.abs(loadings.loc[common]).sum(axis=0).to_numpy()
