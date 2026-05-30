"""Market regime clustering.

Computes a small set of macro features on the equal-weighted market
basket (rolling vol, rolling skew, rolling mean), then k-means clusters
the standardized feature time series into K regimes. The cluster IDs
are aligned to the original date index.

Useful for an interview as "here is how I segment the data before
evaluating signal performance per regime".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def regime_features(returns: pd.DataFrame, vol_window: int = 21,
                    trend_window: int = 63) -> pd.DataFrame:
    market = returns.mean(axis=1)
    feat = pd.DataFrame({
        "vol_21d": market.rolling(vol_window).std() * np.sqrt(252),
        "ret_63d": market.rolling(trend_window).mean() * 252,
        "skew_63d": market.rolling(trend_window).skew(),
    }).dropna()
    return feat


def cluster_regimes(returns: pd.DataFrame, n_clusters: int = 3,
                    seed: int = 0) -> tuple[pd.Series, KMeans]:
    feat = regime_features(returns)
    scaler = StandardScaler()
    X = scaler.fit_transform(feat)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    return pd.Series(labels, index=feat.index, name="regime"), km


def label_regimes(labels: pd.Series, returns: pd.DataFrame) -> dict[int, str]:
    """Heuristically name each cluster by its average market vol & return."""
    market = returns.mean(axis=1).reindex(labels.index)
    by = market.groupby(labels)
    names = {}
    for cid, g in by:
        vol = g.std() * np.sqrt(252)
        ret = g.mean() * 252
        if vol > 0.25:
            names[cid] = "crisis"
        elif ret > 0.10:
            names[cid] = "bull"
        else:
            names[cid] = "calm"
    return names
