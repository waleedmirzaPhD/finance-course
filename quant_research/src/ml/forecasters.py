"""Cross-sectional return forecasters.

Every forecaster shares a common interface:

    fit(X: (n_obs, n_features), y: (n_obs,)) -> None
    predict(X: (n_obs, n_features)) -> (n_obs,)

so the walk-forward harness in `model_compare.py` can treat them uniformly.

Models included:
    - Ridge       (linear baseline)
    - RandomForest
    - GradientBoosting
    - MLP (TensorFlow / Keras, small fully-connected net)

The MLP is deliberately small and regularized — interview interlocutors
discount deep models on noisy financial data, so the point is to show
*how* one is wired in, not to claim it dominates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class Forecaster(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class RidgeForecaster:
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@dataclass
class RandomForestForecaster:
    n_estimators: int = 200
    max_depth: int = 6
    min_samples_leaf: int = 20
    random_state: int = 0

    def __post_init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@dataclass
class GBMForecaster:
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    random_state: int = 0

    def __post_init__(self) -> None:
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@dataclass
class MLPForecaster:
    """Tiny TensorFlow MLP with dropout. Standardizes features internally."""
    hidden: tuple[int, ...] = (32, 16)
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    random_state: int = 0
    verbose: int = 0

    def __post_init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = None

    def _build(self, n_features: int):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.random_state)
        inputs = tf.keras.Input(shape=(n_features,))
        x = inputs
        for h in self.hidden:
            x = tf.keras.layers.Dense(h, activation="relu")(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        out = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mse",
        )
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xs = self.scaler.fit_transform(X)
        self.model = self._build(Xs.shape[1])
        self.model.fit(Xs, y, epochs=self.epochs, batch_size=self.batch_size,
                       verbose=self.verbose, shuffle=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs, verbose=0).ravel()


def default_zoo() -> dict[str, Forecaster]:
    return {
        "Ridge": RidgeForecaster(alpha=1.0),
        "RandomForest": RandomForestForecaster(),
        "GBM": GBMForecaster(),
        "MLP": MLPForecaster(),
    }
