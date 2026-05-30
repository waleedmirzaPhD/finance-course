"""Denoising autoencoder for return-feature compression (TensorFlow).

Trains a small symmetric AE on cross-sectional return windows: each
sample is a length-W vector of recent log returns for one ticker, with
Gaussian noise added at the input. The bottleneck embedding is then
used as a derived signal (`reconstruction_error` flags regime shifts;
the bottleneck vector itself can blend with the classical signals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _window_matrix(returns: pd.DataFrame, window: int) -> tuple[np.ndarray, pd.MultiIndex]:
    """Stack rolling windows: each row = (date, ticker) -> (last W returns).

    Drops rows with any NaN inside the window.
    """
    rows, idx = [], []
    arr = returns.to_numpy()
    dates = returns.index
    tickers = returns.columns
    for j, t in enumerate(tickers):
        col = arr[:, j]
        for i in range(window, len(col) + 1):
            w = col[i - window:i]
            if not np.isfinite(w).all():
                continue
            rows.append(w)
            idx.append((dates[i - 1], t))
    return np.asarray(rows, dtype=np.float32), pd.MultiIndex.from_tuples(
        idx, names=["date", "ticker"]
    )


def train_autoencoder(returns: pd.DataFrame, window: int = 21,
                      bottleneck: int = 4, epochs: int = 10,
                      batch_size: int = 512, noise_std: float = 0.005,
                      seed: int = 0) -> dict:
    """Fit denoising AE on rolling return windows.

    Returns dict with:
        encoder, decoder, model, embeddings (DataFrame), recon_error (Series),
        history (Keras History dict).
    """
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)

    X, idx = _window_matrix(returns, window)
    if len(X) == 0:
        raise ValueError("no clean windows for AE training")

    inputs = tf.keras.Input(shape=(window,))
    noisy = tf.keras.layers.GaussianNoise(noise_std)(inputs)
    h = tf.keras.layers.Dense(window // 2, activation="relu")(noisy)
    z = tf.keras.layers.Dense(bottleneck, activation="linear", name="bottleneck")(h)
    h2 = tf.keras.layers.Dense(window // 2, activation="relu")(z)
    out = tf.keras.layers.Dense(window, activation="linear")(h2)

    model = tf.keras.Model(inputs, out)
    encoder = tf.keras.Model(inputs, z)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    hist = model.fit(X, X, epochs=epochs, batch_size=batch_size,
                     verbose=0, shuffle=True, validation_split=0.1)

    z_vals = encoder.predict(X, verbose=0)
    recon = model.predict(X, verbose=0)
    err = ((recon - X) ** 2).mean(axis=1)

    embeddings = pd.DataFrame(
        z_vals, index=idx, columns=[f"z{k+1}" for k in range(bottleneck)]
    )
    recon_error = pd.Series(err, index=idx, name="recon_error")
    return {
        "model": model,
        "encoder": encoder,
        "embeddings": embeddings,
        "recon_error": recon_error,
        "history": hist.history,
    }
