"""Time-series GAN for synthetic log-return windows.

A simple adversarial setup: generator maps Gaussian noise to a length-W
log-return path; discriminator distinguishes real windows from synthetic.
Both are small fully-connected nets — enough to demonstrate the
adversarial training loop and reproduce simple stylized facts (heavy
tails, vol clustering trace) without claiming to capture full market
dynamics.

Usage:
    from src.rl.gan import train_gan, sample
    out = train_gan(returns, window=21, epochs=400)
    synthetic = sample(out['generator'], n=5000, window=21)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _real_windows(returns: pd.DataFrame, window: int) -> np.ndarray:
    """Stack all valid (window,)-length return windows from each ticker."""
    arr = returns.to_numpy()
    rows = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        for i in range(window, len(col) + 1):
            w = col[i - window:i]
            if np.isfinite(w).all():
                rows.append(w)
    return np.asarray(rows, dtype=np.float32)


def build_generator(noise_dim: int, window: int):
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(noise_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(window, activation="linear")(x)
    return tf.keras.Model(inputs, out, name="generator")


def build_discriminator(window: int):
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(window,))
    x = tf.keras.layers.Dense(64, activation="leaky_relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="leaky_relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, out, name="discriminator")


def train_gan(returns: pd.DataFrame, window: int = 21,
              noise_dim: int = 8, epochs: int = 200, batch_size: int = 128,
              d_lr: float = 1e-4, g_lr: float = 1e-4, seed: int = 0
              ) -> dict:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)

    X = _real_windows(returns, window)
    if len(X) < batch_size:
        raise ValueError("not enough real windows for GAN training")

    # Standardize log returns to a comparable scale; remember mean/std for sampling.
    mu, sd = X.mean(), X.std()
    Xs = (X - mu) / (sd + 1e-9)

    G = build_generator(noise_dim, window)
    D = build_discriminator(window)
    bce = tf.keras.losses.BinaryCrossentropy()
    g_opt = tf.keras.optimizers.Adam(g_lr, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(d_lr, beta_1=0.5)

    rng = np.random.default_rng(seed)
    history = {"d_loss": [], "g_loss": []}
    n = len(Xs)

    for ep in range(epochs):
        # One step per epoch is enough for a demo; bigger budgets would loop here.
        idx = rng.integers(0, n, size=batch_size)
        real = Xs[idx]
        z = rng.standard_normal((batch_size, noise_dim)).astype(np.float32)

        with tf.GradientTape() as tape_d:
            fake = G(z, training=True)
            d_real = D(real, training=True)
            d_fake = D(fake, training=True)
            d_loss = bce(tf.ones_like(d_real), d_real) + \
                bce(tf.zeros_like(d_fake), d_fake)
        d_grads = tape_d.gradient(d_loss, D.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, D.trainable_variables))

        z = rng.standard_normal((batch_size, noise_dim)).astype(np.float32)
        with tf.GradientTape() as tape_g:
            fake = G(z, training=True)
            d_fake = D(fake, training=False)
            g_loss = bce(tf.ones_like(d_fake), d_fake)
        g_grads = tape_g.gradient(g_loss, G.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, G.trainable_variables))

        history["d_loss"].append(float(d_loss))
        history["g_loss"].append(float(g_loss))

    return {
        "generator": G,
        "discriminator": D,
        "mean": float(mu),
        "std": float(sd),
        "history": history,
    }


def sample(out: dict, n: int = 1000, noise_dim: int = 8,
           seed: int = 0) -> np.ndarray:
    """Sample n synthetic log-return windows in the ORIGINAL return scale."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, noise_dim)).astype(np.float32)
    fake = out["generator"](z, training=False).numpy()
    return fake * out["std"] + out["mean"]


def stylized_facts(returns_flat: np.ndarray) -> dict:
    """Mean/std/skew/kurtosis/autocorr-of-abs-returns — for visual comparison
    between real and synthetic distributions."""
    from scipy import stats
    r = np.asarray(returns_flat).ravel()
    r = r[np.isfinite(r)]
    abs_r = np.abs(r)
    return {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "skew": float(stats.skew(r)),
        "kurtosis_excess": float(stats.kurtosis(r)),
        "autocorr_abs_lag1": float(pd.Series(abs_r).autocorr(lag=1)),
    }
