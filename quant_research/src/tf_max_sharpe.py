"""TensorFlow max-Sharpe optimizer (vendored from portfolio_analysis/tf_optimize.py).

Long-only, fully-invested weights via softmax(logits) parameterization,
optimized with Adam to maximize the Sharpe ratio (ret - rf) / vol.

Vendored here so the project is self-contained; the original lives in
portfolio_analysis/tf_optimize.py.
"""

from __future__ import annotations

import numpy as np


def _softmax_weights(logits):
    import tensorflow as tf
    return tf.nn.softmax(logits)


def _optimize(loss_fn, n_assets: int, lr: float, steps: int,
              seed: int = 0) -> np.ndarray:
    import tensorflow as tf
    tf.random.set_seed(seed)
    logits = tf.Variable(tf.zeros([n_assets], dtype=tf.float64))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            w = _softmax_weights(logits)
            loss = loss_fn(w)
        grads = tape.gradient(loss, [logits])
        opt.apply_gradients(zip(grads, [logits]))
    return _softmax_weights(logits).numpy()


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, risk_free: float,
                       lr: float = 0.05, steps: int = 4000) -> np.ndarray:
    import tensorflow as tf
    mu_t = tf.constant(mu, dtype=tf.float64)
    cov_t = tf.constant(cov, dtype=tf.float64)
    rf = tf.constant(risk_free, dtype=tf.float64)
    eps = tf.constant(1e-8, dtype=tf.float64)

    def loss(w):
        ret = tf.tensordot(w, mu_t, 1)
        var = tf.tensordot(w, tf.linalg.matvec(cov_t, w), 1)
        vol = tf.sqrt(var + eps)
        return -(ret - rf) / vol

    return _optimize(loss, len(mu), lr, steps)
