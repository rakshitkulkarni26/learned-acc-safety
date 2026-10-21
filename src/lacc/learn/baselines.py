"""IDM as an open-loop baseline on the same windows the neural network sees.

Two versions are reported: IDM with textbook parameters and IDM calibrated on
the training split (least squares on acceleration). The calibrated IDM is the
fair baseline: if the neural network cannot beat it, its extra complexity is
not justified.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

IDM_PARAMS = ("v0", "T", "s0", "a", "b")
IDM_DEFAULT = np.array([33.3, 1.5, 2.0, 1.5, 2.0])
IDM_BOUNDS = [(5.0, 45.0), (0.1, 4.0), (0.0, 8.0), (0.02, 4.0), (0.1, 8.0)]


def idm_accel(p: np.ndarray, gap: np.ndarray, v: np.ndarray, v_lead: np.ndarray) -> np.ndarray:
    """Vectorised IDM, same equations as ``lacc.controllers.IDM``."""
    v0, T, s0, a, b = p
    dv = v - v_lead
    s_star = s0 + np.maximum(0.0, v * T + v * dv / (2.0 * np.sqrt(a * b)))
    s = np.maximum(gap, 1e-3)
    return a * (1.0 - (v / v0) ** 4 - (s_star / s) ** 2)


def last_state(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """gap, v_ego, v_lead at the last input step of each window."""
    gap, v, dv = X[:, -1, 0], X[:, -1, 1], X[:, -1, 2]
    return gap, v, v + dv


def idm_predict(
    p: np.ndarray, X: np.ndarray, a_min: float = -8.0, a_max: float = 2.5
) -> np.ndarray:
    return np.clip(idm_accel(p, *last_state(X)), a_min, a_max)


def calibrate_idm(X: np.ndarray, y: np.ndarray, max_samples: int = 100_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(max_samples, len(y)), replace=False)
    Xs, ys = X[idx], y[idx]

    def loss(p: np.ndarray) -> float:
        return float(np.mean((idm_predict(p, Xs) - ys) ** 2))

    res = minimize(loss, IDM_DEFAULT, method="L-BFGS-B", bounds=IDM_BOUNDS)
    return {
        "params": dict(zip(IDM_PARAMS, map(float, res.x), strict=True)),
        "train_rmse": res.fun**0.5,
    }
