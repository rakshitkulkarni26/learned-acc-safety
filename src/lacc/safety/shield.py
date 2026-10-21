"""RQ3: monitor-triggered fallback in closed loop.

The learned policy drives. When the monitor raises an alarm, a rule-based fallback
controller takes over for at least ``hold`` steps (and longer while the alarm
persists), then hands control back. The fallback is IDM calibrated on closed-loop
spacing, which had no collisions in the Phase 2b replay (protocol change log,
docs/EXPERIMENT_PROTOCOL.md).
"""

from __future__ import annotations

import numpy as np

from ..learn.baselines import idm_accel
from ..learn.replay import ttc
from .labels import MONITORS, THW_CAP_S


def shield_controller(
    ensemble, idm_params, monitor: str, ecdfs, threshold, hold: int, n_events: int
):
    fallback_until = np.full(n_events, -1)
    was_fallback = np.zeros(n_events, dtype=bool)
    stats = {
        "interventions": np.zeros(n_events, dtype=int),
        "fallback_steps": np.zeros(n_events, dtype=int),
    }

    def ctrl(win: np.ndarray, k: int, idx: np.ndarray):
        p = ensemble.predict(win)
        gap, v, dv = win[:, -1, 0], win[:, -1, 1], win[:, -1, 2]
        vl = v + dv
        raw = {
            "uncertainty": p["epistemic_std"],
            "ttc": 1.0 / np.maximum(ttc(gap, v, vl), 1e-3),
            "thw": 1.0 / np.maximum(np.where(v > 1.0, gap / np.maximum(v, 1e-3), THW_CAP_S), 1e-3),
        }
        score = np.max([ecdfs[s](raw[s]) for s in MONITORS[monitor]], axis=0)
        alarm = score > threshold
        fallback_until[idx[alarm]] = np.maximum(fallback_until[idx[alarm]], k + hold - 1)
        in_fb = fallback_until[idx] >= k
        stats["interventions"][idx] += in_fb & ~was_fallback[idx]
        stats["fallback_steps"][idx] += in_fb
        was_fallback[idx] = in_fb
        acc = np.where(in_fb, idm_accel(idm_params, gap, v, vl), p["mean"])
        return acc, p["epistemic_std"]

    return ctrl, stats


def rms_jerk(sim: dict, valid: np.ndarray, dt: float = 0.1) -> float:
    a = np.where(valid, sim["a"], np.nan)
    j = np.diff(a, axis=1) / dt
    return float(np.sqrt(np.nanmean(j**2)))
