"""Safety and comfort KPIs computed from a simulation trace.

Definitions follow docs/EXPERIMENT_PROTOCOL.md and must not change
silently once experiments start.
"""

from __future__ import annotations

import numpy as np

from .sim import Trace

TTC_CAP = 99.0  # s, value reported when the gap is not closing


def time_to_collision(gap: np.ndarray, v_ego: np.ndarray, v_lead: np.ndarray) -> np.ndarray:
    """Constant-velocity TTC; ``TTC_CAP`` where the gap is opening or constant."""
    closing = v_ego - v_lead
    ttc = np.full_like(gap, TTC_CAP, dtype=float)
    mask = closing > 1e-6
    ttc[mask] = np.clip(gap[mask] / closing[mask], 0.0, TTC_CAP)
    return ttc


def time_headway(gap: np.ndarray, v_ego: np.ndarray, v_min: float = 1.0) -> np.ndarray:
    """Gap divided by ego speed; NaN when the ego is (nearly) stopped."""
    thw = np.full_like(gap, np.nan, dtype=float)
    mask = v_ego > v_min
    thw[mask] = gap[mask] / v_ego[mask]
    return thw


def jerk(a: np.ndarray, dt: float) -> np.ndarray:
    """First difference of acceleration."""
    if a.size < 2:
        return np.zeros(0)
    return np.diff(a) / dt


def kpis(trace: Trace) -> dict[str, float | bool]:
    """Summary KPIs for one run."""
    dt = trace.meta["dt"]
    ttc = time_to_collision(trace.gap, trace.v_ego, trace.v_lead)
    thw = time_headway(trace.gap, trace.v_ego)
    j = jerk(trace.a_ego, dt)
    return {
        "collision": bool(trace.gap.min() <= 0.0),
        "min_gap_m": float(trace.gap.min()),
        "min_ttc_s": float(ttc.min()),
        "min_thw_s": float(np.nanmin(thw)) if np.any(~np.isnan(thw)) else float("nan"),
        "max_decel_mps2": float(-trace.a_ego.min()),
        "max_abs_jerk_mps3": float(np.abs(j).max()) if j.size else 0.0,
        "rms_jerk_mps3": float(np.sqrt(np.mean(j**2))) if j.size else 0.0,
        "duration_s": float(trace.t[-1]),
    }
