"""Failure labels and monitor scores from replay traces.

Definitions follow docs/EXPERIMENT_PROTOCOL.md:
- a *violation* step is a collision, TTC < 1 s, or time headway < 0.3 s (ego > 1 m/s);
- a *failure event* starts at the first step of a run of violation steps (onset);
- a step t is *positive* if a failure onset occurs in (t, t + horizon].
Steps that are already in violation are excluded from the classification task:
announcing a failure that is already happening is trivial.

Monitors are computed on the **delayed** observation the controller itself sees
(the last step of its input window, ``delay`` steps old), so every monitor has
exactly the same information timing.
"""

from __future__ import annotations

import numpy as np

from ..learn.replay import ttc, violations

THW_CAP_S = 99.0


def load_traces(path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def step_masks(tr: dict) -> dict[str, np.ndarray]:
    """valid (controlled, not padded), violation, onset arrays, all (n_events, T)."""
    gap, v = tr["sim_gap"], tr["sim_v"]
    n, T = gap.shape
    k0 = int(tr["sim_k0"])
    steps = np.arange(T)[None, :]
    valid = (steps >= k0) & ~np.isnan(gap) & (steps < tr["rec_length"][:, None])
    g = np.nan_to_num(gap, nan=1e6)
    viol = valid & violations(g, np.nan_to_num(v), np.nan_to_num(tr["rec_v_lead"]))
    prev = np.concatenate([np.zeros((n, 1), dtype=bool), viol[:, :-1]], axis=1)
    onset = viol & ~prev
    return {"valid": valid, "violation": viol, "onset": onset}


def future_onset(onset: np.ndarray, horizon: int) -> np.ndarray:
    """True at t if an onset occurs at any step in (t, t + horizon]."""
    n, T = onset.shape
    cs = np.cumsum(onset, axis=1)
    ext = np.concatenate([cs, np.repeat(cs[:, -1:], horizon, axis=1)], axis=1)
    return (ext[:, horizon : horizon + T] - cs) > 0


def _delayed(x: np.ndarray, delay: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    out[:, delay:] = x[:, :-delay] if delay else x
    return out


def raw_scores(tr: dict, delay: int) -> dict[str, np.ndarray]:
    """Higher = more alarming. Shape (n_events, T); NaN where undefined."""
    g = _delayed(np.nan_to_num(tr["sim_gap"], nan=1e6), delay)
    v = _delayed(np.nan_to_num(tr["sim_v"]), delay)
    vl = _delayed(np.nan_to_num(tr["rec_v_lead"]), delay)
    with np.errstate(invalid="ignore", divide="ignore"):
        inv_ttc = 1.0 / np.maximum(
            ttc(np.nan_to_num(g, nan=1e6), np.nan_to_num(v), np.nan_to_num(vl)), 1e-3
        )
        thw = np.where(v > 1.0, g / np.maximum(v, 1e-3), THW_CAP_S)
        inv_thw = 1.0 / np.maximum(thw, 1e-3)
    inv_ttc[np.isnan(g)] = np.nan
    inv_thw[np.isnan(g)] = np.nan
    # the ensemble's disagreement is computed from the same delayed window at step k
    return {"uncertainty": tr["sim_epistemic_std"].astype(float), "ttc": inv_ttc, "thw": inv_thw}


class ECDF:
    """Empirical CDF fitted on reference (validation, normal-driving) scores."""

    def __init__(self, ref: np.ndarray):
        self.ref = np.sort(ref[~np.isnan(ref)])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.searchsorted(self.ref, x, side="right") / len(self.ref)
        return np.where(np.isnan(x), np.nan, out)

    def to_dict(self, n: int = 101) -> dict:
        return {"quantiles": np.quantile(self.ref, np.linspace(0, 1, n)).tolist()}


# Monitors compared in the protocol. Each maps raw scores to one number in [0, 1]
# (percentile of normal driving on the validation split), so they share a scale.
MONITORS = {
    "uncertainty": ("uncertainty",),
    "ttc": ("ttc",),
    "thw": ("thw",),
    "kinematic": ("ttc", "thw"),  # conventional monitor: TTC or headway
    "combined": ("uncertainty", "ttc", "thw"),  # uncertainty + conventional
}


def fit_ecdfs(raw: dict[str, np.ndarray], normal: np.ndarray) -> dict[str, ECDF]:
    return {k: ECDF(v[normal]) for k, v in raw.items()}


def monitor_scores(raw: dict, ecdfs: dict[str, ECDF]) -> dict[str, np.ndarray]:
    pct = {k: ecdfs[k](raw[k]) for k in raw}
    return {name: np.fmax.reduce([pct[s] for s in parts]) for name, parts in MONITORS.items()}
