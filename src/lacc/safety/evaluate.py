"""RQ1/RQ2: how well, and how early, does each monitor announce failures? (offline)

All thresholds are set on the *validation* replay and applied unchanged to the
*test* replay. Thresholds are expressed as an **alarm budget**: the share of
normal (non-violation) driving time during which the monitor is in alarm on
validation data. Comparing monitors at the same budget makes the comparison fair.
"""

from __future__ import annotations

import numpy as np

from .labels import MONITORS, future_onset
from .scores import auprc, auroc, bootstrap_ci


def thresholds(scores_val: dict, normal_val: np.ndarray, budgets: list[float]) -> dict:
    """threshold[monitor][budget] such that the val alarm-time share equals the budget."""
    out = {}
    for name, s in scores_val.items():
        ref = s[normal_val & ~np.isnan(s)]
        out[name] = {str(b): float(np.quantile(ref, 1.0 - b)) for b in budgets}
    return out


def ranking_metrics(scores, masks, horizon, n_boot=200, stride=5, seed=0) -> dict:
    """AUROC / AUPRC per monitor for 'failure onset within horizon', with event-bootstrap CIs."""
    pos = future_onset(masks["onset"], horizon)
    base = masks["valid"] & ~masks["violation"]
    n, T = base.shape
    event_idx = np.broadcast_to(np.arange(n)[:, None], (n, T))
    out = {"positive_rate": float(pos[base].mean()), "positives": int(pos[base].sum())}
    sub = base.copy()
    sub[:, np.arange(T) % stride != 0] = False  # thinned sample for bootstrap speed
    for name, s in scores.items():
        m = base & ~np.isnan(s)
        ms = sub & ~np.isnan(s)
        out[name] = {
            "auroc": auroc(pos[m], s[m]),
            "auprc": auprc(pos[m], s[m]),
            "auroc_ci95": bootstrap_ci(auroc, pos[ms], s[ms], event_idx[ms], n_boot, seed),
            "auprc_ci95": bootstrap_ci(auprc, pos[ms], s[ms], event_idx[ms], n_boot, seed),
        }
    return out


def alarm_metrics(score, threshold, masks, horizon, dt=0.1) -> dict:
    """Event-level warning performance of one monitor at one threshold."""
    valid, viol, onset = masks["valid"], masks["violation"], masks["onset"]
    alarm = valid & (np.nan_to_num(score, nan=-np.inf) > threshold)
    hours = valid.sum() * dt / 3600

    warned, leads = [], []
    for i, k in zip(*np.nonzero(onset), strict=True):
        lo = max(0, k - horizon)
        hits = np.flatnonzero(alarm[i, lo:k])
        warned.append(hits.size > 0)
        if hits.size:
            leads.append((k - (lo + hits[0])) * dt)

    # alarm episodes start at rising edges during normal driving; an episode is
    # a false alarm if no failure onset follows within the horizon
    prev = np.concatenate([np.zeros((alarm.shape[0], 1), dtype=bool), alarm[:, :-1]], axis=1)
    rising = alarm & ~prev & ~viol
    upcoming = future_onset(onset, horizon)
    false_eps = int((rising & ~upcoming).sum())
    warned = np.array(warned)
    return {
        "failures": int(len(warned)),
        "warned_share": float(warned.mean()) if len(warned) else float("nan"),
        "median_lead_s": float(np.median(leads)) if leads else float("nan"),
        "p25_lead_s": float(np.percentile(leads, 25)) if leads else float("nan"),
        "false_alarms_per_hour": float(false_eps / max(hours, 1e-9)),
        "alarm_time_share": float(alarm[valid & ~viol].mean()),
        "warned_flags": warned,
    }


def complementarity(flags: dict[str, np.ndarray]) -> dict:
    """Failures warned by uncertainty but missed by the kinematic monitor, and vice versa."""
    u, k = flags["uncertainty"], flags["kinematic"]
    return {
        "uncertainty_only": int((u & ~k).sum()),
        "kinematic_only": int((k & ~u).sum()),
        "both": int((u & k).sum()),
        "neither": int((~u & ~k).sum()),
    }


def offline(scores, masks, thr, horizons, budgets, main_horizon, main_budget, dt, n_boot, stride):
    res = {"ranking": {}, "alarms": {}}
    for h_s in horizons:
        h = int(round(h_s / dt))
        res["ranking"][str(h_s)] = ranking_metrics(scores, masks, h, n_boot, stride)
    h = int(round(main_horizon / dt))
    flags = {}
    for b in budgets:
        res["alarms"][str(b)] = {}
        for name in MONITORS:
            m = alarm_metrics(scores[name], thr[name][str(b)], masks, h, dt)
            if b == main_budget:
                flags[name] = m["warned_flags"]
            res["alarms"][str(b)][name] = {k: v for k, v in m.items() if k != "warned_flags"}
    res["complementarity_main"] = complementarity(flags)
    return res
