"""Closed-loop replay: a controller drives the ego behind a *recorded* leader.

Open-loop accuracy (Phase 2) only checks one-step predictions on human inputs.
In closed loop the controller's own actions change what it sees next, so small
errors can accumulate. Replay is the standard way to evaluate car-following
models on trajectory data.

Per event:
1. The first ``history + delay`` steps are the recorded human trajectory (warm-up,
   so the controller has a full input window).
2. From then on the controller chooses the ego acceleration every 0.1 s, using the
   same delayed 3 s window as in training; the leader follows its recording.
3. The simulated gap and speed are compared with the human's, and the protocol's
   safety measures are computed.

All events of a split are simulated together as one vectorised batch.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..metrics import TTC_CAP
from .baselines import IDM_BOUNDS, IDM_DEFAULT, idm_accel

# controller(window (n, H, 3), step k, active rows) -> (accel (n,), epistemic std (n,) or None)
Controller = Callable[[np.ndarray, int, np.ndarray], tuple[np.ndarray, np.ndarray | None]]

TTC_FAIL_S = 1.0  # docs/EXPERIMENT_PROTOCOL.md
THW_FAIL_S = 0.3
V_MIN_THW = 1.0


def pad_events(
    events: pd.DataFrame, dt: float = 0.1, max_steps: int | None = None
) -> dict[str, np.ndarray]:
    """Recorded events as padded (n_events, T) arrays, optionally truncated to ``max_steps``."""
    events = events.sort_values(["event_id", "frame_id"], kind="mergesort")
    groups = [g for _, g in events.groupby("event_id", sort=False)]
    if max_steps:
        groups = [g.iloc[:max_steps] for g in groups]
    n, T = len(groups), max(len(g) for g in groups)
    out = {k: np.full((n, T), np.nan) for k in ("gap", "v_ego", "v_lead", "a_ego")}
    length = np.zeros(n, dtype=int)
    ids = np.zeros(n, dtype=int)
    for i, g in enumerate(groups):
        m = len(g)
        length[i], ids[i] = m, g["event_id"].iloc[0]
        for k in out:
            out[k][i, :m] = g[k].to_numpy()
    # ego position by trapezoidal integration of recorded speed (same scheme as the simulator)
    v = np.nan_to_num(out["v_ego"])
    x = np.zeros_like(v)
    x[:, 1:] = np.cumsum(0.5 * (v[:, 1:] + v[:, :-1]) * dt, axis=1)
    out["x_ego"] = x
    out["rear_lead"] = x + out["gap"]  # leader's rear bumper in the same frame
    out["length"] = length
    out["event_id"] = ids
    return out


def simulate(
    rec: dict[str, np.ndarray],
    controller: Controller,
    dt: float = 0.1,
    history: int = 30,
    delay: int = 5,
    a_min: float = -8.0,
    a_max: float = 4.0,
) -> dict[str, np.ndarray]:
    """Replay all events in ``rec`` with ``controller`` in charge of the ego."""
    n, T = rec["gap"].shape
    L = rec["length"]
    k0 = history + delay - 1  # first controlled step

    gap = np.full((n, T), np.nan)
    v = np.full((n, T), np.nan)
    a = np.full((n, T), np.nan)
    epi = np.full((n, T), np.nan)
    gap[:, : k0 + 1] = rec["gap"][:, : k0 + 1]
    v[:, : k0 + 1] = rec["v_ego"][:, : k0 + 1]
    x = rec["x_ego"][:, k0].copy()
    collided = np.zeros(n, dtype=bool)
    collision_step = np.full(n, -1)

    for k in range(k0, T - 1):
        active = np.flatnonzero((k + 1 < L) & ~collided)
        if active.size == 0:
            break
        lo, hi = k - delay - history + 1, k - delay + 1
        win = np.stack(
            [gap[active, lo:hi], v[active, lo:hi], rec["v_lead"][active, lo:hi] - v[active, lo:hi]],
            axis=-1,
        ).astype(np.float32)
        acc, std = controller(win, k, active)
        acc = np.clip(acc, a_min, a_max)
        a[active, k] = acc
        if std is not None:
            epi[active, k] = std

        v_next = np.maximum(v[active, k] + acc * dt, 0.0)
        x[active] += 0.5 * (v[active, k] + v_next) * dt
        v[active, k + 1] = v_next
        gap[active, k + 1] = rec["rear_lead"][active, k + 1] - x[active]

        hit = active[gap[active, k + 1] <= 0.0]
        collided[hit] = True
        collision_step[hit] = k + 1

    return {"gap": gap, "v": v, "a": a, "epistemic_std": epi, "collided": collided,
            "collision_step": collision_step, "k0": np.array(k0)}  # fmt: skip


def ttc(gap: np.ndarray, v: np.ndarray, v_lead: np.ndarray) -> np.ndarray:
    closing = v - v_lead
    out = np.full_like(gap, TTC_CAP)
    m = closing > 1e-6
    out[m] = np.clip(gap[m] / closing[m], 0.0, TTC_CAP)
    return out


def violations(gap: np.ndarray, v: np.ndarray, v_lead: np.ndarray) -> np.ndarray:
    """Per-step safety violation as defined in the experiment protocol."""
    thw = np.where(v > V_MIN_THW, gap / np.maximum(v, 1e-6), np.inf)
    return (gap <= 0) | (ttc(gap, v, v_lead) < TTC_FAIL_S) | (thw < THW_FAIL_S)


def event_metrics(rec: dict, sim: dict, dt: float = 0.1) -> pd.DataFrame:
    """One row per event: accuracy against the human and safety measures."""
    k0 = int(sim["k0"])
    n, T = rec["gap"].shape
    steps = np.arange(T)[None, :]
    ctrl = (steps >= k0) & (steps < rec["length"][:, None]) & ~np.isnan(sim["gap"])

    def masked(x, fill=np.nan):
        return np.where(ctrl, x, fill)

    sp_err = masked(sim["gap"] - rec["gap"])
    v_err = masked(sim["v"] - rec["v_ego"])
    sim_ttc = masked(ttc(sim["gap"], sim["v"], rec["v_lead"]), TTC_CAP)
    hum_ttc = masked(ttc(rec["gap"], rec["v_ego"], rec["v_lead"]), TTC_CAP)
    sim_viol = masked(violations(sim["gap"], sim["v"], rec["v_lead"]), False).astype(bool)
    hum_viol = masked(violations(rec["gap"], rec["v_ego"], rec["v_lead"]), False).astype(bool)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # all-NaN rows (e.g. IDM has no epistemic std)
        return pd.DataFrame(
            {
                "event_id": rec["event_id"],
                "controlled_s": ctrl.sum(1) * dt,
                "spacing_rmse": np.sqrt(np.nanmean(sp_err**2, axis=1)),
                "speed_rmse": np.sqrt(np.nanmean(v_err**2, axis=1)),
                "final_spacing_error": np.array(
                    [
                        sp_err[i, np.flatnonzero(ctrl[i])[-1]] if ctrl[i].any() else np.nan
                        for i in range(n)
                    ]
                ),
                "collided": sim["collided"],
                "min_gap": np.nanmin(masked(sim["gap"]), axis=1),
                "min_ttc": sim_ttc.min(1),
                "violation": sim_viol.any(1),
                "violation_s": sim_viol.sum(1) * dt,
                "human_min_ttc": hum_ttc.min(1),
                "human_violation": hum_viol.any(1),
                "mean_epistemic_std": np.nanmean(masked(sim["epistemic_std"]), axis=1),
            }
        )


def summarise(m: pd.DataFrame) -> dict:
    hours = m["controlled_s"].sum() / 3600
    return {
        "events": int(len(m)),
        "controlled_hours": round(float(hours), 2),
        "spacing_rmse_mean": float(m["spacing_rmse"].mean()),
        "spacing_rmse_median": float(m["spacing_rmse"].median()),
        "speed_rmse_mean": float(m["speed_rmse"].mean()),
        "collisions": int(m["collided"].sum()),
        "events_with_violation": int(m["violation"].sum()),
        "violation_share": float(m["violation"].mean()),
        "violation_seconds_per_hour": float(m["violation_s"].sum() / max(hours, 1e-9)),
        "human_events_with_violation": int(m["human_violation"].sum()),
    }


# ----------------------------------------------------------------- controllers
def idm_controller(params: np.ndarray) -> Controller:
    """IDM on the last (delayed) step of the window, like the open-loop baseline."""

    def ctrl(win: np.ndarray, k: int, idx: np.ndarray):
        gap, v, dv = win[:, -1, 0], win[:, -1, 1], win[:, -1, 2]
        return idm_accel(params, gap, v, v + dv), None

    return ctrl


def ensemble_controller(ensemble) -> Controller:
    def ctrl(win: np.ndarray, k: int, idx: np.ndarray):
        p = ensemble.predict(win)
        return p["mean"], p["epistemic_std"]

    return ctrl


def replay_recorded_controller(rec: dict) -> Controller:
    """Applies the human's recorded acceleration: reproduces the recording (used in tests)."""

    def ctrl(win: np.ndarray, k: int, idx: np.ndarray):
        return np.nan_to_num(rec["a_ego"][idx, k]), None

    return ctrl


def calibrate_idm_spacing(rec: dict, dt: float = 0.1, history: int = 30, delay: int = 5) -> dict:
    """Fit IDM parameters by minimising mean closed-loop spacing RMSE (standard practice)."""

    def loss(p: np.ndarray) -> float:
        sim = simulate(rec, idm_controller(p), dt, history, delay)
        m = event_metrics(rec, sim, dt)
        penalty = 5.0 * m["collided"].mean()  # discourage parameter sets that crash
        return float(np.nanmean(m["spacing_rmse"]) + penalty)

    res = minimize(loss, IDM_DEFAULT, method="Powell", bounds=IDM_BOUNDS,
                   options={"maxfev": 300, "xtol": 1e-2, "ftol": 1e-3})  # fmt: skip
    return {"params": dict(zip(("v0", "T", "s0", "a", "b"), map(float, res.x), strict=True)),
            "train_spacing_rmse": float(res.fun)}  # fmt: skip


# ----------------------------------------------------------------- full run
def run(cfg: dict, log=print) -> dict:
    """Closed-loop replay of one split with IDM baselines and the GRU ensemble."""
    import json
    from pathlib import Path

    from .model import Ensemble

    dt = cfg.get("dt", 0.1)
    H = int(round(cfg["windows"]["history_s"] / dt))
    D = int(round(cfg["windows"]["delay_s"] / dt))
    rcfg = cfg["replay"]
    events = pd.read_parquet(cfg["events"])

    # IDM calibrated on closed-loop spacing, on a sample of *training* events
    rng = np.random.default_rng(rcfg.get("seed", 0))
    train_ids = events.loc[events["split"] == "train", "event_id"].unique()
    cal_ids = rng.choice(
        train_ids, size=min(rcfg["calibration_events"], len(train_ids)), replace=False
    )
    cal = pad_events(events[events["event_id"].isin(cal_ids)], dt, rcfg["calibration_max_steps"])
    log(f"Calibrating IDM on closed-loop spacing ({len(cal_ids)} training events) ...")
    idm_sp = calibrate_idm_spacing(cal, dt, H, D)
    log(f"  params {', '.join(f'{k}={v:.2f}' for k, v in idm_sp['params'].items())}")

    rep = Path(cfg["report_dir"])
    acc_params = json.loads((rep / "policy_metrics.json").read_text())["idm_calibrated"]["params"]

    split_ev = events[events["split"] == rcfg["split"]]
    if rcfg.get("max_events"):
        ids = rng.choice(split_ev["event_id"].unique(), size=rcfg["max_events"], replace=False)
        split_ev = split_ev[split_ev["event_id"].isin(ids)]
    rec = pad_events(split_ev, dt)
    log(f"Replaying {len(rec['length'])} {rcfg['split']} events ...")

    controllers = {
        "idm_accel_calibrated": idm_controller(np.array(list(acc_params.values()))),
        "idm_spacing_calibrated": idm_controller(np.array(list(idm_sp["params"].values()))),
        "gru_ensemble": ensemble_controller(Ensemble.load(Path(cfg["out_dir"]))),
    }
    results, per_event, gru_sim = {}, [], None
    for name, ctrl in controllers.items():
        log(f"  {name} ...")
        sim = simulate(rec, ctrl, dt, H, D, rcfg["a_min"], rcfg["a_max"])
        m = event_metrics(rec, sim, dt)
        results[name] = summarise(m)
        per_event.append(m.assign(controller=name))
        if name == "gru_ensemble":
            gru_sim = sim

    human = per_event[0]
    results["human"] = {
        "events": int(len(human)),
        "events_with_violation": int(human["human_violation"].sum()),
        "violation_share": float(human["human_violation"].mean()),
    }
    out = {"idm_spacing_calibration": idm_sp, "idm_accel_params": acc_params, "results": results}

    rdir = Path(rcfg["report_dir"])
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "replay_metrics.json").write_text(json.dumps(out, indent=2))
    pd.concat(per_event).to_csv(rdir / "replay_per_event.csv", index=False)
    traces = Path(rcfg["traces_file"])
    traces.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        traces,
        **{f"rec_{k}": v for k, v in rec.items()},
        **{f"sim_{k}": v for k, v in gru_sim.items()},
    )
    _plot(rec, per_event, gru_sim, rdir / "replay_report.png", dt)
    return out


def _plot(rec, per_event, gru_sim, path, dt) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "idm_accel_calibrated": "IDM (accel-cal.)",
        "idm_spacing_calibrated": "IDM (spacing-cal.)",
        "gru_ensemble": "GRU ensemble",
    }
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    data = [m["spacing_rmse"].dropna().to_numpy() for m in per_event]
    ax[0].boxplot(data, showfliers=False)
    ax[0].set_xticks(
        range(1, len(data) + 1), [labels[m["controller"].iloc[0]] for m in per_event], fontsize=8
    )
    ax[0].set_ylabel("spacing RMSE per event [m]")
    ax[0].set_title("Closed-loop accuracy vs human")

    i = int(np.nanargmax(np.where(rec["length"] > 600, per_event[-1]["spacing_rmse"], np.nan)))
    n = rec["length"][i]
    t = np.arange(n) * dt
    ax[1].plot(t, rec["gap"][i, :n], "k", lw=1.5, label="human")
    ax[1].plot(t, gru_sim["gap"][i, :n], color="#3b6ea5", label="GRU ensemble")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("gap [m]")
    ax[1].set_title("Worst long test event (GRU)")
    ax[1].legend(fontsize=8)
    ax2 = ax[2]
    ax2.plot(t, gru_sim["epistemic_std"][i, :n], color="#d1495b")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("epistemic std [m/s²]")
    ax2.set_title("Ensemble disagreement, same event")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
