"""Phase 3 end to end: thresholds on validation, offline evaluation and closed loop on test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..learn.replay import (
    ensemble_controller,
    event_metrics,
    idm_controller,
    pad_events,
    simulate,
    summarise,
)
from .evaluate import offline, thresholds
from .labels import fit_ecdfs, load_traces, monitor_scores, raw_scores, step_masks
from .shield import rms_jerk, shield_controller


def _subset(tr: dict, idx: np.ndarray) -> dict:
    n = len(tr["rec_length"])
    return {k: (v[idx] if v.ndim and v.shape[0] == n else v) for k, v in tr.items()}


def _rec(tr: dict) -> dict:
    return {k[4:]: v for k, v in tr.items() if k.startswith("rec_")}


def _sim(tr: dict) -> dict:
    return {k[4:]: v for k, v in tr.items() if k.startswith("sim_")}


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def ensure_val_traces(cfg, ens, log) -> Path:
    path = Path(cfg["safety"]["val_traces"])
    if path.exists():
        return path
    dt = cfg.get("dt", 0.1)
    H = int(round(cfg["windows"]["history_s"] / dt))
    D = int(round(cfg["windows"]["delay_s"] / dt))
    events = pd.read_parquet(cfg["events"])
    rec = pad_events(events[events["split"] == "val"], dt)
    log(
        f"Replaying {len(rec['length'])} validation events with the GRU ensemble"
        " (for thresholds) ..."
    )
    sim = simulate(
        rec, ensemble_controller(ens), dt, H, D, cfg["replay"]["a_min"], cfg["replay"]["a_max"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{f"rec_{k}": v for k, v in rec.items()}, **{f"sim_{k}": v for k, v in sim.items()}
    )
    return path


def run(cfg: dict, log=print, max_events: int | None = None) -> dict:
    from ..learn.model import Ensemble

    dt = cfg.get("dt", 0.1)
    H = int(round(cfg["windows"]["history_s"] / dt))
    D = int(round(cfg["windows"]["delay_s"] / dt))
    s = cfg["safety"]
    budgets, main_b = s["alarm_budgets"], s["main_alarm_budget"]
    ens = Ensemble.load(Path(cfg["out_dir"]))

    # ---- thresholds from validation
    val = load_traces(ensure_val_traces(cfg, ens, log))
    mv = step_masks(val)
    normal_val = mv["valid"] & ~mv["violation"]
    raw_val = raw_scores(val, D)
    ecdfs = fit_ecdfs(raw_val, normal_val)
    thr = thresholds(monitor_scores(raw_val, ecdfs), normal_val, budgets)

    # ---- offline evaluation on test
    test = load_traces(Path(s["test_traces"]))
    if max_events:
        rng = np.random.default_rng(0)
        test = _subset(
            test, np.sort(rng.choice(len(test["rec_length"]), max_events, replace=False))
        )
    mt = step_masks(test)
    scores = monitor_scores(raw_scores(test, D), ecdfs)
    log(f"Offline evaluation on {len(test['rec_length'])} test events ...")
    off = offline(
        scores,
        mt,
        thr,
        s["horizons_s"],
        budgets,
        s["main_horizon_s"],
        main_b,
        dt,
        s["n_boot"],
        s["bootstrap_stride"],
    )

    # ---- closed loop on test
    rec = _rec(test)
    n = len(rec["length"])
    idm_p = json.loads((Path(cfg["replay"]["report_dir"]) / "replay_metrics.json").read_text())
    idm_p = np.array(list(idm_p["idm_spacing_calibration"]["params"].values()))
    hold = int(round(s["fallback_hold_s"] / dt))
    k0 = H + D - 1

    closed = {}

    def record(name, sim, stats=None):
        m = event_metrics(rec, sim, dt)
        summ = summarise(m)
        valid = step_masks(
            {**{f"rec_{k}": v for k, v in rec.items()}, **{f"sim_{k}": v for k, v in sim.items()}}
        )["valid"]
        summ["rms_jerk"] = rms_jerk(sim, valid, dt)
        if stats is not None:
            hours = summ["controlled_hours"]
            summ["interventions_per_hour"] = float(stats["interventions"].sum() / max(hours, 1e-9))
            summ["fallback_time_share"] = float(stats["fallback_steps"].sum() / max(valid.sum(), 1))
        closed[name] = summ

    log("Closed loop on test: fallback IDM alone ...")
    record("fallback_idm_only", simulate(rec, idm_controller(idm_p), dt, H, D))
    record("gru_only", {**_sim(test), "k0": np.array(k0)})
    for mon in ("kinematic", "uncertainty", "combined"):
        log(f"Closed loop on test: GRU + {mon} monitor -> fallback ...")
        ctrl, stats = shield_controller(ens, idm_p, mon, ecdfs, thr[mon][str(main_b)], hold, n)
        record(
            f"gru+{mon}",
            simulate(rec, ctrl, dt, H, D, cfg["replay"]["a_min"], cfg["replay"]["a_max"]),
            stats,
        )

    out = _jsonable(
        {
            "thresholds": thr,
            "offline": off,
            "closed_loop": closed,
            "settings": {
                k: s[k]
                for k in (
                    "horizons_s",
                    "main_horizon_s",
                    "alarm_budgets",
                    "main_alarm_budget",
                    "fallback_hold_s",
                )
            },
            "test_events": n,
        }
    )
    rd = Path(s["report_dir"])
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "safety_metrics.json").write_text(json.dumps(out, indent=2))
    _plot(out, test, scores, mt, thr, main_b, rd / "safety_report.png", dt)
    return out


def _plot(out, test, scores, masks, thr, main_b, path, dt) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "uncertainty": "#d1495b",
        "kinematic": "#3b6ea5",
        "combined": "#5a9e6f",
        "ttc": "#8fb3d9",
        "thw": "#2b4a6f",
    }
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    hz = list(out["offline"]["ranking"])
    for j, mon in enumerate(["uncertainty", "ttc", "thw", "kinematic", "combined"]):
        vals = [out["offline"]["ranking"][h][mon]["auroc"] for h in hz]
        ci = np.array([out["offline"]["ranking"][h][mon]["auroc_ci95"] for h in hz])
        x = np.arange(len(hz)) + (j - 2) * 0.15
        ax[0, 0].errorbar(
            x,
            vals,
            yerr=[np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)],
            fmt="o",
            color=colors[mon],
            label=mon,
            capsize=3,
        )
    ax[0, 0].axhline(0.5, color="grey", ls=":", lw=1)
    ax[0, 0].set_xticks(range(len(hz)), [f"{h} s" for h in hz])
    ax[0, 0].set_xlabel("warning horizon")
    ax[0, 0].set_ylabel("AUROC (95 % CI)")
    ax[0, 0].set_title("RQ1: does the monitor rank pre-failure moments higher?")
    ax[0, 0].legend(fontsize=8)

    for mon in ["uncertainty", "kinematic", "combined"]:
        pts = [
            (
                out["offline"]["alarms"][b][mon]["false_alarms_per_hour"],
                out["offline"]["alarms"][b][mon]["warned_share"],
            )
            for b in out["offline"]["alarms"]
        ]
        pts.sort()
        ax[0, 1].plot(*zip(*pts, strict=True), marker="o", color=colors[mon], label=mon)
    ax[0, 1].set_xlabel("false alarms per hour")
    ax[0, 1].set_ylabel("share of failures warned in advance")
    ax[0, 1].set_title("RQ2: warnings vs false alarms (budgets 1/2/5 %)")
    ax[0, 1].legend(fontsize=8)

    cl = out["closed_loop"]
    names = list(cl)
    x = np.arange(len(names))
    ax[1, 0].bar(
        x - 0.2, [cl[k]["collisions"] for k in names], 0.4, label="collisions", color="#d1495b"
    )
    ax[1, 0].bar(
        x + 0.2,
        [cl[k]["events_with_violation"] for k in names],
        0.4,
        label="events with violation",
        color="#3b6ea5",
    )
    ax[1, 0].set_xticks(
        x, [k.replace("fallback_", "").replace("_", " ") for k in names], fontsize=8, rotation=15
    )
    ax[1, 0].set_title(f"RQ3: closed loop on test (alarm budget {float(main_b) * 100:.0f} %)")
    ax[1, 0].legend(fontsize=8)

    onset = masks["onset"]
    coll = np.flatnonzero(test["sim_collided"])
    if coll.size:
        i, kind = coll[0], "collision"
        k = int(test["sim_collision_step"][i])
    elif onset.any():
        i, k = (int(x[0]) for x in np.nonzero(onset))
        kind = "safety violation"
    if coll.size or onset.any():
        lo = max(int(test["sim_k0"]), k - 100)
        t = (np.arange(lo, k + 1) - k) * dt
        ax2 = ax[1, 1]
        ax2.plot(t, test["sim_gap"][i, lo : k + 1], "k", label="gap [m]")
        ax3 = ax2.twinx()
        for mon in ["uncertainty", "kinematic"]:
            ax3.plot(t, scores[mon][i, lo : k + 1], color=colors[mon], label=f"{mon} percentile")
            ax3.axhline(thr[mon][str(main_b)], color=colors[mon], ls="--", lw=0.8)
        ax3.set_ylim(0, 1.02)
        ax3.set_ylabel("monitor score (percentile of normal driving)")
        ax2.set_xlabel(f"time before {kind} [s]")
        ax2.set_ylabel("gap [m]")
        ax2.set_title(f"Example: last 10 s before a GRU {kind}")
        ax3.legend(fontsize=8, loc="lower left")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
