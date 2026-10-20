"""End-to-end NGSIM preparation: raw CSV -> clean -> smooth -> events -> splits -> report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import ngsim
from .events import extract_events
from .smoothing import smooth_kinematics
from .splits import assign_splits, choose_test_recordings

EXPECTED_RECORDINGS_PER_LOCATION = 3  # NGSIM US-101 and I-80 each have three 15-min periods


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def prepare_from_frame(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Run all processing steps on an already-loaded raw frame (used by tests too)."""
    dt = cfg.get("dt", 0.1)
    report: dict = {"config": cfg}

    clean, s = ngsim.clean(raw, dt=dt, tolerance_s=cfg.get("recording_tolerance_s", 5.0))
    report["clean"] = s

    sm_cfg = cfg.get("smoothing", {})
    smooth, s = smooth_kinematics(clean, dt=dt, **sm_cfg)
    report["smoothing"] = s

    ev_cfg = cfg.get("events", {})
    events, s = extract_events(
        smooth,
        dt=dt,
        ego_classes=tuple(cfg.get("ego_classes", [2])),
        lanes=tuple(cfg.get("lanes", [1, 2, 3, 4, 5])),
        **ev_cfg,
    )
    report["events"] = s

    sp_cfg = cfg.get("splits", {})
    test_recs = choose_test_recordings(events, sp_cfg.get("test_recordings_per_location", 1))
    events["split"] = assign_splits(
        events, test_recs, sp_cfg.get("val_fraction", 0.15), sp_cfg.get("seed", 42)
    )
    report["splits"] = {
        "test_recordings": test_recs,
        "events_per_split": events.groupby("split", observed=True)["event_id"].nunique().to_dict(),
        "hours_per_split": (events.groupby("split", observed=True).size() * dt / 3600)
        .round(2)
        .to_dict(),
    }

    report["warnings"] = _sanity_warnings(report, smooth)
    report["signal_stats"] = _signal_stats(events, smooth)
    return events, report, smooth


def _sanity_warnings(report: dict, smooth: pd.DataFrame) -> list[str]:
    warnings = []
    recs = report["clean"]["recordings"]
    for loc in {r.rsplit("_r", 1)[0] for r in recs}:
        n = sum(1 for r in recs if r.startswith(loc + "_r"))
        if n != EXPECTED_RECORDINGS_PER_LOCATION:
            warnings.append(
                f"{loc}: found {n} recordings, expected {EXPECTED_RECORDINGS_PER_LOCATION}. "
                "Check recording detection before trusting the splits."
            )
    if report["events"]["events"] == 0:
        warnings.append("No car-following events extracted.")
    return warnings


def _signal_stats(events: pd.DataFrame, smooth: pd.DataFrame) -> dict:
    def q(x: pd.Series) -> dict:
        x = x.dropna()
        return {
            k: round(float(np.quantile(x, p)), 3)
            for k, p in [("p1", 0.01), ("p50", 0.5), ("p99", 0.99)]
        }

    jerk_raw = smooth.groupby("segment")["a_raw"].diff() / 0.1
    jerk_s = smooth.groupby("segment")["a"].diff() / 0.1
    return {
        "gap_m": q(events["gap"]),
        "v_ego_mps": q(events["v_ego"]),
        "a_ego_mps2": q(events["a_ego"]),
        "thw_s": q(
            events.loc[events["v_ego"] > 1, "gap"] / events.loc[events["v_ego"] > 1, "v_ego"]
        ),
        "rms_jerk_raw_vs_smoothed": [
            round(float(np.sqrt(np.nanmean(jerk_raw**2))), 2),
            round(float(np.sqrt(np.nanmean(jerk_s**2))), 2),
        ],
    }


def _plot(events: pd.DataFrame, smooth: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    thw = events.loc[events["v_ego"] > 1, "gap"] / events.loc[events["v_ego"] > 1, "v_ego"]
    for a, data, label, rng in [
        (ax[0, 0], events["gap"], "gap [m]", (0, 80)),
        (ax[0, 1], events["v_ego"] * 3.6, "ego speed [km/h]", (0, 120)),
        (ax[0, 2], thw, "time headway [s]", (0, 6)),
        (ax[1, 0], events["a_ego"], "ego accel, smoothed [m/s²]", (-5, 5)),
    ]:
        a.hist(data.clip(*rng), bins=80, range=rng, color="#3b6ea5")
        a.set_xlabel(label)
        a.grid(alpha=0.3)

    seg = smooth["segment"].value_counts().index[0]
    s = smooth[smooth["segment"] == seg].head(600)
    t = (s["frame_id"] - s["frame_id"].iloc[0]) * 0.1
    ax[1, 1].plot(t, s["a_raw"], lw=0.8, alpha=0.6, label="NGSIM v_Acc")
    ax[1, 1].plot(t, s["a"], lw=1.6, label="smoothed")
    ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_ylabel("accel [m/s²]")
    ax[1, 1].set_ylim(-6, 6)
    ax[1, 1].legend()
    ax[1, 1].grid(alpha=0.3)

    counts = events.groupby("split", observed=True).size() * 0.1 / 3600
    ax[1, 2].bar(counts.index.astype(str), counts.values, color="#5a9e6f")
    ax[1, 2].set_ylabel("hours of car-following")
    ax[1, 2].grid(alpha=0.3, axis="y")

    fig.suptitle("NGSIM car-following events: data quality report")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def prepare(config_path: Path) -> dict:
    """Full pipeline from the raw CSV on disk."""
    cfg = load_config(config_path)
    raw_csv = Path(cfg["raw_csv"])
    interim = Path(cfg["interim_parquet"])
    out_dir = Path(cfg["out_dir"])
    report_dir = Path(cfg["report_dir"])

    if interim.exists():
        print(f"Loading cached {interim}")
        raw = pd.read_parquet(interim)
    else:
        if not raw_csv.exists():
            raise FileNotFoundError(f"{raw_csv} not found. Run `lacc data download` first.")
        print(f"Reading {raw_csv} (locations: {cfg['locations']}) ...")
        raw = ngsim.load_raw(raw_csv, cfg["locations"])
        interim.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(interim, index=False)

    print("Cleaning, smoothing, extracting events ...")
    events, report, smooth = prepare_from_frame(raw, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out_dir / "events.parquet", index=False)
    (report_dir / "data_report.json").write_text(json.dumps(report, indent=2, default=str))
    _plot(events, smooth, report_dir / "data_report.png")
    return report
