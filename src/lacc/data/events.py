"""Extract car-following events: one ego following one leader in one lane.

An event is a maximal run of consecutive frames in which the ego vehicle
keeps the same lane and the same preceding vehicle, and both vehicles have
valid smoothed kinematics. Runs are cut wherever any of this changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _runs(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Run id that increments whenever any key changes or the frame jumps."""
    new = df["frame_id"].diff().ne(1)
    for k in keys:
        new |= df[k].ne(df[k].shift())
    return np.cumsum(new.to_numpy()) - 1


def extract_events(
    df: pd.DataFrame,
    dt: float = 0.1,
    min_duration_s: float = 15.0,
    ego_classes: tuple[int, ...] = (2,),
    lanes: tuple[int, ...] = (1, 2, 3, 4, 5),
    max_abs_accel: float = 8.0,
    min_gap_m: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Return a long table of event samples plus extraction statistics.

    ``df`` must contain smoothed kinematics (see ``smooth_kinematics``).
    """
    min_len = int(round(min_duration_s / dt))
    stats: dict[str, int] = {}

    leaders = df[["recording", "vehicle_id", "frame_id", "y_s", "v", "a", "length", "v_class"]]
    leaders = leaders.rename(
        columns={
            "vehicle_id": "leader_id",
            "y_s": "y_lead",
            "v": "v_lead",
            "a": "a_lead",
            "length": "length_lead",
            "v_class": "class_lead",
        }
    )

    ego = df[
        df["v_class"].isin(ego_classes) & df["lane"].isin(lanes) & (df["preceding"] > 0)
    ].rename(columns={"vehicle_id": "ego_id", "preceding": "leader_id", "v": "v_ego", "a": "a_ego"})
    ego = ego[
        [
            "location",
            "recording",
            "ego_id",
            "leader_id",
            "frame_id",
            "lane",
            "y_s",
            "v_ego",
            "a_ego",
            "raw_linear",
        ]
    ]

    merged = ego.merge(leaders, on=["recording", "leader_id", "frame_id"], how="inner")
    merged = merged.sort_values(["recording", "ego_id", "frame_id"], kind="mergesort")
    merged = merged.reset_index(drop=True)
    merged["gap"] = merged["y_lead"] - merged["length_lead"] - merged["y_s"]

    merged["run"] = _runs(merged, ["recording", "ego_id", "leader_id", "lane"])
    g = merged.groupby("run", sort=False)
    length = g["frame_id"].transform("size")
    bad = (
        (merged["gap"] < min_gap_m)
        | (merged["a_ego"].abs() > max_abs_accel)
        | (merged["a_lead"].abs() > max_abs_accel)
        | (merged["v_ego"] < -0.5)
        | (merged["v_lead"] < -0.5)
    )
    run_bad = bad.groupby(merged["run"]).transform("any")

    stats["candidate_runs"] = int(merged["run"].nunique())
    stats["runs_too_short"] = int(merged.loc[length < min_len, "run"].nunique())
    stats["runs_rejected_implausible"] = int(
        merged.loc[(length >= min_len) & run_bad, "run"].nunique()
    )

    ev = merged[(length >= min_len) & ~run_bad].copy()
    ev["event_id"] = pd.factorize(ev["run"])[0].astype(np.int32)
    ev["t"] = (ev["frame_id"] - ev.groupby("event_id")["frame_id"].transform("min")) * dt
    # Constant-speed artefact: raw position exactly linear for a full window while moving.
    # Standstill (v ~ 0) is genuine and is not flagged.
    ev["artefact"] = ev["raw_linear"] & (ev["v_ego"] > 0.5)
    ev = ev[
        [
            "event_id", "location", "recording", "ego_id", "leader_id", "frame_id", "t", "lane",
            "gap", "v_ego", "v_lead", "a_ego", "a_lead", "class_lead", "artefact",
        ]
    ].reset_index(drop=True)  # fmt: skip

    stats["events"] = int(ev["event_id"].nunique())
    stats["event_hours"] = round(len(ev) * dt / 3600, 2)
    stats["artefact_share"] = round(float(ev["artefact"].mean()), 3)
    return ev, stats
