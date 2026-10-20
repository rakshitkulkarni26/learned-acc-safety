"""Synthetic data in the raw NGSIM format (feet, 10 Hz, original column names).

Each recording contains:
- vehicle 1: leader in lane 2, speed 20 + 3 sin(0.2 t) m/s
- vehicle 2: car following vehicle 1 at gap 25 + 2 sin(0.1 t) m for the full 30 s
- vehicle 3: follows vehicle 2 for 8 s, then changes lane (too short to be an event)
- vehicle 4: in lane 6 (auxiliary lane, excluded)
Vehicle IDs are reused in every recording, recordings start back-to-back,
and one duplicated row is inserted, mimicking known NGSIM quirks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FT = 0.3048
DT = 0.1
N = 300  # 30 s per recording
L = 4.5  # m vehicle length
T = np.arange(N) * DT


def truth():
    y1 = 100 + 20 * T - 15 * np.cos(0.2 * T) + 15
    a1 = 0.6 * np.cos(0.2 * T)
    gap2 = 25 + 2 * np.sin(0.1 * T)
    y2 = y1 - L - gap2
    a2 = a1 + 0.02 * np.sin(0.1 * T)
    v1 = 20 + 3 * np.sin(0.2 * T)
    v2 = v1 - 0.2 * np.cos(0.1 * T)
    return {"y1": y1, "y2": y2, "a1": a1, "a2": a2, "v1": v1, "v2": v2, "gap2": gap2}


def _rows(loc, start_ms, frame0, vid, y_m, lane, preceding, vclass, rng, noise_m):
    n = len(y_m)
    y_noisy = y_m + rng.normal(0, noise_m, n)
    v = np.gradient(y_noisy, DT)
    a = np.gradient(v, DT)
    frames = frame0 + np.arange(n)
    return pd.DataFrame(
        {
            "Vehicle_ID": vid,
            "Frame_ID": frames,
            "Global_Time": start_ms + (frames - frame0) * 100 + frame0 * 100,
            "Local_Y": y_noisy / FT,
            "v_length": L / FT,
            "v_Class": vclass,
            "v_Vel": v / FT,
            "v_Acc": a / FT,
            "Lane_ID": lane,
            "Preceding": preceding,
            "Location": loc,
        }
    )


def make_raw(noise_m: float = 0.1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tr = truth()
    parts = []
    t0 = 1_118_846_980_200  # ms, arbitrary epoch time
    for li, loc in enumerate(["us-101", "i-80"]):
        for r in range(3):
            start = t0 + li * 10_000_000 + r * N * 100  # back-to-back recordings
            frame0 = 10 + 5 * r
            parts.append(_rows(loc, start, frame0, 1, tr["y1"], 2, 0, 2, rng, noise_m))
            parts.append(_rows(loc, start, frame0, 2, tr["y2"], 2, 1, 2, rng, noise_m))
            y3 = tr["y2"] - L - 20.0
            lane3 = np.where(T < 8.0, 2, 3)
            prec3 = np.where(T < 8.0, 2, 0)
            p3 = _rows(loc, start, frame0, 3, y3, 2, 0, 2, rng, noise_m)
            p3["Lane_ID"], p3["Preceding"] = lane3, prec3
            parts.append(p3)
            parts.append(_rows(loc, start, frame0, 4, tr["y1"] + 50, 6, 0, 2, rng, noise_m))
    raw = pd.concat(parts, ignore_index=True)
    raw = pd.concat([raw, raw.iloc[[5]]], ignore_index=True)  # one duplicated row
    return raw.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # unsorted, like CSVs


CONFIG = {
    "locations": ["us-101", "i-80"],
    "dt": 0.1,
    "recording_tolerance_s": 5.0,
    "ego_classes": [2],
    "lanes": [1, 2, 3, 4, 5],
    "smoothing": {"position_window_s": 1.0, "accel_window_s": 2.0},
    "events": {"min_duration_s": 15.0, "max_abs_accel": 8.0, "min_gap_m": 0.5},
    "splits": {"test_recordings_per_location": 1, "val_fraction": 0.5, "seed": 1},
}
