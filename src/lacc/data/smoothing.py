"""Kinematics from noisy video-based positions.

NGSIM positions come from video tracking and are noisy; differentiating them
twice amplifies that noise. We smooth the longitudinal position with a
Savitzky-Golay filter and take velocity and acceleration as its analytic
derivatives. A wider window is used for acceleration than for position and
velocity. Window lengths are configuration, not hidden constants, and are
documented in docs/DATA.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def window_frames(window_s: float, dt: float) -> int:
    return _odd(max(3, int(round(window_s / dt))))


def segment_ids(df: pd.DataFrame) -> np.ndarray:
    """Continuous segments: a new segment starts at a new vehicle or a frame gap."""
    new = (
        (df["recording"].ne(df["recording"].shift()))
        | (df["vehicle_id"].ne(df["vehicle_id"].shift()))
        | (df["frame_id"].diff().ne(1))
    )
    return np.cumsum(new.to_numpy()) - 1


def smooth_kinematics(
    df: pd.DataFrame, dt: float = 0.1, position_window_s: float = 1.0, accel_window_s: float = 2.0
) -> tuple[pd.DataFrame, dict]:
    """Add smoothed ``y_s``, ``v`` and ``a`` columns. ``df`` must be sorted.

    Segments shorter than two acceleration windows are dropped, and half an
    acceleration window is trimmed from both ends of each kept segment.
    """
    w_pos = window_frames(position_window_s, dt)
    w_acc = window_frames(accel_window_s, dt)

    seg = segment_ids(df)
    bounds = np.flatnonzero(np.diff(seg)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(df)]])

    y = df["y"].to_numpy(dtype=np.float64)
    y_s = np.full_like(y, np.nan)
    v = np.full_like(y, np.nan)
    a = np.full_like(y, np.nan)

    # Filter output near segment ends is biased (the window is one-sided there),
    # so half an acceleration window is trimmed from both ends of every segment.
    trim = w_acc // 2
    kept = 0
    for s, e in zip(starts, ends, strict=True):
        if e - s < 2 * w_acc:
            continue
        yy = y[s:e]
        inner = slice(s + trim, e - trim)
        y_s[inner] = savgol_filter(yy, w_pos, 2)[trim:-trim]
        v[inner] = savgol_filter(yy, w_pos, 2, deriv=1, delta=dt)[trim:-trim]
        a[inner] = savgol_filter(yy, w_acc, 3, deriv=2, delta=dt)[trim:-trim]
        kept += 1

    out = df.assign(y_s=y_s, v=v, a=a, segment=seg)
    out = out[~np.isnan(out["y_s"].to_numpy())].reset_index(drop=True)
    stats = {
        "segments_total": int(len(starts)),
        "segments_kept": int(kept),
        "position_window_frames": int(w_pos),
        "accel_window_frames": int(w_acc),
        "edge_trim_frames": int(trim),
        "rows_out": int(len(out)),
    }
    return out, stats
