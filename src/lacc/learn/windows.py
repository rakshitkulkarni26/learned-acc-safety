"""Turn car-following events into supervised learning samples.

One sample = (input window, target):
- input: the last ``history_s`` seconds of [gap, ego speed, relative speed],
  ending ``delay_s`` before the target time;
- target: the ego's smoothed acceleration at the target time.

Why the delay: (1) drivers react with a delay, and (2) the smoothed signals use
a centred filter, so a speed value at time t already contains position
information from up to half a window *after* t. Ending the input at least half
a position window before the target keeps future information out of the inputs.

Windows never cross event boundaries, and every window inherits the split of
its event, so the leakage-safe split from Phase 1 is preserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("gap", "v_ego", "dv")  # dv = v_lead - v_ego (positive: gap opening)
SPLITS = ("train", "val", "test")


def event_features(ev: pd.DataFrame) -> np.ndarray:
    """(n, 3) float32 array of [gap, v_ego, dv] for one event, in time order."""
    dv = ev["v_lead"].to_numpy() - ev["v_ego"].to_numpy()
    return np.column_stack([ev["gap"].to_numpy(), ev["v_ego"].to_numpy(), dv]).astype(np.float32)


def build_windows(
    events: pd.DataFrame,
    dt: float = 0.1,
    history_s: float = 3.0,
    delay_s: float = 0.5,
    stride_s: float = 0.5,
) -> dict[str, np.ndarray]:
    """Slice all events into windows. Returns arrays with one row per sample."""
    H = int(round(history_s / dt))
    D = int(round(delay_s / dt))
    S = max(1, int(round(stride_s / dt)))

    events = events.sort_values(["event_id", "frame_id"], kind="mergesort")
    X, y, art, ev_id, t, split = [], [], [], [], [], []
    for eid, ev in events.groupby("event_id", sort=False):
        n = len(ev)
        first = H - 1 + D  # earliest target index with a full history
        if n <= first:
            continue
        feats = event_features(ev)
        targets = np.arange(first, n, S)
        starts = targets - D - H + 1
        idx = starts[:, None] + np.arange(H)[None, :]
        X.append(feats[idx])
        y.append(ev["a_ego"].to_numpy(dtype=np.float32)[targets])
        art.append(ev["artefact"].to_numpy(dtype=bool)[targets])
        ev_id.append(np.full(len(targets), eid, dtype=np.int32))
        t.append(ev["t"].to_numpy(dtype=np.float32)[targets])
        split.append(np.full(len(targets), SPLITS.index(str(ev["split"].iloc[0])), dtype=np.int8))

    return {
        "X": np.concatenate(X),
        "y": np.concatenate(y),
        "artefact": np.concatenate(art),
        "event_id": np.concatenate(ev_id),
        "t": np.concatenate(t),
        "split": np.concatenate(split),
        "meta": np.array([dt, history_s, delay_s, stride_s], dtype=np.float32),
    }


def save_windows(path: Path, w: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **w)


def load_windows(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def select(w: dict[str, np.ndarray], split: str, exclude_artefacts: bool) -> dict[str, np.ndarray]:
    """Rows of one split, optionally without artefact targets."""
    m = w["split"] == SPLITS.index(split)
    if exclude_artefacts:
        m &= ~w["artefact"]
    return {k: (v[m] if k != "meta" else v) for k, v in w.items()}


def summary(w: dict[str, np.ndarray]) -> dict:
    out = {}
    for s in SPLITS:
        m = w["split"] == SPLITS.index(s)
        out[s] = {
            "windows": int(m.sum()),
            "events": int(np.unique(w["event_id"][m]).size),
            "artefact_targets": int((w["artefact"] & m).sum()),
        }
    return out
