"""Leakage-safe train / val / test splits.

Rules (docs/EXPERIMENT_PROTOCOL.md):
1. Split by recording first: whole recordings are held out as test.
2. In the remaining recordings, split by ego vehicle, never by time window.
3. An ego vehicle never appears in more than one split.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def _unit_hash(text: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)


def choose_test_recordings(events: pd.DataFrame, per_location: int = 1) -> list[str]:
    """Hold out the last ``per_location`` recordings (in time order) of each location."""
    chosen: list[str] = []
    for _, g in events.groupby("location", observed=True):
        recs = sorted(g["recording"].astype(str).unique(), key=lambda r: int(r.rsplit("_r", 1)[1]))
        chosen += recs[-per_location:] if per_location else []
    return chosen


def assign_splits(
    events: pd.DataFrame, test_recordings: list[str], val_fraction: float = 0.15, seed: int = 42
) -> pd.Series:
    """Return a split label per row: 'train', 'val' or 'test'."""
    rec = events["recording"].astype(str)
    ego_key = rec + ":" + events["ego_id"].astype(str)
    unique_keys = ego_key.unique()
    is_val = {k: _unit_hash(k, seed) < val_fraction for k in unique_keys}
    split = ego_key.map(lambda k: "val" if is_val[k] else "train")
    split[rec.isin(test_recordings)] = "test"
    return split.astype("category")
