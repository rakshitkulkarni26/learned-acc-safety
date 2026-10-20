"""Download, load and clean the NGSIM vehicle trajectory data.

Source: US DOT, "Next Generation Simulation (NGSIM) Vehicle Trajectories and
Supporting Data", data.transportation.gov (public). Positions are in feet,
sampled at 10 Hz. This module converts to SI units and assigns a recording key,
because vehicle IDs are only unique *within* one 15-minute recording.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

NGSIM_CSV_URL = "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD"
FT = 0.3048  # feet -> metres

# raw column (lower-cased) -> internal name
COLUMNS = {
    "vehicle_id": "vehicle_id",
    "frame_id": "frame_id",
    "global_time": "global_time",
    "local_y": "y_ft",
    "v_length": "length_ft",
    "v_class": "v_class",
    "v_vel": "v_raw_fts",
    "v_acc": "a_raw_fts2",
    "lane_id": "lane",
    "preceding": "preceding",
    "location": "location",
}


def download(out_path: Path, url: str = NGSIM_CSV_URL, chunk_mb: int = 8) -> Path:
    """Stream the full NGSIM CSV (~1.5 GB) to disk with a simple progress line."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    done = 0
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(chunk_mb * 1024 * 1024):
            f.write(chunk)
            done += len(chunk)
            sys.stdout.write(f"\r  downloaded {done / 1e6:,.0f} MB")
            sys.stdout.flush()
    tmp.replace(out_path)
    print()
    return out_path


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip().lower())
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"NGSIM file is missing columns: {missing}")
    return df[list(COLUMNS)].rename(columns=COLUMNS)


def load_raw(csv_path: Path, locations: list[str], chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read only the needed columns and locations from the big CSV, in chunks."""
    wanted = {loc.lower() for loc in locations}
    parts = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk = _normalise_columns(chunk)
        chunk["location"] = chunk["location"].astype(str).str.strip().str.lower()
        parts.append(chunk[chunk["location"].isin(wanted)])
    if not parts:
        raise ValueError(f"No rows read from {csv_path}")
    return pd.concat(parts, ignore_index=True)


def assign_recordings(df: pd.DataFrame, dt: float = 0.1, tolerance_s: float = 5.0) -> pd.Series:
    """Infer a recording key per row.

    Within one recording, ``global_time - frame_id * dt`` is constant (the
    recording start time). Different 15-minute periods have different start
    times even when they are back-to-back, so clustering this offset per
    location separates them. Returns keys such as ``"us-101_r1"``.
    """
    offset_s = (
        df["global_time"].to_numpy(dtype=np.float64) / 1000.0
        - df["frame_id"].to_numpy(dtype=np.float64) * dt
    )
    keys = pd.Series(index=df.index, dtype=object)
    for loc, idx in df.groupby("location").groups.items():
        off = offset_s[df.index.get_indexer(idx)]
        uniq = np.unique(np.round(off, 1))
        breaks = np.flatnonzero(np.diff(uniq) > tolerance_s)
        starts = np.concatenate([[uniq[0]], uniq[breaks + 1]])
        rec_no = np.searchsorted(starts, off + 1e-6, side="right")  # 1-based
        keys.loc[idx] = [f"{loc}_r{n}" for n in rec_no]
    return keys


def clean(df: pd.DataFrame, dt: float = 0.1, tolerance_s: float = 5.0) -> tuple[pd.DataFrame, dict]:
    """Add recording key, drop duplicates, convert to SI units, sort."""
    stats = {"rows_in": int(len(df))}
    df = df.copy()
    df["recording"] = assign_recordings(df, dt=dt, tolerance_s=tolerance_s)
    key = ["recording", "vehicle_id", "frame_id"]
    df = df.drop_duplicates(subset=key, keep="first")
    stats["duplicates_dropped"] = stats["rows_in"] - int(len(df))

    out = pd.DataFrame(
        {
            "location": df["location"].astype("category"),
            "recording": df["recording"].astype("category"),
            "vehicle_id": df["vehicle_id"].astype(np.int32),
            "frame_id": df["frame_id"].astype(np.int32),
            "y": (df["y_ft"] * FT).astype(np.float64),
            "length": (df["length_ft"] * FT).astype(np.float32),
            "v_class": df["v_class"].astype(np.int8),
            "v_raw": (df["v_raw_fts"] * FT).astype(np.float32),
            "a_raw": (df["a_raw_fts2"] * FT).astype(np.float32),
            "lane": df["lane"].astype(np.int16),
            "preceding": df["preceding"].astype(np.int32),
        }
    )
    out = out.sort_values(key, kind="mergesort").reset_index(drop=True)
    stats["rows_out"] = int(len(out))
    stats["recordings"] = {
        str(r): {
            "vehicles": int(g["vehicle_id"].nunique()),
            "frames": int(g["frame_id"].nunique()),
            "duration_min": round((g["frame_id"].max() - g["frame_id"].min()) * dt / 60, 1),
        }
        for r, g in out.groupby("recording", observed=True)
    }
    return out, stats
