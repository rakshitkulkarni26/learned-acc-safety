import json

import numpy as np
import pytest
import yaml
from _ngsim_synth import CONFIG, DT, FT, N, make_raw, truth

from lacc.data import ngsim
from lacc.data.pipeline import prepare, prepare_from_frame
from lacc.data.smoothing import smooth_kinematics

TRIM = 10  # half of the 2 s (21-frame) acceleration window


@pytest.fixture(scope="module")
def raw():
    return ngsim._normalise_columns(make_raw()).assign(location=lambda d: d["location"].str.lower())


@pytest.fixture(scope="module")
def result(raw):
    return prepare_from_frame(raw, CONFIG)


def test_units_converted_to_metres(raw):
    clean, _ = ngsim.clean(raw)
    row = clean.iloc[0]
    assert row["length"] == pytest.approx(4.5, abs=1e-4)
    src = raw[(raw["vehicle_id"] == row["vehicle_id"]) & (raw["frame_id"] == row["frame_id"])]
    assert row["y"] == pytest.approx(src["y_ft"].iloc[0] * FT)


def test_recordings_separated_despite_reused_ids(raw):
    clean, stats = ngsim.clean(raw)
    assert sorted(stats["recordings"]) == [
        "i-80_r1", "i-80_r2", "i-80_r3", "us-101_r1", "us-101_r2", "us-101_r3",
    ]  # fmt: skip
    for info in stats["recordings"].values():
        assert info["vehicles"] == 4


def test_duplicate_rows_dropped(raw):
    _, stats = ngsim.clean(raw)
    assert stats["duplicates_dropped"] == 1


def test_smoothing_recovers_true_acceleration(raw):
    clean, _ = ngsim.clean(raw)
    sm, _ = smooth_kinematics(clean, dt=DT)
    v2 = sm[(sm["recording"] == "us-101_r1") & (sm["vehicle_id"] == 2)]
    tr = truth()
    kept = slice(TRIM, N - TRIM)  # edges are trimmed by the pipeline itself
    assert len(v2) == N - 2 * TRIM
    a_err = v2["a"].to_numpy() - tr["a2"][kept]
    v_err = v2["v"].to_numpy() - tr["v2"][kept]
    a_raw_err = v2["a_raw"].to_numpy() - tr["a2"][kept]
    assert np.sqrt(np.mean(a_err**2)) < 0.3
    assert np.abs(a_err).max() < 1.0  # no edge spikes left
    assert np.sqrt(np.mean(v_err**2)) < 0.15
    # smoothing must be far better than naive double differentiation
    assert np.sqrt(np.mean(a_err**2)) < 0.1 * np.sqrt(np.mean(a_raw_err**2))


def test_one_event_per_recording_with_correct_gap(result):
    events, report, _ = result
    assert report["events"]["events"] == 6
    assert set(events["ego_id"]) == {2}
    assert set(events["leader_id"]) == {1}
    ev = events[events["recording"] == "i-80_r2"]
    assert len(ev) == N - 2 * TRIM
    assert ev["t"].iloc[0] == 0.0
    assert ev["t"].iloc[-1] == pytest.approx((N - 2 * TRIM - 1) * DT)
    assert np.abs(ev["gap"].to_numpy() - truth()["gap2"][TRIM : N - TRIM]).max() < 0.5


def test_short_and_aux_lane_runs_excluded(result):
    events, report, _ = result
    assert 3 not in set(events["ego_id"])  # 8 s car-following, then lane change
    assert 4 not in set(events["ego_id"])  # auxiliary lane
    assert report["events"]["runs_too_short"] >= 6


def test_splits_are_leakage_free(result):
    events, report, _ = result
    assert report["splits"]["test_recordings"] == ["i-80_r3", "us-101_r3"]
    test = events[events["split"] == "test"]
    assert set(test["recording"]) == {"i-80_r3", "us-101_r3"}
    key = events["recording"].astype(str) + ":" + events["ego_id"].astype(str)
    assert (events.groupby(key)["split"].nunique() == 1).all()


def test_splits_deterministic(raw):
    a, _, _ = prepare_from_frame(raw, CONFIG)
    b, _, _ = prepare_from_frame(raw, CONFIG)
    assert (a["split"] == b["split"]).all()


def test_no_warnings_on_clean_data(result):
    assert result[1]["warnings"] == []


def test_prepare_end_to_end_from_csv(tmp_path):
    csv = tmp_path / "raw.csv"
    make_raw().to_csv(csv, index=False)
    cfg = dict(CONFIG)
    cfg.update(
        raw_csv=str(csv),
        interim_parquet=str(tmp_path / "interim.parquet"),
        out_dir=str(tmp_path / "processed"),
        report_dir=str(tmp_path / "report"),
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    report = prepare(cfg_path)
    assert report["events"]["events"] == 6
    assert (tmp_path / "processed" / "events.parquet").exists()
    assert (tmp_path / "report" / "data_report.png").exists()
    saved = json.loads((tmp_path / "report" / "data_report.json").read_text())
    assert saved["events"]["events"] == 6
    # second run uses the cached interim file
    assert prepare(cfg_path)["events"]["events"] == 6
