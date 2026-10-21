import numpy as np
import pandas as pd
import pytest

from lacc.controllers import IDM
from lacc.learn.baselines import IDM_DEFAULT, calibrate_idm, idm_accel
from lacc.learn.windows import FEATURES, build_windows, select, summary
from lacc.sim import Observation


def make_events(n_events=6, n=80, seed=0):
    """Small synthetic event table in the Phase 1 output format."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        t = np.arange(n) * 0.1
        v_lead = 15 + 2 * np.sin(0.3 * t + e)
        v_ego = 15 + 2 * np.sin(0.3 * t + e - 0.4)
        gap = 20 + np.cumsum(v_lead - v_ego) * 0.1
        rows.append(
            pd.DataFrame(
                {
                    "event_id": e,
                    "frame_id": 100 + np.arange(n),
                    "t": t,
                    "gap": gap,
                    "v_ego": v_ego,
                    "v_lead": v_lead,
                    "a_ego": np.gradient(v_ego, 0.1) + rng.normal(0, 0.01, n),
                    "artefact": np.arange(n) % 17 == 0,
                    "split": ["train", "train", "train", "val", "test", "test"][e],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_window_shapes_and_counts():
    ev = make_events()
    w = build_windows(ev, history_s=3.0, delay_s=0.5, stride_s=0.5)
    H, D, S = 30, 5, 5
    per_event = len(range(H - 1 + D, 80, S))
    assert w["X"].shape == (6 * per_event, H, len(FEATURES))
    assert w["y"].shape == (6 * per_event,)
    s = summary(w)
    assert s["train"]["events"] == 3 and s["val"]["events"] == 1 and s["test"]["events"] == 2


def test_windows_are_causal_and_delayed():
    ev = make_events(n_events=1)
    w = build_windows(ev, history_s=3.0, delay_s=0.5, stride_s=0.5)
    first = 30 - 1 + 5
    # last input step is exactly `delay` before the target
    assert w["X"][0, -1, 0] == pytest.approx(ev["gap"].iloc[first - 5])
    assert w["X"][0, 0, 1] == pytest.approx(ev["v_ego"].iloc[first - 5 - 29])
    assert w["y"][0] == pytest.approx(ev["a_ego"].iloc[first])
    assert w["t"][0] == pytest.approx(ev["t"].iloc[first])
    dv = ev["v_lead"] - ev["v_ego"]
    assert w["X"][0, -1, 2] == pytest.approx(dv.iloc[first - 5])


def test_windows_do_not_cross_events():
    ev = make_events()
    w = build_windows(ev, history_s=3.0, delay_s=0.5, stride_s=0.1)
    for eid in np.unique(w["event_id"]):
        g = ev[ev["event_id"] == eid]["gap"].to_numpy()
        Xg = w["X"][w["event_id"] == eid][:, :, 0]
        assert np.isin(Xg, g.astype(np.float32)).all()


def test_split_selection_and_artefact_exclusion():
    w = build_windows(make_events(), stride_s=0.1)
    tr_all = select(w, "train", exclude_artefacts=False)
    tr = select(w, "train", exclude_artefacts=True)
    assert len(tr["y"]) < len(tr_all["y"])
    assert not tr["artefact"].any()
    assert set(np.unique(tr_all["event_id"])) == {0, 1, 2}


def test_vectorised_idm_matches_controller():
    rng = np.random.default_rng(1)
    gap, v, vl = rng.uniform(3, 60, 50), rng.uniform(0, 30, 50), rng.uniform(0, 30, 50)
    vec = idm_accel(IDM_DEFAULT, gap, v, vl)
    ctrl = IDM()
    ref = [
        ctrl(Observation(gap=g, v_ego=a, v_lead=b, a_ego=0.0))
        for g, a, b in zip(gap, v, vl, strict=True)
    ]
    np.testing.assert_allclose(vec, ref, rtol=1e-9)


def test_idm_calibration_recovers_parameters():
    rng = np.random.default_rng(2)
    true = np.array([25.0, 1.2, 3.0, 1.0, 2.5])
    n = 4000
    X = np.zeros((n, 3, 3), dtype=np.float32)
    X[:, -1, 0] = rng.uniform(5, 50, n)
    X[:, -1, 1] = rng.uniform(2, 25, n)
    X[:, -1, 2] = rng.uniform(-3, 3, n)
    gap, v = X[:, -1, 0], X[:, -1, 1]
    y = np.clip(idm_accel(true, gap, v, v + X[:, -1, 2]), -8, 2.5)
    res = calibrate_idm(X, y)
    assert res["train_rmse"] < 0.05
