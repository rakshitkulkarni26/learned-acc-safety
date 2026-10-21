import numpy as np
import pandas as pd
import pytest

from lacc.learn.baselines import IDM_DEFAULT
from lacc.learn.replay import (
    event_metrics,
    idm_controller,
    pad_events,
    replay_recorded_controller,
    simulate,
    summarise,
    violations,
)


def make_events(n_events=4, n=200, dt=0.1):
    rows = []
    for e in range(n_events):
        t = np.arange(n) * dt
        v_lead = 14 + 2 * np.sin(0.2 * t + e)
        a_ego = 0.4 * np.cos(0.2 * t + e - 0.5)
        v_ego = 14 + np.concatenate([[0], np.cumsum(a_ego[:-1] * dt)])
        x_lead = np.concatenate([[0], np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * dt)])
        x_ego = np.concatenate([[0], np.cumsum(0.5 * (v_ego[1:] + v_ego[:-1]) * dt)])
        rows.append(
            pd.DataFrame(
                {
                    "event_id": e,
                    "frame_id": np.arange(n),
                    "gap": 25 + x_lead - x_ego,
                    "v_ego": v_ego,
                    "v_lead": v_lead,
                    "a_ego": a_ego,
                }  # fmt: skip
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_replaying_recorded_accelerations_reproduces_recording():
    rec = pad_events(make_events())
    sim = simulate(rec, replay_recorded_controller(rec))
    m = event_metrics(rec, sim)
    assert (m["spacing_rmse"] < 1e-6).all()
    assert (m["speed_rmse"] < 1e-9).all()
    assert not m["collided"].any()


def test_variable_event_lengths_are_respected():
    ev = make_events()
    ev = ev[~((ev["event_id"] == 1) & (ev["frame_id"] >= 120))]
    rec = pad_events(ev)
    assert list(rec["length"]) == [200, 120, 200, 200]
    sim = simulate(rec, idm_controller(IDM_DEFAULT))
    assert np.isnan(sim["gap"][1, 120:]).all()
    m = event_metrics(rec, sim)
    assert m.loc[1, "controlled_s"] == pytest.approx((120 - 34) * 0.1)


def test_collision_is_detected_and_event_stops():
    rec = pad_events(make_events(n_events=1))

    def reckless(win, k, idx):
        return np.full(len(idx), 4.0), None  # full throttle into the leader

    sim = simulate(rec, reckless)
    m = event_metrics(rec, sim)
    assert m["collided"].iloc[0]
    assert m["violation"].iloc[0]
    k = sim["collision_step"][0]
    assert sim["gap"][0, k] <= 0 and np.isnan(sim["gap"][0, k + 1 :]).all()
    assert summarise(m)["collisions"] == 1


def test_controller_sees_delayed_window():
    rec = pad_events(make_events(n_events=1))
    seen = {}

    def spy(win, k, idx):
        seen.setdefault(k, win.copy())
        return np.zeros(len(idx)), np.full(len(idx), 0.1)

    sim = simulate(rec, spy, history=30, delay=5)
    k0 = 34
    np.testing.assert_allclose(seen[k0][0, -1, 0], rec["gap"][0, k0 - 5], rtol=1e-6)
    np.testing.assert_allclose(seen[k0][0, 0, 1], rec["v_ego"][0, k0 - 5 - 29], rtol=1e-6)
    assert np.nanmax(sim["epistemic_std"]) == pytest.approx(0.1)


def test_violation_definition():
    gap = np.array([10.0, 10.0, 2.0, -0.1])
    v = np.array([10.0, 20.0, 10.0, 5.0])
    v_lead = np.array([10.0, 5.0, 10.0, 5.0])
    # ok; TTC = 10/15 < 1 s; THW = 0.2 s < 0.3 s; collision
    assert violations(gap, v, v_lead).tolist() == [False, True, True, True]
