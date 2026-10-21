import numpy as np
import pandas as pd
import pytest

from lacc.learn.baselines import IDM_DEFAULT
from lacc.learn.replay import event_metrics, pad_events, simulate
from lacc.safety.evaluate import alarm_metrics, thresholds
from lacc.safety.labels import ECDF, future_onset, monitor_scores
from lacc.safety.scores import auprc, auroc, bootstrap_ci
from lacc.safety.shield import shield_controller


# ------------------------------------------------------------------ ranking metrics
def test_auroc_extremes_and_ties():
    y = np.array([0, 0, 1, 1], dtype=bool)
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auroc(y, np.ones(4)) == 0.5
    assert np.isnan(auroc(np.zeros(4, dtype=bool), np.arange(4)))


def test_auprc_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1], dtype=bool)
    assert auprc(y, np.array([0, 0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    y = rng.random(20000) < 0.1
    assert auprc(y, rng.random(20000)) == pytest.approx(0.1, abs=0.02)


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(1)
    y = rng.random(4000) < 0.2
    s = y + rng.normal(0, 1, 4000)
    lo, hi = bootstrap_ci(auroc, y, s, np.arange(4000) // 20, n_boot=100)
    assert lo < auroc(y, s) < hi


# ------------------------------------------------------------------ labels and monitors
def test_future_onset_window():
    onset = np.zeros((1, 20), dtype=bool)
    onset[0, 10] = True
    f = future_onset(onset, 3)
    assert f[0].tolist() == [i in (7, 8, 9) for i in range(20)]


def test_ecdf_and_combined_monitor():
    e = ECDF(np.arange(100.0))
    assert e(np.array([-1.0, 49.0, 1000.0])).tolist() == [0.0, 0.5, 1.0]
    raw = {"uncertainty": np.array([[10.0]]), "ttc": np.array([[90.0]]), "thw": np.array([[50.0]])}
    ecdfs = {k: ECDF(np.arange(100.0)) for k in raw}
    s = monitor_scores(raw, ecdfs)
    assert s["uncertainty"][0, 0] == pytest.approx(0.11)
    assert s["kinematic"][0, 0] == pytest.approx(0.91)  # max of ttc, thw
    assert s["combined"][0, 0] == pytest.approx(0.91)


def test_thresholds_hit_the_alarm_budget():
    rng = np.random.default_rng(2)
    s = {"m": rng.random((50, 400))}
    normal = np.ones((50, 400), dtype=bool)
    thr = thresholds(s, normal, [0.02])["m"]["0.02"]
    assert (s["m"] > thr).mean() == pytest.approx(0.02, abs=0.002)


def test_alarm_metrics_lead_time_and_false_alarms():
    T = 200
    masks = {
        "valid": np.ones((1, T), dtype=bool),
        "violation": np.zeros((1, T), dtype=bool),
        "onset": np.zeros((1, T), dtype=bool),
    }
    masks["violation"][0, 100:110] = True
    masks["onset"][0, 100] = True
    score = np.zeros((1, T))
    score[0, 85:100] = 1.0  # warning 1.5 s before the failure
    score[0, 20:25] = 1.0  # unrelated alarm episode: false alarm
    m = alarm_metrics(score, 0.5, masks, horizon=30, dt=0.1)
    assert m["failures"] == 1 and m["warned_share"] == 1.0
    assert m["median_lead_s"] == pytest.approx(1.5)
    hours = T * 0.1 / 3600
    assert m["false_alarms_per_hour"] == pytest.approx(1 / hours)


# ------------------------------------------------------------------ closed-loop fallback
class RecklessEnsemble:
    """Stand-in for the GRU ensemble: always accelerates hard, never uncertain."""

    def predict(self, win):
        n = len(win)
        return {"mean": np.full(n, 4.0), "epistemic_std": np.zeros(n)}


def _events(n_events=3, n=300, dt=0.1):
    rows = []
    for e in range(n_events):
        t = np.arange(n) * dt
        v = np.full(n, 12.0)
        rows.append(
            pd.DataFrame(
                {
                    "event_id": e,
                    "frame_id": np.arange(n),
                    "gap": 30.0,
                    "v_ego": v,
                    "v_lead": v,
                    "a_ego": 0.0 * t,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_kinematic_fallback_prevents_reckless_collisions():
    rec = pad_events(_events())
    ens = RecklessEnsemble()
    ref = np.linspace(0, 1, 1000)
    ecdfs = {k: ECDF(ref) for k in ("uncertainty", "ttc", "thw")}

    def no_shield(win, k, idx):
        return ens.predict(win)["mean"], None

    crash = event_metrics(rec, simulate(rec, no_shield))
    assert crash["collided"].all()

    ctrl, stats = shield_controller(ens, IDM_DEFAULT, "kinematic", ecdfs, 0.5, hold=20, n_events=3)
    safe = event_metrics(rec, simulate(rec, ctrl))
    assert not safe["collided"].any()
    assert (stats["interventions"] >= 1).all()
    assert (stats["fallback_steps"] >= 20).all()


def test_uncertainty_monitor_never_fires_when_models_agree():
    rec = pad_events(_events(n_events=1))
    ens = RecklessEnsemble()
    ecdfs = {k: ECDF(np.linspace(0, 1, 1000)) for k in ("uncertainty", "ttc", "thw")}
    ctrl, stats = shield_controller(
        ens, IDM_DEFAULT, "uncertainty", ecdfs, 0.5, hold=20, n_events=1
    )
    m = event_metrics(rec, simulate(rec, ctrl))
    assert stats["interventions"].sum() == 0
    assert m["collided"].all()  # uncertainty alone cannot catch a confident but wrong policy
