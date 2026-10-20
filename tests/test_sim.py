import numpy as np
import pytest

from lacc.controllers import IDM
from lacc.scenarios import LeadBrake
from lacc.sim import Observation, VehicleParams, simulate


def run(scn: LeadBrake, controller=None, params=None):
    return simulate(
        controller or IDM(),
        scn.lead_profile(),
        v_ego0=scn.v_ego0,
        v_lead0=scn.v_lead0,
        gap0=scn.gap0,
        duration=scn.duration,
        params=params,
    )


def test_deterministic():
    a, b = run(LeadBrake()), run(LeadBrake())
    np.testing.assert_array_equal(a.gap, b.gap)
    np.testing.assert_array_equal(a.a_ego, b.a_ego)


def test_initial_state_matches_scenario():
    scn = LeadBrake(v_ego0=20.0, v_lead0=22.0, gap0=33.0)
    tr = run(scn)
    assert tr.gap[0] == pytest.approx(33.0)
    assert tr.v_ego[0] == pytest.approx(20.0)
    assert tr.v_lead[0] == pytest.approx(22.0)


def test_acceleration_limits_respected():
    p = VehicleParams(a_max=1.0, a_min=-3.0)
    tr = run(LeadBrake(decel=6.0), params=p)
    assert tr.a_ego.max() <= 1.0 + 1e-9
    assert tr.a_ego.min() >= -3.0 - 1e-9


def test_no_reverse_driving():
    # lead brakes to a full stop; ego must stop, never go negative
    tr = run(LeadBrake(decel=6.0, brake_duration=10.0, duration=30.0))
    assert tr.v_ego.min() >= 0.0
    assert tr.v_lead.min() >= 0.0


def test_constant_speed_controller_closes_gap_linearly():
    def hold(_: Observation) -> float:
        return 0.0

    scn = LeadBrake(v_ego0=20.0, v_lead0=15.0, gap0=50.0, decel=0.0, duration=5.0)
    tr = run(scn, controller=hold)
    # gap shrinks by (20 - 15) m/s * 5 s = 25 m
    assert tr.gap[-1] == pytest.approx(25.0, abs=1e-6)


def test_simulation_stops_at_collision():
    def hold(_: Observation) -> float:
        return 0.0

    scn = LeadBrake(v_ego0=25.0, v_lead0=10.0, gap0=10.0, decel=0.0, duration=10.0)
    tr = run(scn, controller=hold)
    assert tr.gap[-1] <= 0.0
    assert tr.t[-1] < 10.0
