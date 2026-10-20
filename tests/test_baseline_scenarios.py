"""Scenario regression tests for the IDM baseline.

Pass cases: the baseline must handle these without collision.
Known-failure cases: physically impossible to survive with the given brake
limit; they prove the failure detection actually fires.
"""

import pytest

from lacc.controllers import IDM
from lacc.metrics import kpis
from lacc.scenarios import LeadBrake
from lacc.sim import VehicleParams, simulate


def run(scn: LeadBrake, params: VehicleParams | None = None) -> dict:
    tr = simulate(
        IDM(),
        scn.lead_profile(),
        v_ego0=scn.v_ego0,
        v_lead0=scn.v_lead0,
        gap0=scn.gap0,
        duration=scn.duration,
        params=params,
    )
    return kpis(tr)


@pytest.mark.parametrize(
    "scn",
    [
        LeadBrake(),  # demo case
        LeadBrake(decel=2.0, brake_duration=5.0),  # gentle, long braking
        LeadBrake(v_ego0=30.0, v_lead0=30.0, gap0=50.0, decel=6.0, brake_duration=2.0),
        LeadBrake(v_ego0=15.0, v_lead0=15.0, gap0=25.0, decel=6.0, brake_duration=10.0),  # to stop
    ],
)
def test_idm_handles_regular_braking(scn):
    k = run(scn)
    assert not k["collision"]
    assert k["min_gap_m"] > 1.0


def test_known_failure_weak_brakes():
    # lead brakes hard to standstill, ego can only brake at 2 m/s^2 -> collision expected
    scn = LeadBrake(v_ego0=30.0, v_lead0=30.0, gap0=20.0, decel=8.0, brake_duration=10.0)
    k = run(scn, VehicleParams(a_min=-2.0))
    assert k["collision"]
