"""Deterministic longitudinal closed-loop simulator (ego follows one lead vehicle).

Kept deliberately simple and transparent: point-mass longitudinal dynamics,
a first-order actuator lag, acceleration limits and no reverse driving.
Every assumption lives in ``VehicleParams`` so it can be documented and tested.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class VehicleParams:
    """Ego vehicle assumptions."""

    length: float = 4.5  # m, used to convert positions to bumper-to-bumper gap
    a_max: float = 2.5  # m/s^2, max acceleration the actuator can deliver
    a_min: float = -8.0  # m/s^2, max braking the actuator can deliver
    tau: float = 0.3  # s, first-order actuator time constant


@dataclass(frozen=True)
class Observation:
    """What a controller is allowed to see at each step."""

    gap: float  # m, bumper-to-bumper distance to lead
    v_ego: float  # m/s
    v_lead: float  # m/s
    a_ego: float  # m/s^2, measured (actual) ego acceleration


Controller = Callable[[Observation], float]
LeadProfile = Callable[[float], float]  # time -> commanded lead acceleration


@dataclass
class Trace:
    """Time series produced by one simulation run."""

    t: np.ndarray
    gap: np.ndarray
    v_ego: np.ndarray
    v_lead: np.ndarray
    a_ego: np.ndarray
    a_cmd: np.ndarray
    meta: dict = field(default_factory=dict)


def simulate(
    controller: Controller,
    lead_profile: LeadProfile,
    *,
    v_ego0: float,
    v_lead0: float,
    gap0: float,
    duration: float,
    dt: float = 0.1,
    params: VehicleParams | None = None,
    stop_on_collision: bool = True,
) -> Trace:
    """Run one closed-loop simulation and return the full trace.

    Integration is semi-implicit Euler at step ``dt``. The run is fully
    deterministic: same inputs always give the same trace.
    """
    p = params or VehicleParams()
    n = int(round(duration / dt)) + 1

    t = np.zeros(n)
    gap = np.zeros(n)
    v_ego = np.zeros(n)
    v_lead = np.zeros(n)
    a_ego = np.zeros(n)
    a_cmd = np.zeros(n)

    x_ego, x_lead = 0.0, gap0 + p.length
    ve, vl, ae = v_ego0, v_lead0, 0.0
    last = n - 1

    for k in range(n):
        tk = k * dt
        g = x_lead - x_ego - p.length
        t[k], gap[k], v_ego[k], v_lead[k], a_ego[k] = tk, g, ve, vl, ae

        if stop_on_collision and g <= 0.0:
            last = k
            break

        cmd = float(controller(Observation(gap=g, v_ego=ve, v_lead=vl, a_ego=ae)))
        cmd = float(np.clip(cmd, p.a_min, p.a_max))
        a_cmd[k] = cmd

        # actuator lag, then limits
        ae += (cmd - ae) * min(dt / p.tau, 1.0)
        ae = float(np.clip(ae, p.a_min, p.a_max))

        # ego: no reverse driving
        ve_next = max(ve + ae * dt, 0.0)
        x_ego += 0.5 * (ve + ve_next) * dt
        ve = ve_next

        # lead follows its scripted acceleration, also without reversing
        vl_next = max(vl + lead_profile(tk) * dt, 0.0)
        x_lead += 0.5 * (vl + vl_next) * dt
        vl = vl_next

    s = slice(0, last + 1)
    return Trace(
        t=t[s],
        gap=gap[s],
        v_ego=v_ego[s],
        v_lead=v_lead[s],
        a_ego=a_ego[s],
        a_cmd=a_cmd[s],
        meta={"dt": dt, "params": p},
    )
