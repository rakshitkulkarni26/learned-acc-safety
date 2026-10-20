"""Parametric lead-vehicle scenarios.

Each scenario is a small, explicit set of parameters. Later phases sample
these parameters from distributions fitted to naturalistic data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .sim import LeadProfile


@dataclass(frozen=True)
class LeadBrake:
    """Lead cruises, brakes at a constant rate for a while, then holds speed."""

    v_ego0: float = 25.0  # m/s
    v_lead0: float = 25.0  # m/s
    gap0: float = 40.0  # m
    t_brake: float = 5.0  # s, when the lead starts braking
    decel: float = 4.0  # m/s^2 (positive number)
    brake_duration: float = 3.0  # s
    duration: float = 25.0  # s, total simulated time

    def lead_profile(self) -> LeadProfile:
        t0, t1, d = self.t_brake, self.t_brake + self.brake_duration, self.decel

        def profile(t: float) -> float:
            return -d if t0 <= t < t1 else 0.0

        return profile

    def as_dict(self) -> dict:
        return asdict(self)
