"""Baseline (non-learned) controllers.

The Intelligent Driver Model (IDM; Treiber, Hennecke & Helbing, 2000) is the
transparent reference policy. The learned policy added in a later phase is
always compared against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .sim import Observation


@dataclass(frozen=True)
class IDM:
    """Intelligent Driver Model car-following controller."""

    v0: float = 33.3  # m/s, desired speed (~120 km/h)
    T: float = 1.5  # s, desired time headway
    s0: float = 2.0  # m, minimum standstill gap
    a: float = 1.5  # m/s^2, comfortable acceleration
    b: float = 2.0  # m/s^2, comfortable deceleration
    delta: float = 4.0  # free-road exponent

    def desired_gap(self, v: float, dv: float) -> float:
        """Dynamic desired gap s*; dv = v_ego - v_lead (positive when closing)."""
        return self.s0 + max(0.0, v * self.T + v * dv / (2.0 * math.sqrt(self.a * self.b)))

    def __call__(self, obs: Observation) -> float:
        v = obs.v_ego
        dv = v - obs.v_lead
        s = max(obs.gap, 1e-3)
        s_star = self.desired_gap(v, dv)
        return self.a * (1.0 - (v / self.v0) ** self.delta - (s_star / s) ** 2)
