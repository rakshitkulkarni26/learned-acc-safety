import numpy as np
import pytest

from lacc.metrics import TTC_CAP, jerk, time_headway, time_to_collision


def test_ttc_closing():
    ttc = time_to_collision(np.array([20.0]), np.array([25.0]), np.array([20.0]))
    assert ttc[0] == pytest.approx(4.0)


def test_ttc_opening_is_capped():
    ttc = time_to_collision(np.array([20.0]), np.array([20.0]), np.array([25.0]))
    assert ttc[0] == TTC_CAP


def test_time_headway_undefined_at_standstill():
    thw = time_headway(np.array([10.0, 10.0]), np.array([0.0, 20.0]))
    assert np.isnan(thw[0])
    assert thw[1] == pytest.approx(0.5)


def test_jerk_of_ramp():
    a = np.array([0.0, -0.5, -1.0, -1.5])
    np.testing.assert_allclose(jerk(a, 0.1), [-5.0, -5.0, -5.0])
