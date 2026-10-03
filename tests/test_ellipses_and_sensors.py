import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import ellipses  # noqa: E402
from model.noise import noise_suite, sensor_view  # noqa: E402


def test_influence_axes_match_the_worked_example():
    a, b, stretch = ellipses.influence_axes(4.5, 1.8, 25.0, 80.0)
    assert math.isclose(stretch, 42.5) and math.isclose(a, 44.75) and math.isclose(b, 2.2)
    a20, _, _ = ellipses.influence_axes(4.5, 1.8, 25.0, 20.0)
    assert math.isclose(a20, 22.25)


def test_body_points_lie_on_the_ellipse_behind_the_front_bumper():
    pts = ellipses.body_ellipse(100.0, 5.0, 4.0, 2.0)
    assert len(pts) == ellipses.N_POINTS
    for x, y in pts:                                   # centre (98, 5), half-axes 2 and 1
        assert math.isclose(((x - 98.0) / 2.0) ** 2 + ((y - 5.0) / 1.0) ** 2, 1.0, rel_tol=1e-9)
    assert max(x for x, _ in pts) <= 100.0 + 1e-9


def test_influence_zone_lies_mostly_ahead():
    pts = ellipses.influence_ellipse(100.0, 0.0, 4.0, 2.0, speed=20.0, ai=50.0)
    xs = [x for x, _ in pts]
    assert max(xs) - 100.0 > 98.0 - min(xs)


def test_categories_scale_with_the_threshold():
    assert ellipses.category(80, 70) == "aggressive"
    assert ellipses.category(50, 70) == "normal"
    assert ellipses.category(30, 70) == "conservative"
    assert ellipses.category(15, 12) == "aggressive"
    assert ellipses.category(8, 12) == "normal"
    assert ellipses.category(5, 12) == "conservative"
    assert ellipses.color_for(80, 70, alpha=70) == ellipses.RED + (70,)


def test_sensor_view_without_noise_keeps_the_signals_and_no_leader_stays_zero():
    t = np.arange(20.0)
    speed = np.linspace(10, 20, 20)
    gap = np.where(t < 10, 15.0, 0.0)
    lateral = np.full(20, 0.3)
    v, a, g, lat = sensor_view(t, speed, np.zeros(20), gap, lateral, noise_suite("gaussian", 0.0), key="car")
    assert np.allclose(v, speed) and np.allclose(g, gap) and np.allclose(lat, lateral)
    assert np.allclose(a[2:-2], 10 / 19)              # the slope of the speed ramp
    v2, a2, g2, lat2 = sensor_view(t, speed, np.zeros(20), gap, lateral, noise_suite("gaussian", 2.0), key="car")
    assert not np.allclose(v2, speed)
    assert np.all(g2[t >= 10] == 0.0) and np.all(g2[t < 10] > 0.0)
