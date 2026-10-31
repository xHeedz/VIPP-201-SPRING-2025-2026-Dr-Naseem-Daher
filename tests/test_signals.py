"""Extra signals (TTC, relative speed, IDM residual) with hand computed values."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.signals import add_signals, idm_accel  # noqa: E402


def test_idm_hand_value():
    # 25 m/s, 30 m gap, closing 2 m/s, limit 120 km/h: free 1.026, s* = 2 + 37.5 + 14.43, braking 1.5 (53.93 / 30)^2
    a = idm_accel(np.array([25.0]), np.array([30.0]), np.array([2.0]), np.array([120 / 3.6]))[0]
    free = 1.5 * (1 - (25 / (120 / 3.6)) ** 4)
    s_star = 2 + 25 * 1.5 + 25 * 2 / (2 * np.sqrt(3.0))
    assert np.isclose(a, free - 1.5 * (s_star / 30) ** 2)
    assert np.isclose(a, -3.82, atol=0.01)


def test_ttc_and_relative_speed():
    # gap 30, 28, 26 m at 1 s steps: closing 2 m/s; middle second TTC = 28 / 2 = 14 s; 132 km/h in a 120 zone: +10%
    d = pd.DataFrame({"vid": "a", "t": [0.0, 1.0, 2.0], "speed_kmh": 132.0, "accel": 0.0, "gap_m": [30.0, 28.0, 26.0],
                      "wave_m": 0.0, "leader": 1.0, "speed_limit_kmh": 120.0, "traffic_kmh": 110.0})
    s = add_signals(d)
    assert np.isclose(s["closing_ms"].iloc[1], 2.0) and np.isclose(s["ttc_s"].iloc[1], 14.0)
    assert s["t_ttc"].iloc[1] == 0.0                                  # 14 s is above 4 s
    assert np.isclose(s["rel_limit"].iloc[1], 0.1) and np.isclose(s["t_rel_limit"].iloc[1], 0.1 / 0.3)
    assert np.isclose(s["rel_traffic"].iloc[1], 0.2)
    assert np.isnan(s["ttc_s"].iloc[0])                               # first second: closing speed unknown
