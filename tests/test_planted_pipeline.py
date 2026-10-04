"""Time headway variant and the SUMO log -> per second -> windows path."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.sumo_log import per_second, windows_from_per_second  # noqa: E402
from model.aggressiveness_model import headway_features, index_features  # noqa: E402


def test_headway_by_hand():
    # 15.8 m at 33.88 km/h (9.411 m/s): thw 1.679 s, n_p = 1 - 1.679/3 = 0.4404, squared 0.1939
    f = headway_features(33.88, 0.0, 15.8, 0.0)
    assert round(float(f[2]), 4) == 0.1939
    # same gap at 120 km/h: thw 0.474 s, n_p 0.842, squared 0.7090; in metres both give 0.4679
    assert round(float(headway_features(120.0, 0.0, 15.8, 0.0)[2]), 4) == 0.7090
    assert round(float(index_features(120.0, 0.0, 15.8, 0.0)[2]), 4) == 0.4679


def test_headway_no_leader_and_stopped():
    f = headway_features([50.0, 1.0, 50.0], [0, 0, 0], [0.0, 5.0, 200.0], [0, 0, 0])
    assert np.allclose(f[:, 2], 0.0)       # no leader, stopped (< 0.5 m/s), headway beyond 3 s


def _raw(label="aggressive", accel=1.0, n=200):
    t = np.round(np.arange(n) * 0.1, 1)
    v = 10.0 + accel * t
    return pd.DataFrame({"t": t, "vehicle_id": f"{label}_r0.0", "label": label, "speed_ms": v, "accel_ms2": accel,
                         "gap_m": 20.0, "lat_m": -0.3, "lane": 0, "edge": "hw1"})


def test_per_second_samples_whole_seconds_and_derives_accel():
    ps = per_second(_raw(accel=1.0))
    assert np.allclose(np.diff(ps["t"]), 1.0)
    assert np.allclose(ps["accel"].iloc[2:-2], 1.0)          # derivative of a 1 m/s2 ramp, the UAH way
    assert np.allclose(ps["wave_m"], 0.3)


def test_windows_from_per_second_keeps_label():
    ps = pd.concat([per_second(_raw(l, a)).assign(vehicle_id=f"{l}_x", label=l)
                    for l, a in [("aggressive", 2.0), ("normal", 0.5)]])
    w = windows_from_per_second(ps, "highway", "highway")
    assert set(w["behavior"]) == {"aggressive", "normal"}
    assert (w.groupby("behavior")["phi_accel"].mean()["aggressive"] > w.groupby("behavior")["phi_accel"].mean()["normal"])
