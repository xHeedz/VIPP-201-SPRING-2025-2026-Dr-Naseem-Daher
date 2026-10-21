"""Cleaning, reference formula and the configurable labeling environment, with hand computed values."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.cleaning import clean  # noqa: E402
from env.labeling_ref import LabelingRefEnv, state_vector  # noqa: E402
from model.aggressiveness_model import reference_score, reference_terms  # noqa: E402


def test_cleaning_rules():
    ps = pd.DataFrame({"t": [0, 1, 2, 3], "speed_kmh": [72.0, 72.0, 72.0, 72.0], "accel": [0.5, 9.0, 0.0, 0.0],
                       "gap_m": [0.0, 10.0, 1.0, 30.0], "wave_m": [0.2, 0.2, 3.0, 0.0], "lane_valid": [True, True, True, False]})
    d, log = clean(ps, "uah")
    assert log["rule1_accel_dropped"] == 1 and len(d) == 3          # 9 m/s2 dropped
    assert np.isnan(d["gap_m"].iloc[0]) and np.isnan(d["leader"].iloc[0])   # camera: 0 = not detected
    assert np.isnan(d["gap_m"].iloc[1])                             # 1 m at 72 km/h: impossible
    assert d["gap_m"].iloc[2] == 30.0 and d["leader"].iloc[2] == 1.0
    assert np.isnan(d["wave_m"].iloc[1]) and np.isnan(d["wave_m"].iloc[2])  # above 2 m, lane not valid
    d2, _ = clean(ps, "highD")
    assert d2["gap_m"].iloc[0] == 0.0 and d2["leader"].iloc[0] == 0.0       # drone: 0 = no car ahead


def test_reference_terms_and_unknown():
    # 100 km/h, 1 m/s2, 20 m, 0.3 m, highway (mix 0.5): 0.667, 0.04, (0.36 + 0.577) / 2, 0.2
    t = reference_terms(100.0, 1.0, 20.0, 0.3, "highway")
    assert np.allclose(t, [100 / 150, 0.04, 0.5 * 0.36 + 0.5 * (1 - (20 / (100 / 3.6)) / 3) ** 2, 0.2], atol=1e-6)
    w = (0.25, 0.25, 0.25, 0.25)
    assert np.isclose(reference_score([0.4, 0.0, np.nan, np.nan], w), 20.0)   # (0.25*0.4) / 0.5 x 100


def test_state_and_reward():
    terms = np.array([[0.6, 0.0, 0.5, np.nan], [0.6, 0.0, np.nan, np.nan]])
    lead = np.array([1.0, 0.0])
    w = (0.4, 0.1, 0.4, 0.1)
    v = state_vector(terms, lead, 2, 60, "urban", ["pts_speed_run", "prox_follow", "leader_share", "time", "env_urban"], w)
    assert np.allclose(v, [0.24, 0.2, 0.5, 2 / 60, 1.0])
    seq = [{"terms": np.tile([0.5, 0.0, 0.0, 0.0], (20, 1)), "leader": np.zeros(20), "label": "normal",
            "environment": "highway", "group": 0, "scenario": "x", "trip": "a"}]
    env = LabelingRefEnv(seq, features=["time"], wait_growth=0.001, cost="ordinal", wait5=True)
    env.reset(options={"seq": 0})
    _, r, done, _, _ = env.step(1)                      # wait 5 s from n = 1: 5 x 0.002 + 0.001 x (1+2+3+4+5)
    assert not done and np.isclose(r, -(5 * 0.002 + 0.001 * 15))
    _, r, done, _, _ = env.step(2)                      # label conservative, truth normal: neighbours, cost 1
    assert done and r == -1.0
    env.reset(options={"seq": 0})
    seq[0]["label"] = "aggressive"
    _, r, _, _, _ = env.step(2)                         # conservative for an aggressive driver: cost 2
    assert r == -2.0
