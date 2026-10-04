"""NGSIM recording periods and the pNEUMA cone gap."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.ngsim import _split_periods  # noqa: E402
from datasets.pneuma import per_second  # noqa: E402


def test_split_periods_separates_reused_ids():
    # vehicle 63 appears in two periods 17 min apart with the same Frame_IDs (as in Lankershim)
    t0a, t0b = 1118935679900, 1118936699900
    rows = [(63, f, t0a + 100 * f, 0) for f in (304, 305)] + [(63, f, t0b + 100 * f, 7) for f in (304, 305)]
    df = pd.DataFrame(rows, columns=["Vehicle_ID", "Frame_ID", "Global_Time", "Preceding"])
    out = _split_periods(df)
    assert out["Vehicle_ID"].nunique() == 2
    assert not out.duplicated(["Vehicle_ID", "Frame_ID"]).any()
    assert set(out["Preceding"]) == {0, 200007}          # leader id moved into the same period


def test_split_periods_single_period_unchanged():
    df = pd.DataFrame({"Vehicle_ID": [983, 983], "Frame_ID": [1, 2], "Global_Time": [1000100, 1000200], "Preceding": [979, 979]})
    assert _split_periods(df)["Vehicle_ID"].tolist() == [983, 983]


def _car(vid, x0, y, speed_ms=10.0, typ="Car", n=12):
    t = np.arange(0, n, 0.04)
    return pd.DataFrame({"vehicle_id": vid, "type": typ, "t": np.round(t, 2), "x": x0 + speed_ms * t, "y": y,
                         "speed_kmh": speed_ms * 3.6, "lon_acc": 0.0, "lat_acc": 0.0})


def test_pneuma_gap_to_car_ahead_in_cone():
    raw = pd.concat([_car(1, 0.0, 0.0), _car(2, 20.0, 0.5), _car(3, 10.0, 3.5)])   # 2 ahead in lane, 3 in next lane
    ps = per_second(raw)
    g = ps[(ps["vehicle_id"] == 1) & (ps["t"] == 5)]["gap_m"].iloc[0]
    assert np.isclose(g, 20.0 - 4.5)                     # centre distance minus half of two 4.5 m cars
    assert (ps.loc[ps["vehicle_id"] == 2, "gap_m"] == 0).all()   # nobody ahead of car 2
    assert (ps["wave_m"] == 0).all()
