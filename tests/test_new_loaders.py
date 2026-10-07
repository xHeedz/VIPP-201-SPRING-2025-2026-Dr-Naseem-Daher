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


def test_drivedna_per_second(tmp_path):
    from datasets.drivedna import per_second as dd_per_second
    t = np.round(np.arange(0, 30, 0.1), 1)
    df = pd.DataFrame({"time_s": t, "vEgo": 10.0 + 0.5 * t, "leadOne_status": (t >= 10).astype(float),
                       "leadOne_dRel": np.where(t >= 10, 20.0, 0.0), "laneLeft_y": np.nan, "laneRight_y": np.nan,
                       "is_human": 1, "cs_enabled": 0})
    p = tmp_path / "drive.csv"
    df.to_csv(p, index=False)
    ps = dd_per_second(str(p))
    assert len(ps) == 30
    assert np.allclose(ps["accel"].iloc[2:-2], 0.5)                     # 0.5 m/s2 ramp, measured the UAH way
    assert (ps.loc[ps["t"] < 10, "gap_m"] == 0).all() and (ps.loc[ps["t"] >= 10, "gap_m"] == 20).all()
    assert (ps["wave_m"] == 0).all() and not ps["lane_valid"].any()     # car model without lane positions


def make_highd(root):
    """one recording, 10 s at 25 Hz: a pair on each carriageway, leader 25.5 m (right) and 26 m (left) ahead"""
    import pandas as pd
    pd.DataFrame([{"id": 1, "frameRate": 25, "speedLimit": -1, "upperLaneMarkings": "5;8.5;12",
                   "lowerLaneMarkings": "20;23.5;27"}]).to_csv(os.path.join(root, "01_recordingMeta.csv"), index=False)
    pd.DataFrame({"id": [1, 2, 3, 4], "class": ["Car"] * 4, "drivingDirection": [2, 2, 1, 1]}).to_csv(
        os.path.join(root, "01_tracksMeta.csv"), index=False)
    rows = []
    for f in range(251):
        t = f / 25
        # id, x, y, width (length), height (width), xVelocity, precedingId
        for vid, x, y, L, W, vx, pre in [(1, 100 + 30 * t, 21.0, 4.5, 2.0, 30.0, 2), (2, 130 + 30 * t, 20.5, 5.0, 2.0, 30.0, 0),
                                          (3, 500 - 25 * t, 9.0, 4.0, 2.0, -25.0, 4), (4, 470 - 25 * t, 5.5, 4.0, 2.0, -25.0, 0)]:
            rows.append({"frame": f, "id": vid, "x": x, "y": y, "width": L, "height": W, "xVelocity": vx, "yVelocity": 0.0,
                         "xAcceleration": 0.0, "precedingId": pre, "dhw": 0.0, "laneId": 0})
    pd.DataFrame(rows).to_csv(os.path.join(root, "01_tracks.csv"), index=False)


def test_highd_gap_wave_speed(tmp_path):
    from datasets.highd import per_second, recordings
    make_highd(str(tmp_path))
    assert recordings(str(tmp_path)) == ["01"]
    ps = per_second(str(tmp_path), "01")
    assert len(ps) == 4 * 11                                    # whole seconds 0..10
    v1, v3 = ps[ps["vehicle_id"] == "01_1"], ps[ps["vehicle_id"] == "01_3"]
    # right: leader rear 130 + 30t, own front 100 + 30t + 4.5 -> 25.5 m; centre y 22.0 in lane 20..23.5 (21.75) -> 0.25
    assert np.allclose(v1["gap_m"], 25.5) and np.allclose(v1["wave_m"], 0.25) and np.allclose(v1["speed_kmh"], 108.0)
    # left: own front 500 - 25t, leader rear 470 - 25t + 4 -> 26 m; centre y 10.0 in lane 8.5..12 (10.25) -> 0.25
    assert np.allclose(v3["gap_m"], 26.0) and np.allclose(v3["wave_m"], 0.25) and np.allclose(v3["accel"], 0.0)
    assert (ps.loc[ps["vehicle_id"].isin(["01_2", "01_4"]), "gap_m"] == 0).all()   # leaders: no car ahead
