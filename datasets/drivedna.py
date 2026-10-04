"""
DriveDNA-Sample loader (huggingface.co/datasets/HenryYHW/DriveDNA-Sample, research-only licence;
Wang et al., arXiv 2607.23822). One CSV per drive at 10 Hz (openpilot logs, schema drivedna_v1).

Per second table of datasets/uah.py (t, speed_kmh, accel, gap_m, wave_m), human driving only
(is_human = 1, cruise control off):
    speed   vEgo once per second, 3 s mean (UAH way); accel = its derivative
    gap     leadOne_dRel (radar distance to the lead vehicle) when leadOne_status = 1, else 0 (no car ahead)
    wave    |(laneLeft_y + laneRight_y) / 2|: offset of the car from the centre of its detected lane; several
            car models log no lane positions (all NaN): wave 0 there and lane_valid False
Drives in the older schema (legacy_gnss_removed, no lead or lane signals) are skipped.
No aggressiveness labels: driver ids allow driver level comparisons only.
"""
import os

import numpy as np
import pandas as pd

NEEDED = ["time_s", "vEgo", "leadOne_status", "leadOne_dRel", "laneLeft_y", "laneRight_y", "is_human", "cs_enabled"]


def find_drives(root):
    idx = pd.read_csv(os.path.join(root, "index.csv"))
    idx["path"] = [os.path.join(root, "Dataset", r.car_model, r.driver, r.drive, f"{r.drive}.csv") for r in idx.itertuples()]
    return idx[idx["csv_schema"] == "drivedna_v1"].reset_index(drop=True)


def per_second(path):
    d = pd.read_csv(path, usecols=NEEDED)
    d = d[(d["is_human"] == 1) & (d["cs_enabled"] == 0)]
    whole = np.isclose(d["time_s"] % 1.0, 0.0, atol=0.051) | np.isclose(d["time_s"] % 1.0, 1.0, atol=0.051)
    s = d[whole].copy()
    s["t"] = np.round(s["time_s"])
    s = s.drop_duplicates("t").sort_values("t")
    t = s["t"].to_numpy(float)
    v = pd.Series(np.clip(s["vEgo"].to_numpy(float), 0, None)).rolling(3, center=True, min_periods=1).mean().to_numpy()
    # derivative only across consecutive seconds (human segments can be interrupted)
    accel = np.zeros_like(v)
    if len(t) > 2:
        accel = np.gradient(v, t)
        gaps = np.diff(t) > 1.5
        bad = np.concatenate([[False], gaps]) | np.concatenate([gaps, [False]])
        accel[bad] = 0.0
    gap = np.where((s["leadOne_status"].to_numpy() > 0.5) & (s["leadOne_dRel"].to_numpy() > 0), s["leadOne_dRel"].to_numpy(), 0.0)
    wave = np.abs((s["laneLeft_y"].to_numpy() + s["laneRight_y"].to_numpy()) / 2.0)
    lane_valid = np.isfinite(wave)
    gap = np.nan_to_num(gap, nan=0.0)
    return pd.DataFrame({"t": t, "speed_kmh": v * 3.6, "accel": np.clip(accel, -9, 9), "gap_m": gap,
                         "wave_m": np.where(lane_valid, wave, 0.0), "lane_valid": lane_valid})
