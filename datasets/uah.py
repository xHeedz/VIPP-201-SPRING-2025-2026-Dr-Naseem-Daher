"""
UAH-DriveSet loader (Romera, Bergasa and Arroyo, ITSC 2016).

Expected layout, as downloaded:
    UAH-DRIVESET-v1/D1/20151110175712-16km-D1-NORMAL1-SECONDARY/RAW_GPS.txt
                                                                PROC_LANE_DETECTION.txt
                                                                PROC_VEHICLE_DETECTION.txt
Column meanings follow the official reader (github.com/Eromera/uah_driveset_reader);
files are space-separated with the timestamp in seconds in column 0:
    RAW_GPS                 1 speed (km/h)
    PROC_LANE_DETECTION     1 car position from lane center (m), 3 road width (m)
    PROC_VEHICLE_DETECTION  1 distance to vehicle ahead (m, negative when none)

Per second of each trip this produces speed, acceleration (from GPS speed),
gap to the vehicle ahead and lateral offset from the lane center, which are
the four inputs of the Aggressiveness Index.
"""
import os
import re

import numpy as np
import pandas as pd

# UAH road types mapped onto the agent's environments (UAH has no weather trips)
ROAD_TO_ENV = {"motorway": "highway", "secondary": "urban"}

TRIP_RE = re.compile(r"(?P<date>\d{14})-(?:(?P<km>[\d.]+)km-)?(?P<driver>D\d+)-"
                     r"(?P<behavior>NORMAL\d?|AGGRESSIVE|DROWSY)-(?P<road>MOTORWAY|SECONDARY)", re.I)


def find_trips(root):
    trips = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            m = TRIP_RE.search(d)
            if m:
                trips.append({
                    "path": os.path.join(dirpath, d),
                    "trip": d,
                    "driver": m.group("driver").upper(),
                    "behavior": re.sub(r"\d", "", m.group("behavior").lower()),
                    "road": m.group("road").lower(),
                    "environment": ROAD_TO_ENV[m.group("road").lower()],
                })
    if not trips:
        raise FileNotFoundError(f"no UAH-DriveSet trip folders found under {root}")
    return sorted(trips, key=lambda t: (t["driver"], t["road"], t["behavior"], t["trip"]))


def _read(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return df if len(df) else None


def load_trip(trip):
    """Per-second table: t, speed_kmh, accel, gap_m, wave_m, lane_valid."""
    gps = _read(os.path.join(trip["path"], "RAW_GPS.txt"))
    if gps is None or gps.shape[1] < 2:
        raise ValueError(f"RAW_GPS.txt missing or malformed in {trip['path']}")
    gps = gps[[0, 1]].rename(columns={0: "t", 1: "speed_kmh"}).sort_values("t").drop_duplicates("t")
    gps = gps[(gps["speed_kmh"] >= 0) & (gps["speed_kmh"] < 250)]
    t = gps["t"].to_numpy(float)
    v = pd.Series(gps["speed_kmh"].to_numpy(float) / 3.6).rolling(3, center=True, min_periods=1).mean().to_numpy()
    accel = np.gradient(v, t) if len(t) > 2 else np.zeros_like(v)
    accel = np.clip(accel, -9.0, 9.0)

    out = pd.DataFrame({"t": t, "speed_kmh": v * 3.6, "accel": accel, "gap_m": 0.0, "wave_m": 0.0,
                        "lane_valid": False})

    lane = _read(os.path.join(trip["path"], "PROC_LANE_DETECTION.txt"))
    if lane is not None and lane.shape[1] >= 4:
        ok = lane[(lane[3] > 0) & (lane[1].abs() <= 2.0)]
        if len(ok):
            idx = np.searchsorted(ok[0].to_numpy(), t).clip(0, len(ok) - 1)
            near = np.abs(ok[0].to_numpy()[idx] - t) <= 1.0
            out.loc[near, "wave_m"] = np.abs(ok[1].to_numpy()[idx][near])
            out.loc[near, "lane_valid"] = True

    veh = _read(os.path.join(trip["path"], "PROC_VEHICLE_DETECTION.txt"))
    if veh is not None and veh.shape[1] >= 2:
        vt, vd = veh[0].to_numpy(float), veh[1].to_numpy(float)
        idx = np.searchsorted(vt, t).clip(0, len(vt) - 1)
        near = np.abs(vt[idx] - t) <= 1.5
        out.loc[near, "gap_m"] = np.where(vd[idx][near] > 0, vd[idx][near], 0.0)
    return out


def windows(per_second, trip, length_s=10.0, step_s=5.0):
    """Mean index features over sliding windows; one row per window."""
    from model.aggressiveness_model import index_features
    f = index_features(per_second["speed_kmh"], per_second["accel"], per_second["gap_m"], per_second["wave_m"])
    t = per_second["t"].to_numpy()
    rows = []
    start = t[0] if len(t) else 0.0
    while len(t) and start + length_s <= t[-1]:
        m = (t >= start) & (t < start + length_s)
        if m.sum() >= length_s * 0.6:
            mean = f[m].mean(axis=0)
            rows.append({"trip": trip["trip"], "driver": trip["driver"], "road": trip["road"],
                         "environment": trip["environment"],
                         "behavior": trip["behavior"], "t0": start,
                         "phi_speed": mean[0], "phi_accel": mean[1], "phi_prox": mean[2], "phi_wave": mean[3],
                         "speed_kmh": per_second["speed_kmh"].to_numpy()[m].mean()})
        start += step_s
    return pd.DataFrame(rows)


def load_windows(root, length_s=10.0, step_s=5.0, min_speed_kmh=5.0):
    """All trips -> window table, dropping windows where the car is essentially stopped."""
    frames, report = [], []
    for trip in find_trips(root):
        ps = load_trip(trip)
        w = windows(ps, trip, length_s, step_s)
        if len(w):
            w = w[w["speed_kmh"] >= min_speed_kmh]
        frames.append(w)
        report.append({"trip": trip["trip"], "seconds": len(ps), "windows": len(w),
                       "lane_coverage": float(ps["lane_valid"].mean()) if len(ps) else 0.0,
                       "leader_coverage": float((ps["gap_m"] > 0).mean()) if len(ps) else 0.0})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(report)
