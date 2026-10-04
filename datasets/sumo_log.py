"""
SUMO raw logs (10 Hz TraCI values, scripts/planted_drivers.py) reduced to the same
per second table as datasets/uah.py load_trip, so SUMO, UAH and NGSIM share one
feature path (index_features / windows).

Raw log columns: t, vehicle_id, label, speed_ms, accel_ms2, gap_m, lat_m, lane, edge.

Per second, the UAH way: speed is sampled once per second (UAH: GPS speed at 1 Hz),
smoothed over 3 s, and acceleration is its time derivative (not SUMO's exact
acceleration); gap and lane offset are the values at the same instant.
"""
import numpy as np
import pandas as pd


def per_second(raw):
    """raw 10 Hz rows of ONE vehicle -> t, speed_kmh, accel, gap_m, wave_m, lane, edge."""
    r = raw.sort_values("t")
    whole = np.isclose(r["t"].to_numpy() % 1.0, 0.0, atol=1e-6) | np.isclose(r["t"].to_numpy() % 1.0, 1.0, atol=1e-6)
    s = r[whole]
    t = np.round(s["t"].to_numpy(float))
    v = pd.Series(s["speed_ms"].to_numpy(float)).rolling(3, center=True, min_periods=1).mean().to_numpy()
    accel = np.clip(np.gradient(v, t), -9.0, 9.0) if len(t) > 2 else np.zeros_like(v)
    return pd.DataFrame({"t": t, "speed_kmh": v * 3.6, "accel": accel,
                         "gap_m": s["gap_m"].to_numpy(float), "wave_m": np.abs(s["lat_m"].to_numpy(float)),
                         "lane": s["lane"].to_numpy(), "edge": s["edge"].to_numpy()})


def windows_from_per_second(ps, scenario, environment, length_s=10.0, step_s=5.0, min_speed_kmh=5.0,
                            features=None):
    """Window table (datasets/uah.py windows) from a per second table holding many vehicles
    (columns of per_second plus vehicle_id and label)."""
    from datasets.uah import windows
    frames = []
    for vid, g in ps.groupby("vehicle_id", sort=False):
        if len(g) < length_s:
            continue
        trip = {"trip": vid, "driver": vid, "road": scenario, "environment": environment,
                "behavior": g["label"].iloc[0]}
        w = windows(g.reset_index(drop=True), trip, length_s, step_s, features)
        if len(w):
            frames.append(w[w["speed_kmh"] >= min_speed_kmh])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def vehicle_windows(raw, scenario, environment, length_s=10.0, step_s=5.0, min_speed_kmh=5.0, features=None):
    """Window table for every vehicle in a raw 10 Hz log."""
    ps = pd.concat([per_second(g).assign(vehicle_id=v, label=g["label"].iloc[0])
                    for v, g in raw.groupby("vehicle_id", sort=False)], ignore_index=True)
    return windows_from_per_second(ps, scenario, environment, length_s, step_s, min_speed_kmh, features)
