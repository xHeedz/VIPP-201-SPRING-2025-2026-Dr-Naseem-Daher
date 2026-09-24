"""
Synthetic stand-ins for UAH-DriveSet and NGSIM, written in the real file
formats. They exist only to test that the loaders and scripts run end to end;
numbers produced from them say nothing about real drivers.
"""
import os

import numpy as np
import pandas as pd

BEHAVIOR = {  # speed offset (km/h), accel spread (m/s^2), mean gap (m), lane wander (m)
    "NORMAL": (0, 0.6, 35, 0.25), "NORMAL1": (0, 0.6, 35, 0.25), "NORMAL2": (0, 0.6, 35, 0.25),
    "AGGRESSIVE": (18, 1.6, 14, 0.45), "DROWSY": (-12, 0.4, 45, 0.7),
}
ROAD = {"MOTORWAY": (110, "25km"), "SECONDARY": (70, "16km")}


def make_uah(root, seconds=240, seed=0):
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "UAH-DRIVESET-v1")
    for d in range(1, 7):
        drv = rng.normal(0, 4)
        for road, (v0, km) in ROAD.items():
            behaviors = ["NORMAL", "AGGRESSIVE", "DROWSY"] if road == "MOTORWAY" else ["NORMAL1", "NORMAL2", "AGGRESSIVE", "DROWSY"]
            for b in behaviors:
                dv, acc, gap, wander = BEHAVIOR[b]
                name = f"20151{d:02d}10175712-{km}-D{d}-{b}-{road}"
                p = os.path.join(base, f"D{d}", name)
                os.makedirs(p, exist_ok=True)
                t = np.arange(seconds, dtype=float)
                v = np.clip(v0 + dv + drv + np.cumsum(rng.normal(0, acc * 3.6, seconds)) * 0.3, 5, 200)
                gps = np.column_stack([t, v, 40.5 + t * 1e-5, -3.3 + t * 1e-5, 600 + 0 * t, 3 + 0 * t, 5 + 0 * t,
                                       90 + 0 * t, 0 * t, 1 + 0 * t, 0 * t, 0 * t])
                np.savetxt(os.path.join(p, "RAW_GPS.txt"), gps, fmt="%.4f")
                tl = np.arange(0, seconds, 0.2)
                x = np.clip(np.cumsum(rng.normal(0, wander * 0.1, len(tl))) * 0.3, -1.4, 1.4)
                lane = np.column_stack([tl, x, 0 * tl, 3.5 + 0 * tl, 1 + 0 * tl])
                np.savetxt(os.path.join(p, "PROC_LANE_DETECTION.txt"), lane, fmt="%.4f")
                tv = np.arange(0, seconds, 0.5)
                dist = np.where(rng.random(len(tv)) < 0.7, np.clip(rng.normal(gap, 6, len(tv)), 2, 90), -1)
                veh = np.column_stack([tv, dist, np.where(dist > 0, dist / 20, -1), (dist > 0).astype(int), 0 * tv + v0])
                np.savetxt(os.path.join(p, "PROC_VEHICLE_DETECTION.txt"), veh, fmt="%.4f")
    return base


def make_ngsim(path, seconds=90, n_lanes=3, per_lane=12, header=True, seed=0):
    """Platoons per lane with a slow vehicle in lane 1 to create a disturbance."""
    rng = np.random.default_rng(seed)
    rows, vid = [], 1
    for lane in range(1, n_lanes + 1):
        ids = list(range(vid, vid + per_lane))
        vid += per_lane
        pos = np.array([600.0 - 60.0 * k for k in range(per_lane)])          # ft, leader first
        spd = np.full(per_lane, 90.0 if lane > 1 else 60.0)                   # ft/s
        for f in range(int(seconds * 10)):
            for k, v in enumerate(ids):
                if k > 0:
                    gap = pos[k - 1] - pos[k] - 15.0
                    spd[k] += 0.1 * (min(spd[k - 1] + (gap - 40) * 0.3, 100.0) - spd[k]) + rng.normal(0, 0.5)
                else:
                    spd[k] += rng.normal(0, 0.3)
                spd[k] = max(spd[k], 0.0)
                pos[k] += spd[k] * 0.1
                prec = ids[k - 1] if k > 0 else 0
                sh = pos[k - 1] - pos[k] if k > 0 else 0.0
                x = 6.0 + 12.0 * (lane - 1) + rng.normal(0, 0.3)
                rows.append([v, f, int(seconds * 10), 1118846980200 + f * 100, x, pos[k], 0, 0, 15.0, 6.0, 2,
                             spd[k], 0.0, lane, prec, 0, sh, sh / max(spd[k], 1)])
    cols = ["Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time", "Local_X", "Local_Y", "Global_X", "Global_Y",
            "v_length", "v_Width", "v_Class", "v_Vel", "v_Acc", "Lane_ID", "Preceding", "Following",
            "Space_Headway", "Time_Headway"]
    df = pd.DataFrame(rows, columns=cols)
    if header:
        df["Location"] = "us-101"
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, sep=" ", header=False, index=False)
    return path
