"""
highD loader (drone recordings of German motorways, 25 Hz; Krajewski et al., ITSC 2018, levelXdata,
research-only licence, data not redistributed).

Files per recording XX (01 to 60):
    XX_recordingMeta.csv   frameRate, speedLimit (m/s, -1 = none), upperLaneMarkings, lowerLaneMarkings ("y1;y2;...")
    XX_tracksMeta.csv      id, class (Car / Truck), drivingDirection (1 = to the left, 2 = to the right)
    XX_tracks.csv          frame, id, x, y (top left corner of the box, m), width (= length), height (= width),
                           xVelocity, yVelocity, xAcceleration, precedingId (0 = none), dhw, laneId

Per second table of datasets/uah.py (t, speed_kmh, accel, gap_m, wave_m) plus vehicle_id, class, lane,
speed limit. How each input is measured here:
    speed   |velocity| at each whole second, 3 s mean (UAH way); accel = its derivative
    gap     own front bumper to the leader's rear bumper (precedingId), from the boxes; equals highD's dhw to
            within 1 cm on recording 01 (10,481 vehicle-seconds), kept as dhw_m; no leader = 0 (no car ahead)
    wave    |box centre y - centre of the lane it is in|, lanes from the lane markings
"""
import glob
import os

import numpy as np
import pandas as pd


def recordings(root):
    return sorted(os.path.basename(p)[:2] for p in glob.glob(os.path.join(root, "*_tracks.csv")))


def _markings(text):
    return [float(v) for v in str(text).split(";") if v.strip()]


def lane_offset(yc, markings):
    """|yc - centre of the lane between consecutive markings that contains yc|; nan outside every lane."""
    m = np.sort(np.asarray(markings, float))
    k = np.searchsorted(m, yc) - 1
    ok = (k >= 0) & (k < len(m) - 1)
    kk = np.clip(k, 0, len(m) - 2)
    return np.where(ok, np.abs(yc - (m[kk] + m[kk + 1]) / 2.0), np.nan)


def read_recording(root, rec):
    meta = pd.read_csv(os.path.join(root, f"{rec}_recordingMeta.csv")).iloc[0]
    tmeta = pd.read_csv(os.path.join(root, f"{rec}_tracksMeta.csv"), usecols=["id", "class", "drivingDirection"])
    cols = ["frame", "id", "x", "y", "width", "height", "xVelocity", "yVelocity", "precedingId", "dhw", "laneId"]
    tr = pd.read_csv(os.path.join(root, f"{rec}_tracks.csv"), usecols=cols)
    return meta, tmeta, tr


def per_second(root, rec):
    """Per second table for every vehicle of one recording."""
    meta, tmeta, tr = read_recording(root, rec)
    fps = int(round(float(meta["frameRate"])))
    tr = tr.merge(tmeta, on="id")
    # gap: leader's box in the same frame
    lead = tr[["frame", "id", "x", "width"]].rename(columns={"id": "precedingId", "x": "lx", "width": "lw"})
    tr = tr.merge(lead, on=["frame", "precedingId"], how="left")
    to_left = tr["drivingDirection"] == 1
    front = np.where(to_left, tr["x"], tr["x"] + tr["width"])
    lead_rear = np.where(to_left, tr["lx"] + tr["lw"], tr["lx"])
    gap = np.where(to_left, front - lead_rear, lead_rear - front)
    tr["gap_m"] = np.where((tr["precedingId"] > 0) & np.isfinite(gap) & (gap > 0), gap, 0.0)
    # wave: offset from the centre of the lane, markings of the vehicle's carriageway
    yc = (tr["y"] + tr["height"] / 2.0).to_numpy()
    upper, lower = _markings(meta["upperLaneMarkings"]), _markings(meta["lowerLaneMarkings"])
    tr["wave_m"] = np.where(to_left, lane_offset(yc, upper), lane_offset(yc, lower))
    tr["speed_ms"] = np.hypot(tr["xVelocity"], tr["yVelocity"])
    s = tr[(tr["frame"] % fps) == 0].sort_values(["id", "frame"])
    frames = []
    for vid, g in s.groupby("id", sort=False):
        t = g["frame"].to_numpy(float) / fps
        v = pd.Series(g["speed_ms"].to_numpy(float)).rolling(3, center=True, min_periods=1).mean().to_numpy()
        accel = np.clip(np.gradient(v, t), -9.0, 9.0) if len(t) > 2 else np.zeros_like(v)
        frames.append(pd.DataFrame({
            "t": t, "vehicle_id": f"{rec}_{vid}", "speed_kmh": v * 3.6, "accel": accel,
            "gap_m": g["gap_m"].to_numpy(), "wave_m": np.nan_to_num(g["wave_m"].to_numpy(), nan=0.0),
            "lane_valid": np.isfinite(g["wave_m"].to_numpy()), "lane": g["laneId"].to_numpy(),
            "direction": int(g["drivingDirection"].iloc[0]),
            "dhw_m": g["dhw"].to_numpy(), "class": g["class"].iloc[0], "recording": rec,
            "speed_limit_kmh": float(meta["speedLimit"]) * 3.6 if float(meta["speedLimit"]) > 0 else np.nan,
            "label": "unlabelled"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
