"""
exiD loader (drone recordings of German motorway entries and exits, 25 Hz; Moers et al., IV 2022, levelXdata,
research-only licence, data not redistributed).

Files per recording XX (00 to 92):
    XX_recordingMeta.csv   frameRate, speedLimit (m/s), locationId
    XX_tracksMeta.csv      trackId, class (car, van, truck, ... and VRUs), width, length
    XX_tracks.csv          frame, trackId, xCenter, yCenter, length, lonVelocity, lonAcceleration,
                           latLaneCenterOffset, laneChange, leadId (-1 = none), leadDHW

Per second table of datasets/uah.py (t, speed_kmh, accel, gap_m, wave_m) plus vehicle_id, class, location,
speed limit, lane change flag. How each input is measured here:
    speed   |lonVelocity| at each whole second, 3 s mean (UAH way); accel = its derivative
            (exiD's own lonAcceleration kept as accel_exid for comparison)
    gap     leadDHW when a leader exists (leadId >= 0), else 0 (no car ahead)
    wave    |latLaneCenterOffset|, offset of the vehicle centre from the centre of its lane
Only motor vehicles with a car following gap are scored: car, van, truck, bus.
"""
import glob
import os

import numpy as np
import pandas as pd

SCORED = ("car", "van", "truck", "bus", "truck_bus")
COLS = ["frame", "trackId", "xCenter", "yCenter", "length", "lonVelocity", "lonAcceleration", "latLaneCenterOffset",
        "laneChange", "leadId", "leadDHW"]


def recordings(root):
    return sorted(os.path.basename(p)[:2] for p in glob.glob(os.path.join(root, "*_tracks.csv")))


def read_recording(root, rec):
    meta = pd.read_csv(os.path.join(root, f"{rec}_recordingMeta.csv")).iloc[0]
    tmeta = pd.read_csv(os.path.join(root, f"{rec}_tracksMeta.csv"), usecols=["trackId", "class"])
    tr = pd.read_csv(os.path.join(root, f"{rec}_tracks.csv"), usecols=COLS)
    return meta, tmeta, tr


def per_second(root, rec):
    """Per second table for every scored vehicle of one recording."""
    meta, tmeta, tr = read_recording(root, rec)
    fps = int(round(float(meta["frameRate"])))
    tr = tr.merge(tmeta, on="trackId")
    tr = tr[tr["class"].isin(SCORED) & (tr["frame"] % fps == 0)].sort_values(["trackId", "frame"])
    limit = float(meta["speedLimit"])
    frames = []
    for vid, g in tr.groupby("trackId", sort=False):
        t = g["frame"].to_numpy(float) / fps
        v = pd.Series(np.abs(g["lonVelocity"].to_numpy(float))).rolling(3, center=True, min_periods=1).mean().to_numpy()
        accel = np.clip(np.gradient(v, t), -9.0, 9.0) if len(t) > 2 else np.zeros_like(v)
        lead = (g["leadId"].to_numpy() >= 0) & (g["leadDHW"].to_numpy() > 0)
        frames.append(pd.DataFrame({
            "t": t, "vehicle_id": f"{rec}_{vid}", "speed_kmh": v * 3.6, "accel": accel,
            "gap_m": np.where(lead, g["leadDHW"].to_numpy(float), 0.0),
            "wave_m": np.abs(g["latLaneCenterOffset"].to_numpy(float)), "lane_change": g["laneChange"].to_numpy(),
            "accel_exid": g["lonAcceleration"].to_numpy(float), "class": g["class"].iloc[0], "recording": rec,
            "location": int(meta["locationId"]), "speed_limit_kmh": limit * 3.6 if limit > 0 else np.nan,
            "label": "unlabelled"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
