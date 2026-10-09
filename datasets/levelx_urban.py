"""
inD (intersections), rounD (roundabouts) and uniD (university campus) loader: drone recordings in Germany, 25 Hz,
levelXdata (research-only licence, data not redistributed). Same file layout for the three:
    XX_recordingMeta.csv   frameRate, speedLimit (m/s), locationId
    XX_tracksMeta.csv      trackId, class (car, van, truck_bus, truck, bus, motorcycle, bicycle, pedestrian), length
    XX_tracks.csv          frame, trackId, xCenter, yCenter, xVelocity, yVelocity
No leader ids and no lane positions, so the inputs are measured the pNEUMA way (datasets/pneuma.py per_second):
    speed   |velocity| at each whole second, 3 s mean; accel = its derivative
    gap     nearest vehicle ahead in a +-1.6 m cone with a heading within 30 deg, minus half of both lengths
    wave    not measurable without lanes: 0 (three term score)
Pedestrians are dropped (never a leader); bicycles and motorcycles can be leaders but are not scored. Parked
vehicles (top speed below PARKED_KMH over the whole track) are dropped: they are not traffic.
"""
import glob
import os

import numpy as np
import pandas as pd

from datasets.pneuma import per_second as pneuma_per_second

TYPE = {"car": "Car", "van": "Medium Vehicle", "truck_bus": "Heavy Vehicle", "truck": "Heavy Vehicle",
        "bus": "Bus", "trailer": "Heavy Vehicle", "motorcycle": "Motorcycle", "bicycle": "Bicycle"}
SCORED = ("Car", "Medium Vehicle", "Heavy Vehicle", "Bus")
PARKED_KMH = 2.0


def recordings(root):
    return sorted(os.path.basename(p)[:2] for p in glob.glob(os.path.join(root, "*_tracks.csv")))


def per_second(root, rec):
    meta = pd.read_csv(os.path.join(root, f"{rec}_recordingMeta.csv")).iloc[0]
    fps = int(round(float(meta["frameRate"])))
    tm = pd.read_csv(os.path.join(root, f"{rec}_tracksMeta.csv"), usecols=["trackId", "class"])
    tm = tm[tm["class"].isin(TYPE)]
    tr = pd.read_csv(os.path.join(root, f"{rec}_tracks.csv"),
                     usecols=["frame", "trackId", "xCenter", "yCenter", "xVelocity", "yVelocity"])
    tr = tr[(tr["frame"] % fps == 0) & tr["trackId"].isin(tm["trackId"])].merge(tm, on="trackId")
    top = (np.hypot(tr["xVelocity"], tr["yVelocity"]) * 3.6).groupby(tr["trackId"]).transform("max")
    tr = tr[top >= PARKED_KMH]
    if not len(tr):
        return pd.DataFrame()
    raw = pd.DataFrame({"vehicle_id": tr["trackId"].to_numpy(), "type": tr["class"].map(TYPE).to_numpy(),
                        "t": tr["frame"].to_numpy() / fps, "x": tr["xCenter"].to_numpy(), "y": tr["yCenter"].to_numpy(),
                        "speed_kmh": np.hypot(tr["xVelocity"], tr["yVelocity"]).to_numpy() * 3.6})
    ps = pneuma_per_second(raw)
    ps = ps[ps["type"].isin(SCORED)].copy()
    limit = float(meta["speedLimit"])
    ps["vehicle_id"] = rec + "_" + ps["vehicle_id"].astype(str)
    ps["class"] = ps.pop("type")
    ps["recording"] = rec
    ps["location"] = int(meta["locationId"])
    ps["speed_limit_kmh"] = limit * 3.6 if limit > 0 else np.nan
    return ps.reset_index(drop=True)
