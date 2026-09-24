"""
NGSIM vehicle trajectory loader (US DOT, 10 Hz, imperial units).

Reads either format NGSIM is distributed in:
  * the combined CSV from data.transportation.gov, with a header row and a
    Location column (us-101, i-80, lankershim, peachtree)
  * the per-site text files (for example trajectories-0750am-0805am.txt),
    whitespace-separated with no header, 18 columns

Everything is converted to SI units. Raw NGSIM speeds and accelerations are
known to be noisy, so speed and acceleration are recomputed from the smoothed
longitudinal position rather than taken from v_Vel and v_Acc.
"""
import os

import numpy as np
import pandas as pd

FT = 0.3048
TXT_COLUMNS = ["Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time", "Local_X", "Local_Y", "Global_X",
               "Global_Y", "v_Length", "v_Width", "v_Class", "v_Vel", "v_Acc", "Lane_ID", "Preceding",
               "Following", "Space_Headway", "Time_Headway"]
_CANON = {c.lower(): c for c in TXT_COLUMNS}
_CANON.update({"space_hdwy": "Space_Headway", "time_hdwy": "Time_Headway", "location": "Location"})
NEEDED = ["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y", "v_Length", "v_Width", "Lane_ID", "Preceding",
          "Space_Headway"]
FREEWAYS = ("us-101", "i-80")


def _has_header(path):
    with open(path) as f:
        return any(ch.isalpha() for ch in f.readline())


def read_raw(path, location=None, minutes=None, chunksize=500_000):
    """Raw rows (imperial) for one location, optionally only the first `minutes` of it."""
    if _has_header(path):
        parts = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk.columns = [_CANON.get(c.strip().lower(), c.strip()) for c in chunk.columns]
            if location and "Location" in chunk.columns:
                chunk = chunk[chunk["Location"].astype(str).str.lower() == location.lower()]
            if len(chunk):
                parts.append(chunk[[c for c in NEEDED + ["Location"] if c in chunk.columns]])
        if not parts:
            raise ValueError(f"no rows for location {location!r} in {path}")
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=TXT_COLUMNS, usecols=range(18))
        df = df[NEEDED]
    df = df.drop_duplicates(["Vehicle_ID", "Frame_ID"])
    df["t"] = (df["Global_Time"] - df["Global_Time"].min()) / 1000.0
    if minutes:
        df = df[df["t"] <= minutes * 60.0]
    return df


def trajectories(raw, smooth_s=1.0):
    """SI trajectory table with columns t, vehicle_id, lane, pos, x_lat, speed, accel, gap, wave."""
    d = raw.sort_values(["Vehicle_ID", "t"]).copy()
    d["pos"] = d["Local_Y"] * FT
    d["x_lat"] = d["Local_X"] * FT
    win = max(1, int(round(smooth_s / 0.1)))
    g = d.groupby("Vehicle_ID", sort=False)
    d["pos_s"] = g["pos"].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
    d["lat_s"] = g["x_lat"].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())

    def deriv(frame, col):
        tt = frame["t"].to_numpy()
        yy = frame[col].to_numpy()
        return pd.Series(np.gradient(yy, tt) if len(tt) > 2 else np.zeros(len(tt)), index=frame.index)

    d["speed"] = g.apply(lambda f: deriv(f, "pos_s"), include_groups=False).reset_index(level=0, drop=True)
    d["speed"] = d["speed"].clip(lower=0.0)
    d["speed_s"] = d.groupby("Vehicle_ID", sort=False)["speed"].transform(
        lambda s: s.rolling(win, center=True, min_periods=1).mean())
    d["accel"] = d.groupby("Vehicle_ID", sort=False).apply(
        lambda f: deriv(f, "speed_s"), include_groups=False).reset_index(level=0, drop=True).clip(-8.0, 8.0)

    # bumper-to-bumper gap: front-to-front headway minus the length of the car ahead
    lengths = d[["Vehicle_ID", "Frame_ID", "v_Length"]].rename(
        columns={"Vehicle_ID": "Preceding", "v_Length": "lead_len"})
    d = d.merge(lengths, on=["Preceding", "Frame_ID"], how="left")
    gap = (d["Space_Headway"] - d["lead_len"].fillna(15.0)) * FT
    d["gap"] = np.where((d["Preceding"] > 0) & (d["Space_Headway"] > 0), gap.clip(lower=0.1), 0.0)

    # lateral offset from the lane centre (median lateral position of each lane)
    centers = d.groupby("Lane_ID")["lat_s"].median()
    d["wave"] = (d["lat_s"] - d["Lane_ID"].map(centers)).abs()

    return d.rename(columns={"Vehicle_ID": "vehicle_id", "Lane_ID": "lane"})[
        ["t", "vehicle_id", "lane", "pos", "x_lat", "speed", "accel", "gap", "wave"]].reset_index(drop=True)


def context_for(location):
    return "motorway" if (location or "").lower() in FREEWAYS else "secondary"
