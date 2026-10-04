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
OPTIONAL = ["Int_ID", "Section_ID", "Direction", "Movement", "v_Class"]     # arterial sites (lankershim, peachtree)
FREEWAYS = ("us-101", "i-80")


def _has_header(path):
    with open(path) as f:
        return any(ch.isalpha() for ch in f.readline())


def read_raw(path, location=None, minutes=None, chunksize=500_000):
    """Raw rows (imperial) for one location, optionally only the first `minutes` of it."""
    if _has_header(path):
        parts = []
        t_min = None                       # earliest Global_Time seen so far (ms)
        # thousands=",": the data.transportation.gov CSV writes numbers like "1,759.977"
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, thousands=","):
            chunk.columns = [_CANON.get(c.strip().lower(), c.strip()) for c in chunk.columns]
            if location and "Location" in chunk.columns:
                chunk = chunk[chunk["Location"].astype(str).str.lower() == location.lower()]
            if len(chunk):
                chunk = chunk[[c for c in NEEDED + OPTIONAL + ["Location"] if c in chunk.columns]]
                if minutes:
                    # keep memory small on the 2 GB file: drop rows that are already later than
                    # the first `minutes` after the earliest time seen (t_min only ever decreases)
                    t_min = min(t_min, chunk["Global_Time"].min()) if t_min is not None else chunk["Global_Time"].min()
                    parts = [p[p["Global_Time"] <= t_min + minutes * 60_000] for p in parts]
                    chunk = chunk[chunk["Global_Time"] <= t_min + minutes * 60_000]
                parts.append(chunk)
        if not parts:
            raise ValueError(f"no rows for location {location!r} in {path}")
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=TXT_COLUMNS, usecols=range(18))
        df = df[NEEDED]
    df = _split_periods(df)
    df = df.drop_duplicates(["Vehicle_ID", "Frame_ID"])
    df["t"] = (df["Global_Time"] - df["Global_Time"].min()) / 1000.0
    if minutes:
        df = df[df["t"] <= minutes * 60.0]
    return df


def _split_periods(df):
    """Sites recorded in several periods (i-80, lankershim, peachtree) restart Frame_ID and reuse
    Vehicle_ID in each period, so (Vehicle_ID, Frame_ID) is not unique and two cars would be spliced
    into one trajectory. Global_Time - 100 ms x Frame_ID is constant within a period (its start);
    with more than one period, Vehicle_ID and Preceding become period x 100000 + id."""
    start = np.round((df["Global_Time"] - 100 * df["Frame_ID"]) / 30_000.0)    # 30 s bins of the period start
    periods = np.sort(start.unique())
    if len(periods) <= 1:
        return df
    k = pd.Series(np.searchsorted(periods, start) + 1, index=df.index)
    df = df.copy()
    df["Vehicle_ID"] = k * 100_000 + df["Vehicle_ID"]
    df["Preceding"] = np.where(df["Preceding"] > 0, k * 100_000 + df["Preceding"], 0)
    return df


def trajectories(raw, smooth_s=1.0, planar=False):
    """SI trajectory table with columns t, vehicle_id, lane, pos, x_lat, speed, accel, gap, wave.

    planar=False (freeways): speed from Local_Y only, the lane centre is the median lateral
    position of each Lane_ID. planar=True (arterials, where cars drive both ways along Local_Y
    and turn at intersections): speed from the 2-D path, lane centre per (Direction, Section_ID,
    Lane_ID); Int_ID, Section_ID, Direction and Movement are kept for filtering."""
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
    if planar:
        vx = g.apply(lambda f: deriv(f, "lat_s"), include_groups=False).reset_index(level=0, drop=True)
        d["speed"] = np.hypot(d["speed"], vx)
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
    keys = ["Lane_ID"] + ([c for c in ("Direction", "Section_ID") if c in d.columns] if planar else [])
    d["wave"] = (d["lat_s"] - d.groupby(keys)["lat_s"].transform("median")).abs()

    d["accel_1hz"] = _accel_1hz(d)

    extra = [c for c in OPTIONAL if c in d.columns] if planar else []
    return d.rename(columns={"Vehicle_ID": "vehicle_id", "Lane_ID": "lane"})[
        ["t", "vehicle_id", "lane", "pos", "x_lat", "speed", "accel", "accel_1hz", "gap", "wave"] + extra].reset_index(drop=True)


def _accel_1hz(d):
    """Acceleration measured the way UAH-DriveSet measures it (datasets/uah.py load_trip):
    speed once per second, smoothed over 3 seconds, then its time derivative. The index
    uses this one, so the agent sees NGSIM acceleration on the same scale it was trained
    on. The 10 Hz "accel", twice differentiated from video-tracked positions, is several
    times larger and is kept for the shockwave factor."""
    key = pd.DataFrame({"vid": d["Vehicle_ID"].to_numpy(), "sec": np.floor(d["t"].to_numpy()).astype(int),
                        "v": d["speed"].to_numpy()})
    ps = key.groupby(["vid", "sec"], sort=True)["v"].mean().reset_index()
    ps["v"] = ps.groupby("vid")["v"].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())

    def slope(g):
        v, sec = g["v"].to_numpy(), g["sec"].to_numpy(dtype=float)
        return pd.Series(np.gradient(v, sec) if len(g) > 1 else np.zeros(len(g)), index=g.index)

    ps["a"] = ps.groupby("vid", group_keys=False).apply(slope).clip(-9.0, 9.0)
    return key.merge(ps[["vid", "sec", "a"]], on=["vid", "sec"], how="left")["a"].fillna(0.0).to_numpy()


def environment_for(location):
    """Freeway sites map to the agent's highway environment, arterials to urban."""
    return "highway" if (location or "").lower() in FREEWAYS else "urban"


def per_second(traj):
    """The per second table of datasets/uah.py (t, speed_kmh, accel, gap_m, wave_m) for every vehicle,
    from a trajectories() table: the row at each whole second, accel = accel_1hz (measured like UAH)."""
    s = traj[np.isclose(traj["t"] % 1.0, 0.0, atol=0.05)].copy()
    s["t"] = np.round(s["t"])
    s = s.drop_duplicates(["vehicle_id", "t"])
    out = pd.DataFrame({"t": s["t"], "vehicle_id": s["vehicle_id"], "speed_kmh": s["speed"] * 3.6,
                        "accel": s["accel_1hz"], "gap_m": s["gap"], "wave_m": s["wave"], "lane": s["lane"],
                        "pos": s["pos"], "label": "unlabelled"})
    for c in OPTIONAL:
        if c in s.columns:
            out[c] = s[c].to_numpy()
    return out.reset_index(drop=True)
