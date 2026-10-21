"""
Audit of every per second table before it is used: the same checks on every dataset, nothing changed.

    python scripts/audit_datasets.py [--only uah,planted,...]

Sources (all next to the repo): UAH (40 trips), planted SUMO (70 runs), NGSIM (US-101, I-80, Lankershim,
Peachtree), pNEUMA (10 drones), highD, exiD, inD, rounD, uniD (cached per second tables), DriveDNA sample.
Each source is reduced to: vid, t, speed_kmh, accel, gap_m, wave_m (+ lane_valid where the source has it).

Checks per source (shares of rows unless stated):
  rows, vehicles, median seconds per vehicle, duplicate (vid, t), time breaks (next row not 1 s later)
  speed: NaN, above 200 km/h, stopped (below 5 km/h)
  accel: NaN, |a| above 4 (hard), above 8 (implausible for a car), exactly at a clip value (+-9),
         mismatch with the speed change (|a - dv/dt| above 1.5 m/s2, consecutive seconds only)
  gap: exactly 0 (no car ahead OR not detected: the same value), 0 < gap < 2 m while moving above 20 km/h,
       time headway below 0.3 s while moving above 20 km/h, jumps above 30 m in one second (leader change)
  wave: exactly 0 (centred OR not detected), above 2 m (outside a lane), lane_valid share where available
Outputs data/dataset_audit.csv (one row per source) and data/dataset_audit_columns.csv (percentiles).
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402

EXT = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026"))
EXT2 = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "12-10-2026"))
COLS = ["vid", "t", "speed_kmh", "accel", "gap_m", "wave_m"]


# ── loaders: every source to the same columns ───────────────────────────────
def uah():
    from datasets.uah import find_trips, load_trip
    parts = []
    for trip in find_trips(os.path.join(EXT, "UAH-DRIVESET-v1")):
        ps = load_trip(trip)
        parts.append(ps.assign(vid=trip["trip"],
                               label=trip.get("behavior", "")))
    return pd.concat(parts, ignore_index=True)


def planted():
    parts = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", "*_s*.per_second.csv.gz"))):
        if "us101" in os.path.basename(f):
            continue
        d = pd.read_csv(f)
        parts.append(d.assign(vid=os.path.basename(f).split(".")[0] + "/" + d["vehicle_id"].astype(str)))
    return pd.concat(parts, ignore_index=True)


def ngsim(site):
    from score_unlabelled import SITES, ngsim_per_second
    name, file, loc, minutes, road, planar = [s for s in SITES if s[2] == site][0]
    ps = ngsim_per_second(file, loc, minutes, planar)
    return ps.rename(columns={"trip": "vid"}) if "trip" in ps else ps.assign(vid=ps["vehicle_id"])


def pneuma():
    from score_unlabelled import pneuma_per_second
    return pneuma_per_second().assign(vid=lambda d: d["vehicle_id"])


def levelx(name):
    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(EXT2, name, "per_second", "*.csv.gz")))]
    d = pd.concat(parts, ignore_index=True)
    return d.assign(vid=d["vehicle_id"].astype(str))


def drivedna():
    from datasets.drivedna import find_drives, per_second
    root = os.path.join(EXT2, "DriveDNA-Sample")
    parts = []
    for r in find_drives(root).itertuples():
        try:
            parts.append(per_second(r.path).assign(vid=f"{r.driver}/{r.drive}"))
        except Exception as e:  # noqa: BLE001
            print("  skipped", r.path, e)
    return pd.concat(parts, ignore_index=True)


SOURCES = {"uah": uah, "planted": planted,
           "ngsim us-101": lambda: ngsim("us-101"), "ngsim i-80": lambda: ngsim("i-80"),
           "ngsim lankershim": lambda: ngsim("lankershim"), "ngsim peachtree": lambda: ngsim("peachtree"),
           "pneuma": pneuma, "highD": lambda: levelx("highD"), "exiD": lambda: levelx("exiD"),
           "inD": lambda: levelx("inD"), "rounD": lambda: levelx("rounD"), "uniD": lambda: levelx("uniD"),
           "drivedna sample": drivedna}


# ── checks ──────────────────────────────────────────────────────────────────
def audit(name, d):
    d = d.sort_values(["vid", "t"]).reset_index(drop=True)
    n = len(d)
    same = d["vid"].eq(d["vid"].shift(-1))
    dt = d["t"].shift(-1) - d["t"]
    consec = same & np.isclose(dt, 1.0, atol=0.05)
    v = d["speed_kmh"] / 3.6
    dvdt = (v.shift(-1) - v.shift(1)) / 2.0
    both = consec & consec.shift(1, fill_value=False)
    moving = d["speed_kmh"] > 20
    gap = d["gap_m"]
    thw = np.where((gap > 0) & (v > 0.5), gap / v.clip(lower=0.5), np.nan)
    r = dict(source=name, rows=n, vehicles=d["vid"].nunique(),
             median_s_per_vehicle=float(d.groupby("vid").size().median()),
             duplicate_rows=float(d.duplicated(["vid", "t"]).mean()),
             time_breaks=float((same & ~consec).sum() / max(same.sum(), 1)),
             speed_nan=float(d["speed_kmh"].isna().mean()), speed_above_200=float((d["speed_kmh"] > 200).mean()),
             stopped=float((d["speed_kmh"] < 5).mean()),
             accel_nan=float(d["accel"].isna().mean()), accel_above_4=float((d["accel"].abs() > 4).mean()),
             accel_above_8=float((d["accel"].abs() > 8).mean()),
             accel_at_clip=float(np.isclose(d["accel"].abs(), 9.0).mean()),
             accel_mismatch=float(((d["accel"] - dvdt).abs() > 1.5)[both].mean()) if both.any() else np.nan,
             gap_zero=float((gap == 0).mean()), gap_nan=float(gap.isna().mean()),
             gap_under_2m_moving=float(((gap > 0) & (gap < 2) & moving).mean()),
             thw_under_0p3_moving=float(((thw < 0.3) & moving).mean()),
             gap_jump_30m=float(((gap.shift(-1) - gap).abs() > 30)[consec & (gap > 0) & (gap.shift(-1) > 0)].mean()),
             wave_zero=float((d["wave_m"] == 0).mean()), wave_nan=float(d["wave_m"].isna().mean()),
             wave_above_2m=float((d["wave_m"] > 2).mean()),
             lane_valid=float(d["lane_valid"].mean()) if "lane_valid" in d else np.nan)
    pct = []
    for c in ["speed_kmh", "accel", "gap_m", "wave_m"]:
        x = d[c].dropna()
        x = x[x != 0] if c in ("gap_m", "wave_m") else x
        q = x.quantile([0.01, 0.5, 0.99]).to_numpy() if len(x) else [np.nan] * 3
        pct.append(dict(source=name, column=c, p1=q[0], median=q[1], p99=q[2], min=x.min() if len(x) else np.nan,
                        max=x.max() if len(x) else np.nan, note="nonzero only" if c in ("gap_m", "wave_m") else ""))
    return r, pct


def main(only):
    rows, cols = [], []
    for name, fn in SOURCES.items():
        if only and name not in only:
            continue
        print(f"{name} ...", flush=True)
        d = fn()
        if "vid" not in d:
            d = d.assign(vid=d["vehicle_id"])
        r, p = audit(name, d)
        rows.append(r)
        cols += p
        print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items() if k != "source"),
              flush=True)
    out = os.path.join(DATA_DIR, "dataset_audit.csv" if not only else "dataset_audit_partial.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(cols).to_csv(out.replace(".csv", "_columns.csv"), index=False)
    print("written", out)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="", help="comma separated source names")
    a = p.parse_args()
    main([s.strip() for s in a.only.split(",") if s.strip()])
