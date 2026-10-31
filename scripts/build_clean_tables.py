"""
Clean every source once (datasets/cleaning.py) and store it, so every model reads exactly the same data.

    python scripts/build_clean_tables.py [--only uah,planted,...]

Writes data/clean/<source>.csv.gz with: source, vid, group (driver, seed or recording: the split unit),
label (normal / aggressive / conservative, or unlabelled), environment (highway, urban, weather), scenario,
t, speed_kmh, accel, gap_m, wave_m, leader (+ density where the source has it), and data/cleaning_log.csv
(rows touched by each rule, per source). data/clean/ is git-ignored (large).
"""
import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402
from datasets.cleaning import clean  # noqa: E402
import audit_datasets as A  # noqa: E402

OUT = os.path.join(DATA_DIR, "clean")
KEEP = ["source", "vid", "group", "label", "environment", "scenario", "t", "speed_kmh", "accel", "gap_m", "wave_m",
        "leader", "density", "speed_limit_kmh", "traffic_kmh"]
# speed limits: SUMO edges from scripts/planted_drivers.py; NGSIM and pNEUMA posted limits (assumed, not in the files):
# US-101 and I-80 65 mph, Lankershim and Peachtree 35 mph, Athens urban 50 km/h
EDGE_LIMIT = {"hw": 120.0, "main": 100.0, "acc": 100.0, "ramp": 80.0, "r_": 30.0, "_in": 50.0, "_out": 50.0,
              "abl": 120.0, "abf": -1.0}         # autobahn: abf has no speed limit (-1, becomes NaN)
NGSIM_LIMIT = {"us-101": 104.6, "i-80": 104.6, "lankershim": 56.3, "peachtree": 56.3}
PLANTED_ENV = {"highway": "highway", "jam": "highway", "merge": "highway", "weather": "weather",
               "urban": "urban", "roundabout": "urban", "autobahn": "highway"}


def uah():
    from datasets.uah import find_trips, load_trip
    parts = []
    for trip in find_trips(os.path.join(A.EXT, "UAH-DRIVESET-v1")):
        ps = load_trip(trip).assign(vid=trip["trip"], group=trip["driver"], label=trip["behavior"],
                                    environment=trip["environment"], scenario=trip["road"])
        osm_path = os.path.join(trip["path"], "PROC_OPENSTREETMAP_DATA.txt")
        lim = np.full(len(ps), np.nan)
        if os.path.exists(osm_path):          # column 2: max speed of the current OpenStreetMap way (-1 unknown)
            osm = pd.read_csv(osm_path, sep=r"\s+", header=None)
            osm = osm[osm[1] > 0]
            if len(osm):
                i = np.searchsorted(osm[0].to_numpy(), ps["t"].to_numpy()).clip(0, len(osm) - 1)
                near = np.abs(osm[0].to_numpy()[i] - ps["t"].to_numpy()) <= 5.0
                lim = np.where(near, osm[1].to_numpy()[i], np.nan)
        parts.append(ps.assign(speed_limit_kmh=lim))
    return pd.concat(parts, ignore_index=True)


def edge_limit(edge):
    for k, v in EDGE_LIMIT.items():
        if k in str(edge) and not str(edge).startswith(":"):
            return v
    return np.nan


def add_traffic(d, key):
    """mean speed of the other vehicles with the same key at the same second (NaN when alone)"""
    k = [key, d["t"].round()]
    tot = d.groupby(k)["speed_kmh"].transform("sum")
    cnt = d.groupby(k)["speed_kmh"].transform("count")
    return ((tot - d["speed_kmh"]) / (cnt - 1)).where(cnt > 1)


def planted(folder="sumo_planted"):
    if folder == "sumo_planted":
        d = A.planted()
    else:
        import glob
        parts = []
        for f in sorted(glob.glob(os.path.join(DATA_DIR, folder, "*_s*.per_second.csv.gz"))):
            x = pd.read_csv(f)
            parts.append(x.assign(vid=os.path.basename(f).split(".")[0] + "/" + x["vehicle_id"].astype(str)))
        d = pd.concat(parts, ignore_index=True)
    run = d["vid"].str.split("/").str[0]
    m = run.str.extract(r"(?P<scen>[a-z0-9]+)_(?P<dens>[a-z]+)_s(?P<seed>\d+)")
    lim = d["edge"].map({e: edge_limit(e) for e in d["edge"].unique()})
    lim = lim.groupby(d["vid"]).transform(lambda x: x.ffill().bfill())       # internal junction edges
    lim = lim.where(lim > 0)                                                 # -1: no speed limit
    d = d.assign(group=m["seed"].astype(int), scenario=m["scen"] + "_" + m["dens"],
                 environment=m["scen"].map(PLANTED_ENV), speed_limit_kmh=lim)
    return d.assign(traffic_kmh=add_traffic(d, run))


def ngsim(site, env):
    def f():
        d = A.ngsim(site)
        d = d.assign(group=site, label="unlabelled", environment=env, scenario=site, speed_limit_kmh=NGSIM_LIMIT[site])
        return d.assign(traffic_kmh=add_traffic(d, d["group"]))
    return f


def unlabelled(fn, name, env):
    def f():
        d = fn()
        rec = d["recording"].astype(str) if "recording" in d else name
        dens = d["density_50m"] if "density_50m" in d else d.get("density")
        if "speed_limit_kmh" in d:
            lim = d["speed_limit_kmh"].where(d["speed_limit_kmh"] > 0)     # levelX: -1 = no limit
        else:
            lim = 50.0                                                     # pNEUMA, Athens urban (assumed)
        key = rec if "recording" in d else (d["vehicle_id"] // 100000)     # pNEUMA: one drone per recording
        d = d.assign(group=rec, label="unlabelled", environment=env, scenario=name, density=dens, speed_limit_kmh=lim)
        return d.assign(traffic_kmh=add_traffic(d, key))
    return f


def drivedna():
    d = A.drivedna()
    return d.assign(group=d["vid"].str.split("/").str[0], label="unlabelled", environment="highway", scenario="drivedna",
                    speed_limit_kmh=np.nan, traffic_kmh=np.nan)


BUILD = {"uah": uah, "planted": planted, "planted_de": lambda: planted("sumo_planted_de"),
         "ngsim us-101": ngsim("us-101", "highway"), "ngsim i-80": ngsim("i-80", "highway"),
         "ngsim lankershim": ngsim("lankershim", "urban"), "ngsim peachtree": ngsim("peachtree", "urban"),
         "pneuma": unlabelled(A.pneuma, "pneuma", "urban"),
         "highD": unlabelled(lambda: A.levelx("highD"), "highD", "highway"),
         "exiD": unlabelled(lambda: A.levelx("exiD"), "exiD", "highway"),
         "inD": unlabelled(lambda: A.levelx("inD"), "inD", "urban"),
         "rounD": unlabelled(lambda: A.levelx("rounD"), "rounD", "urban"),
         "uniD": unlabelled(lambda: A.levelx("uniD"), "uniD", "urban"),
         "drivedna sample": drivedna}


def fname(source):
    return os.path.join(OUT, re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_") + ".csv.gz")


def main(only):
    os.makedirs(OUT, exist_ok=True)
    logs = []
    for name, fn in BUILD.items():
        if only and name not in only:
            continue
        print(f"{name} ...", flush=True)
        d = fn()
        if "vid" not in d:
            d = d.assign(vid=d["vehicle_id"])
        if "traffic_kmh" not in d:
            d["traffic_kmh"] = np.nan                                      # one car recorded (UAH)
        d, log = clean(d, "planted" if name == "planted_de" else (name.split(" ")[0] if name.startswith(("ngsim", "drivedna")) else name))
        d = d.assign(source=name)
        if "density" not in d:
            d["density"] = float("nan")
        d[KEEP].to_csv(fname(name), index=False)
        logs.append(log | {"source": name})
        print("  " + ", ".join(f"{k} {v}" for k, v in log.items() if k != "source"), flush=True)
    path = os.path.join(DATA_DIR, "cleaning_log.csv" if not only else "cleaning_log_partial.csv")
    pd.DataFrame(logs).to_csv(path, index=False)
    print("written", path)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="")
    a = p.parse_args()
    main([s.strip() for s in a.only.split(",") if s.strip()])
