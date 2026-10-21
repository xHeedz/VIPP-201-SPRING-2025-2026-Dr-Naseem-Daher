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

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402
from datasets.cleaning import clean  # noqa: E402
import audit_datasets as A  # noqa: E402

OUT = os.path.join(DATA_DIR, "clean")
KEEP = ["source", "vid", "group", "label", "environment", "scenario", "t", "speed_kmh", "accel", "gap_m", "wave_m",
        "leader", "density"]
PLANTED_ENV = {"highway": "highway", "jam": "highway", "merge": "highway", "weather": "weather",
               "urban": "urban", "roundabout": "urban"}


def uah():
    from datasets.uah import find_trips, load_trip
    parts = []
    for trip in find_trips(os.path.join(A.EXT, "UAH-DRIVESET-v1")):
        parts.append(load_trip(trip).assign(vid=trip["trip"], group=trip["driver"], label=trip["behavior"],
                                            environment=trip["environment"], scenario=trip["road"]))
    return pd.concat(parts, ignore_index=True)


def planted():
    d = A.planted()
    run = d["vid"].str.split("/").str[0]
    m = run.str.extract(r"(?P<scen>[a-z0-9]+)_(?P<dens>[a-z]+)_s(?P<seed>\d+)")
    return d.assign(group=m["seed"].astype(int), scenario=m["scen"] + "_" + m["dens"],
                    environment=m["scen"].map(PLANTED_ENV))


def ngsim(site, env):
    def f():
        d = A.ngsim(site)
        return d.assign(group=site, label="unlabelled", environment=env, scenario=site)
    return f


def unlabelled(fn, name, env):
    def f():
        d = fn()
        rec = d["recording"].astype(str) if "recording" in d else name
        dens = d["density_50m"] if "density_50m" in d else d.get("density")
        return d.assign(group=rec, label="unlabelled", environment=env, scenario=name, density=dens)
    return f


def drivedna():
    d = A.drivedna()
    return d.assign(group=d["vid"].str.split("/").str[0], label="unlabelled", environment="highway", scenario="drivedna")


BUILD = {"uah": uah, "planted": planted,
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
        d, log = clean(d, name.split(" ")[0] if name.startswith(("ngsim", "drivedna")) else name)
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
