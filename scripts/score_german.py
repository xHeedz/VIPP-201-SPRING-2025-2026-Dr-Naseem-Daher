"""
Score highD (German motorways) and exiD (German motorway entries and exits) with the index in metres, in time
headway and with the highway proximity mix (model/aggressiveness_model.py PROX_MIX), per density bin.

    python scripts/score_german.py [--datasets highD,exiD]

Data in ../12-10-2026/{highD,exiD}/data (levelXdata, research-only licence, not redistributed). Per second tables
are cached there as per_second/<rec>.csv.gz. Same path as every other source: per second table -> 10 s windows
(5 s step, windows below 5 km/h dropped) -> mean index features -> score.
Density: number of scored vehicles in the same recording at the same second (same driving direction for highD),
averaged over the window; bins by quartile within each dataset.
Outputs data/german_windows.csv.gz and data/german_summary.csv.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.uah import windows  # noqa: E402
from model.aggressiveness_model import (PROX_MIX, THRESHOLDS, headway_features, index_features,  # noqa: E402
                                        mixed_features, original_score)

BASE = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "12-10-2026"))
FEATS = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]


def loader(name):
    if name == "highD":
        from datasets import highd as m
    else:
        from datasets import exid as m
    return m


def cached_per_second(name, rec):
    root = os.path.join(BASE, name, "data")
    cache = os.path.join(BASE, name, "per_second", f"{rec}.csv.gz")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    ps = loader(name).per_second(root, rec)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    ps.to_csv(cache, index=False)
    return ps


def add_density(ps, name):
    key = ["t"] + (["direction"] if "direction" in ps.columns else [])
    ps["density"] = ps.groupby(key)["vehicle_id"].transform("size")
    return ps


def score_recording(name, rec):
    ps = cached_per_second(name, rec)
    if not len(ps):
        return pd.DataFrame()
    ps = add_density(ps, name)
    frames = []
    for vid, g in ps.groupby("vehicle_id", sort=False):
        g = g.reset_index(drop=True)
        trip = {"trip": vid, "driver": vid, "road": name, "environment": "highway", "behavior": "unlabelled"}
        w = {k: windows(g, trip, features=f) for k, f in
             [("m", index_features), ("h", headway_features),
              ("x", lambda s, a, p, wv: mixed_features(s, a, p, wv, PROX_MIX["highway"]))]}
        if not len(w["m"]):
            continue
        out = w["m"][["trip", "t0", "speed_kmh"] + FEATS].rename(columns={"trip": "vehicle_id"})
        out["score_metres"] = original_score(w["m"][FEATS].to_numpy())
        out["score_headway"] = original_score(w["h"][FEATS].to_numpy())
        out["score_mix"] = original_score(w["x"][FEATS].to_numpy())
        out["phi_prox_headway"] = w["h"]["phi_prox"].to_numpy()
        t = g["t"].to_numpy()
        out["density"] = [g["density"].to_numpy()[(t >= a) & (t < a + 10)].mean() for a in out["t0"]]
        out["leader_share"] = [(g["gap_m"].to_numpy()[(t >= a) & (t < a + 10)] > 0).mean() for a in out["t0"]]
        out["class"] = g["class"].iloc[0]
        frames.append(out[out["speed_kmh"] >= 5.0])
    w = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return w.assign(dataset=name, recording=rec)


def main(names):
    allw = []
    for name in names:
        t = time.time()
        recs = loader(name).recordings(os.path.join(BASE, name, "data"))
        for i, rec in enumerate(recs):
            allw.append(score_recording(name, rec))
            if i % 10 == 0:
                print(f"  {name} {rec}: {time.time() - t:.0f} s", flush=True)
        print(f"{name}: {len(recs)} recordings in {time.time() - t:.0f} s", flush=True)
    w = pd.concat(allw, ignore_index=True)
    w["density_bin"] = w.groupby("dataset")["density"].transform(
        lambda d: pd.qcut(d, 4, labels=["q1 light", "q2", "q3", "q4 dense"], duplicates="drop"))
    w.to_csv(os.path.join(DATA_DIR, "german_windows.csv.gz"), index=False)
    groups = [((ds, "all"), g) for ds, g in w.groupby("dataset")]
    groups += [((ds, b), g) for (ds, b), g in w.groupby(["dataset", "density_bin"], observed=True)]
    rows = []
    for (ds, b), g in groups:
        r = {"dataset": ds, "density_bin": b, "windows": len(g), "vehicles": g["vehicle_id"].nunique(),
             "median_speed_kmh": g["speed_kmh"].median(), "median_density": g["density"].median(),
             "leader_share": g["leader_share"].mean()}
        for k in ("metres", "headway", "mix"):
            r[f"median_{k}"] = g[f"score_{k}"].median()
            r[f"aggressive_{k}"] = (g[f"score_{k}"] >= THRESHOLDS[1]).mean()
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(DATA_DIR, "german_summary.csv"), index=False)
    with pd.option_context("display.width", 250):
        print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="highD,exiD")
    main(p.parse_args().datasets.split(","))
