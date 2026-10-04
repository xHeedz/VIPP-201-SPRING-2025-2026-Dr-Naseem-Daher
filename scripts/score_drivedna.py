"""
DriveDNA-Sample through the same per second -> 10 s window -> index path as UAH.

    python scripts/score_drivedna.py

No aggressiveness labels, so three driver level questions. Several car models log no lane positions, so the
comparisons use the three term score (speed, acceleration, proximity; the lane term left out for every car):
  1. one driver, 14 cars (driver_078): is the median score per car stable (spread across cars vs between drivers)?
  2. one car, several drivers (shared HONDA_CIVIC): how far apart are the drivers?
  3. real radar car following at low speed (< 30 km/h, leader present) against free driving: does the index climb
     with metres and not with headway (the density effect seen in SUMO and NGSIM)?
Outputs data/drivedna_windows.csv.gz and data/drivedna_summary.csv.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.drivedna import find_drives, per_second  # noqa: E402
from datasets.uah import windows  # noqa: E402
from model.aggressiveness_model import THRESHOLDS, headway_features, original_score  # noqa: E402

ROOT = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "drivedna_sample"))
FEATS = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]


def all_windows():
    frames = []
    for r in find_drives(ROOT).itertuples():
        ps = per_second(r.path)
        trip = {"trip": f"{r.driver}/{r.drive}", "driver": r.driver, "road": r.car_model, "environment": "highway",
                "behavior": "unlabelled"}
        w = windows(ps, trip)
        wh = windows(ps, trip, features=headway_features)
        if not len(w):
            continue
        keep = w["speed_kmh"] >= 5.0
        # per window: share of seconds with a leader
        t = ps["t"].to_numpy()
        lead = np.array([(ps["gap_m"].to_numpy()[(t >= a) & (t < a + 10)] > 0).mean() for a in w["t0"]])
        f3, fh3 = w[FEATS].to_numpy().copy(), wh[FEATS].to_numpy().copy()
        f3[:, 3] = fh3[:, 3] = 0.0
        frames.append(w.assign(car=r.car_model, score=original_score(f3), score_headway=original_score(fh3),
                               score_4term=original_score(w[FEATS].to_numpy()), lane_valid=ps["lane_valid"].mean(),
                               lead_share=lead)[keep])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    w = all_windows()
    w.to_csv(os.path.join(DATA_DIR, "drivedna_windows.csv.gz"), index=False)
    print(f"{len(w)} windows, {w['trip'].nunique()} drives, {w['driver'].nunique()} drivers, {w['car'].nunique()} cars; "
          f"leader present in {100 * (w['lead_share'] > 0.5).mean():.0f}% of windows; lane positions logged for "
          f"{w.loc[w['lane_valid'] > 0.5, 'car'].nunique()} cars")
    rows = []
    # 1. driver_078 across cars
    d78 = w[w["driver"] == "driver_078"]
    per_car = d78.groupby("car")[["score", "score_headway"]].median()
    rows.append({"question": "driver_078, spread of the per-car median across 14 cars (sd)",
                 "metres": per_car["score"].std(), "headway": per_car["score_headway"].std()})
    rows.append({"question": "driver_078, range of the per-car median (max - min)",
                 "metres": per_car["score"].max() - per_car["score"].min(),
                 "headway": per_car["score_headway"].max() - per_car["score_headway"].min()})
    # 2. shared civic
    civ = w[w["car"] == "HONDA_CIVIC"].groupby("driver")[["score", "score_headway"]].median()
    for drv, r in civ.iterrows():
        rows.append({"question": f"HONDA_CIVIC, median score of {drv}", "metres": r["score"], "headway": r["score_headway"]})
    rows.append({"question": "HONDA_CIVIC, range across drivers", "metres": civ["score"].max() - civ["score"].min(),
                 "headway": civ["score_headway"].max() - civ["score_headway"].min()})
    # 3. low speed car following vs free driving
    follow = w[(w["speed_kmh"] < 30) & (w["lead_share"] > 0.8)]
    free = w[(w["speed_kmh"] >= 60) & (w["lead_share"] < 0.2)]
    for name, g in [("slow car following (< 30 km/h, leader)", follow), ("free driving (>= 60 km/h, no leader)", free)]:
        rows.append({"question": f"median score, {name}, {len(g)} windows", "metres": g["score"].median(),
                     "headway": g["score_headway"].median()})
        rows.append({"question": f"share labelled aggressive (>= {THRESHOLDS[1]:g}), {name}",
                     "metres": (g["score"] >= THRESHOLDS[1]).mean(), "headway": (g["score_headway"] >= THRESHOLDS[1]).mean()})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(DATA_DIR, "drivedna_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(s.round(3).to_string(index=False))
        print(per_car.round(1).to_string())
