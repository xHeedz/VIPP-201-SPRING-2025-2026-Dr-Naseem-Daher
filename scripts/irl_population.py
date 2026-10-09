"""
Population level inverse RL (model/irl.py): one reward per group, fitted on all its decisions pooled.

    python scripts/irl_population.py

Groups:
  planted SUMO highway settings, seeds 0 to 6, one fit per true type (conservative, normal, aggressive): check
      whether the model separates the types when data is plentiful (per vehicle fits are weak: AUC 0.73)
  highD, exiD (German motorways), NGSIM US-101: one fit each, and highD / exiD per density quartile
Per fit: desired speed v*, desired headway h*, discomfort weight, decisions used, mean log likelihood per decision.
Outputs data/irl_population.csv.
"""
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from model.irl import NAMES, decisions, desired_headway_s, desired_speed_kmh, fit, log_likelihood  # noqa: E402

BASE = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "12-10-2026"))
NGSIM = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "ngsim_us-101_5min.csv"))
SCENARIOS = ("highway_low", "highway_medium", "jam_jam", "merge_medium")
MAX_DEC = 400_000      # random subsample per fit (LBFGS on more adds nothing)


def collect(ps_iter):
    F, Y = [], []
    for g in ps_iter:
        f, y = decisions(g[g["speed_kmh"] >= 5.0])
        if len(y):
            F.append(f)
            Y.append(y)
    return (np.concatenate(F), np.concatenate(Y)) if F else (np.zeros((0, 12, 5)), np.zeros(0, int))


def fit_row(name, F, Y, rng):
    if len(Y) > MAX_DEC:
        k = rng.choice(len(Y), MAX_DEC, replace=False)
        F, Y = F[k], Y[k]
    th = fit(F, Y, iters=300)
    return {"group": name, "decisions": len(Y), "desired_speed_kmh": float(desired_speed_kmh(th)),
            "desired_headway_s": float(desired_headway_s(th)), "discomfort": float(th[4]),
            "loglik_per_decision": log_likelihood(F, Y, th) / len(Y), **{f"theta_{n}": float(v) for n, v in zip(NAMES, th)}}


def by_vehicle(ps):
    return (g for _, g in ps.groupby("vehicle_id", sort=False))


def main():
    rng = np.random.default_rng(0)
    rows = []
    groups = {"conservative": [], "normal": [], "aggressive": []}
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", "*_s*.per_second.csv.gz"))):
        m = re.match(r"([a-z0-9]+_[a-z]+)_s(\d+)\.per_second", os.path.basename(p))
        if m.group(1) in SCENARIOS and int(m.group(2)) < 7:
            ps = pd.read_csv(p)
            for lab, g in ps.groupby("label"):
                groups[lab].append(collect(by_vehicle(g)))
    for lab, parts in groups.items():
        rows.append(fit_row(f"planted {lab}", np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]), rng))
        print(rows[-1], flush=True)
    german = pd.read_csv(os.path.join(DATA_DIR, "german_windows.csv.gz"), usecols=["dataset", "vehicle_id", "density_bin"])
    bin_of = german.groupby("vehicle_id")["density_bin"].agg(lambda s: s.mode().iloc[0])
    for name in ("highD", "exiD"):
        F, Y, B = [], [], []
        for p in sorted(glob.glob(os.path.join(BASE, name, "per_second", "*.csv.gz"))):
            ps = pd.read_csv(p)
            for vid, g in ps.groupby("vehicle_id", sort=False):
                f, y = decisions(g[g["speed_kmh"] >= 5.0])
                if len(y):
                    F.append(f)
                    Y.append(y)
                    B.append(np.full(len(y), bin_of.get(vid, "none"), dtype=object))
        F, Y, B = np.concatenate(F), np.concatenate(Y), np.concatenate(B)
        rows.append(fit_row(name, F, Y, rng))
        print(rows[-1], flush=True)
        for b in ("q1 light", "q2", "q3", "q4 dense"):
            rows.append(fit_row(f"{name} {b}", F[B == b], Y[B == b], rng))
    from datasets.ngsim import per_second, read_raw, trajectories
    us = per_second(trajectories(read_raw(NGSIM, "us-101")))
    rows.append(fit_row("NGSIM US-101", *collect(by_vehicle(us)), rng))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA_DIR, "irl_population.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(out[["group", "decisions", "desired_speed_kmh", "desired_headway_s", "discomfort", "loglik_per_decision"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
