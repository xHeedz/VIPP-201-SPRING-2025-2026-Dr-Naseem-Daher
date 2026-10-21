"""
Inverse RL (model/irl.py) on the cleaned planted drivers, compared with the reference formula, and a variant fitted
only on the seconds where the car follows another car (time headway below 4 s), where a driver's preferred gap can
show. The selection depends on the situation only, not on the driver's choice, so the fit stays unbiased.

    python scripts/rerun_irl.py [--lam 0.01]

Same as scripts/irl_planted.py otherwise (population theta from seeds 0 to 6, per vehicle fits pulled toward it,
logistic combination of desired speed, desired headway and discomfort, tested on seeds 7 to 9).
Outputs data/rerun_irl_summary.csv, data/rerun_irl_vehicles.csv.
"""
import argparse
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402
from irl_planted import MIN_DEC, SCENARIOS, TEST_SEEDS, _fit_one, auc, logistic  # noqa: E402
from model.aggressiveness_model import reference_score, reference_terms  # noqa: E402
from model.irl import A_GRID, NAMES, decisions, desired_headway_s, desired_speed_kmh, fit  # noqa: E402


def load(following_only):
    d = pd.read_csv(os.path.join(DATA_DIR, "clean", "planted.csv.gz"))
    d = d[d["scenario"].isin(SCENARIOS) & (d["speed_kmh"] >= 5.0)]
    d["gap_m"] = d["gap_m"].fillna(0.0)
    out = []
    for vid, g in d.groupby("vid", sort=False):
        g = g.sort_values("t")
        f, y = decisions(g)
        if following_only and len(y):
            # headway feature (index 2, headway / 4 s, 1 = no car within 4 s) of the "keep speed" choice
            keep = f[:, int(np.flatnonzero(A_GRID == 0.0)[0]), 2] < 1.0
            f, y = f[keep], y[keep]
        if len(y) < MIN_DEC:
            continue
        T = reference_terms(g["speed_kmh"], g["accel"], g["gap_m"].where(g["gap_m"] > 0, 0.0), g["wave_m"], g["environment"].to_numpy())
        out.append({"scenario": g["scenario"].iloc[0], "seed": int(g["group"].iloc[0]), "vid": vid, "label": g["label"].iloc[0],
                    "f": f, "y": y, "score": float(reference_score(np.nanmean(T, axis=0)))})
    return out


def run(veh, lam, tag):
    train = [v for v in veh if v["seed"] not in TEST_SEEDS]
    theta0 = fit(np.concatenate([v["f"] for v in train]), np.concatenate([v["y"] for v in train]))
    with Pool(8) as p:
        th = p.map(_fit_one, [(v["f"], v["y"], theta0, lam) for v in veh], chunksize=50)
    d = pd.DataFrame([{k: v[k] for k in ("scenario", "seed", "vid", "label", "score")} | {"decisions": len(v["y"])}
                      | {f"theta_{n}": t[i] for i, n in enumerate(NAMES)} for v, t in zip(veh, th)])
    TH = d[[f"theta_{n}" for n in NAMES]].to_numpy()
    d["desired_speed_kmh"] = pd.Series(desired_speed_kmh(TH)).fillna(np.nan).clip(0, 300).to_numpy()
    d["desired_headway_s"] = pd.Series(desired_headway_s(TH)).clip(0, 10).to_numpy()
    for c in ("desired_speed_kmh", "desired_headway_s"):
        d[c] = d[c].fillna(d[c].median())
    tr, te = d[~d["seed"].isin(TEST_SEEDS)], d[d["seed"].isin(TEST_SEEDS)]
    cols = ["desired_speed_kmh", "desired_headway_s", "theta_discomfort"]
    rows = []
    for pos, neg in [("aggressive", "normal"), ("conservative", "normal")]:
        a, b = tr[tr["label"].isin([pos, neg])], te[te["label"].isin([pos, neg])]
        yb = (b["label"] == pos).astype(int).to_numpy()
        comb = logistic(a[cols].to_numpy(), (a["label"] == pos).astype(int).to_numpy(), b[cols].to_numpy())
        sgn = 1 if pos == "aggressive" else -1
        rows.append({"variant": tag, "comparison": f"{pos} vs {neg}", "test_vehicles": len(b),
                     "median_decisions": float(b["decisions"].median()), "auc_irl": auc(comb, yb),
                     "auc_formula": auc(sgn * b["score"].to_numpy(), yb)})
    pop = {lab: float(desired_headway_s(fit(np.concatenate([v["f"] for v in train if v["label"] == lab]),
                                            np.concatenate([v["y"] for v in train if v["label"] == lab])))) for lab in
           ("conservative", "normal", "aggressive")}
    for r in rows:
        r.update({f"pooled_headway_{k}": v for k, v in pop.items()})
    med = te.groupby("label")["desired_headway_s"].median()
    for r in rows:
        r.update({f"per_vehicle_headway_{k}": float(v) for k, v in med.items()})
    return d.assign(variant=tag), rows


def main(lam, tag=""):
    allrows, vehs = [], []
    for variant, fol in (("all seconds", False), ("following seconds only", True)):
        veh = load(fol)
        print(f"{variant}: {len(veh)} vehicles with at least {MIN_DEC} decisions", flush=True)
        d, rows = run(veh, lam, variant)
        vehs.append(d)
        allrows += rows
        print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    pd.DataFrame(allrows).to_csv(os.path.join(DATA_DIR, f"rerun_irl_summary{tag}.csv"), index=False)
    pd.concat(vehs).to_csv(os.path.join(DATA_DIR, f"rerun_irl_vehicles{tag}.csv"), index=False)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--tag", default="", help="suffix of the output files")
    a = p.parse_args()
    main(a.lam, a.tag)
