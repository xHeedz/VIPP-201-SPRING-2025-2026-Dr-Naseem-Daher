"""
Does inverse RL recover the planted driver types? model/irl.py on the planted SUMO highway settings
(highway low and medium, jam, merge; 10 seeds).

    python scripts/irl_planted.py [--lam 0.01]

1. population theta0 from seeds 0 to 6 (all decisions pooled)
2. theta per vehicle (at least MIN_DEC decisions), pulled toward theta0 with lam; desired speed from theta
3. test seeds 7 to 9: AUC aggressive vs normal and conservative vs normal of desired speed, risk and discomfort
   weights, and of a logistic combination of the three fitted on the per vehicle weights of seeds 0 to 6
4. same test vehicles: AUC of the vehicle's mean index score (metres, headway, highway mix)
Outputs data/irl_planted_vehicles.csv and data/irl_planted_summary.csv.
"""
import argparse
import glob
import os
import re
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from model.aggressiveness_model import PROX_MIX, headway_features, index_features, mixed_features, original_score  # noqa: E402
from model.irl import NAMES, decisions, desired_speed_kmh, fit  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_uah import auc  # noqa: E402

SCENARIOS = ("highway_low", "highway_medium", "jam_jam", "merge_medium")
TEST_SEEDS = (7, 8, 9)
MIN_DEC = 20


def load():
    out = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", "*_s*.per_second.csv.gz"))):
        m = re.match(r"([a-z0-9]+_[a-z]+)_s(\d+)\.per_second", os.path.basename(p))
        if m.group(1) not in SCENARIOS:
            continue
        ps = pd.read_csv(p)
        for vid, g in ps.groupby("vehicle_id", sort=False):
            g = g[g["speed_kmh"] >= 5.0]
            f, y = decisions(g)
            if len(y) < MIN_DEC:
                continue
            s = {}
            for k, fn in [("metres", index_features), ("headway", headway_features),
                          ("mix", lambda a, b, c, d: mixed_features(a, b, c, d, PROX_MIX["highway"]))]:
                s[k] = float(original_score(fn(g["speed_kmh"], g["accel"], g["gap_m"], g["wave_m"]).mean(axis=0)))
            out.append({"scenario": m.group(1), "seed": int(m.group(2)), "vehicle_id": vid, "label": g["label"].iloc[0],
                        "f": f, "y": y, **{f"score_{k}": v for k, v in s.items()}})
    return out


def _fit_one(args):
    f, y, theta0, lam = args
    torch.set_num_threads(1)
    return fit(f, y, theta0, lam, iters=100)


def logistic(X, y, Xt):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    A = torch.tensor((X - mu) / sd)
    t = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(A.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w[:-1] + w[-1], t) + 1e-3 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return (torch.tensor((Xt - mu) / sd) @ w[:-1] + w[-1]).detach().numpy()


def main(lam):
    veh = load()
    train = [v for v in veh if v["seed"] not in TEST_SEEDS]
    F = np.concatenate([v["f"] for v in train])
    Y = np.concatenate([v["y"] for v in train])
    theta0 = fit(F, Y)
    print(f"{len(veh)} vehicles ({len(train)} train), {len(Y)} train decisions; theta0 {dict(zip(NAMES, theta0.round(3)))}")
    with Pool(8) as p:
        th = p.map(_fit_one, [(v["f"], v["y"], theta0, lam) for v in veh], chunksize=50)
    rows = [{k: v[k] for k in ("scenario", "seed", "vehicle_id", "label", "score_metres", "score_headway", "score_mix")}
            | {"decisions": len(v["y"]), **{f"theta_{n}": t[i] for i, n in enumerate(NAMES)}} for v, t in zip(veh, th)]
    d = pd.DataFrame(rows)
    d["desired_speed_kmh"] = desired_speed_kmh(d[[f"theta_{n}" for n in NAMES]].to_numpy())
    d["desired_speed_kmh"] = d["desired_speed_kmh"].fillna(d["desired_speed_kmh"].median()).clip(0, 300)
    d.to_csv(os.path.join(DATA_DIR, "irl_planted_vehicles.csv"), index=False)
    tr, te = d[~d["seed"].isin(TEST_SEEDS)], d[d["seed"].isin(TEST_SEEDS)]
    cols = ["desired_speed_kmh", "theta_risk", "theta_discomfort"]
    out = []
    for pos, neg in [("aggressive", "normal"), ("conservative", "normal")]:
        a, b = tr[tr["label"].isin([pos, neg])], te[te["label"].isin([pos, neg])]
        yb = (b["label"] == pos).astype(int).to_numpy()
        comb = logistic(a[cols].to_numpy(), (a["label"] == pos).astype(int).to_numpy(), b[cols].to_numpy())
        r = {"comparison": f"{pos} vs {neg}", "test_vehicles": len(b), "auc_irl_combined": auc(comb, yb)}
        for c in cols:
            r[f"auc_{c}"] = auc(b[c], yb)
        for k in ("metres", "headway", "mix"):
            r[f"auc_index_{k}"] = auc(b[f"score_{k}"], yb)
        for sc, g in b.assign(comb=comb, y=yb).groupby("scenario"):
            r[f"irl_{sc}"] = auc(g["comb"], g["y"])
            r[f"index_mix_{sc}"] = auc(g["score_mix"], g["y"])
        out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(DATA_DIR, "irl_planted_summary.csv"), index=False)
    print(te.groupby("label")[cols].median().round(3).to_string())
    with pd.option_context("display.width", 250):
        print(s.round(3).T.to_string())


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=0.01)
    main(p.parse_args().lam)
