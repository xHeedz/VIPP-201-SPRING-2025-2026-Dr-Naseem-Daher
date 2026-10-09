"""
Proximity as a mix of gap in metres and time headway, one mix per environment (Dr. Daher, Oct 2026).

    python scripts/fit_proximity_mix.py

model/aggressiveness_model.py mixed_features: n_p^2 = alpha * metres term + (1 - alpha) * headway term.
The window mean is linear in the per second terms, so every alpha is computed from two window tables
(metres and headway) of the same windows.

For alpha = 0, 0.1, ..., 1 and each environment (highway, urban, weather):
  - AUC aggressive vs normal windows, reference weights: planted SUMO drivers seeds 0 to 6 (fit) and 7 to 9
    (check), UAH-DriveSet (motorway = highway, secondary = urban; no weather trips)
  - density effect: median score of planted normal drivers in the jam minus on the low density highway
alpha per environment = the one with the highest mean fit AUC (planted seeds 0 to 6 and UAH where available).
Outputs data/proximity_mix_grid.csv and data/proximity_mix.json.
"""
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.sumo_log import windows_from_per_second  # noqa: E402
from datasets.uah import load_windows  # noqa: E402
from model.aggressiveness_model import headway_features, index_features, original_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from planted_drivers import ENVIRONMENT  # noqa: E402
from train_uah import auc  # noqa: E402

UAH_ROOT = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "UAH-DRIVESET-v1"))
FEATS = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 1)
TEST_SEEDS = (7, 8, 9)


def paired_windows(make):
    """window table with phi_prox_m (metres) and phi_prox_h (headway) of the same windows"""
    m, h = make(index_features), make(headway_features)
    assert len(m) == len(h) and (m["t0"].to_numpy() == h["t0"].to_numpy()).all()
    return m.rename(columns={"phi_prox": "phi_prox_m"}).assign(phi_prox_h=h["phi_prox"].to_numpy())


def scores(w, alpha):
    f = w[["phi_speed", "phi_accel", "phi_prox_m", "phi_wave"]].to_numpy().copy()
    f[:, 2] = alpha * w["phi_prox_m"].to_numpy() + (1 - alpha) * w["phi_prox_h"].to_numpy()
    return original_score(f)


def planted():
    frames = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", "*_s*.per_second.csv.gz"))):
        m = re.match(r"([a-z0-9]+)_([a-z]+)_s(\d+)\.per_second", os.path.basename(p))
        if m.group(1) == "us101":
            continue
        ps = pd.read_csv(p)
        scen, env = f"{m.group(1)}_{m.group(2)}", ENVIRONMENT[m.group(1)]
        w = paired_windows(lambda f: windows_from_per_second(ps, scen, env, features=f))
        frames.append(w.assign(scenario=scen, seed=int(m.group(3)), source="planted"))
    return pd.concat(frames, ignore_index=True)


def main():
    pl = planted()
    uah = paired_windows(lambda f: load_windows(UAH_ROOT, features=f)[0]).assign(source="uah", seed=-1, scenario="uah")
    print(f"planted windows {len(pl)}, UAH windows {len(uah)}")
    rows = []
    for env in ("highway", "urban", "weather"):
        sets = {"planted_fit": pl[(pl["environment"] == env) & ~pl["seed"].isin(TEST_SEEDS)],
                "planted_check": pl[(pl["environment"] == env) & pl["seed"].isin(TEST_SEEDS)],
                "uah": uah[uah["environment"] == env]}
        for a in ALPHAS:
            r = {"environment": env, "alpha": a}
            for name, w in sets.items():
                w = w[w["behavior"].isin(["normal", "aggressive"])]
                r[f"auc_{name}"] = auc(scores(w, a), (w["behavior"] == "aggressive").astype(int)) if len(w) else np.nan
            if env == "highway":
                nm = pl[(pl["behavior"] == "normal") & ~pl["seed"].isin(TEST_SEEDS)]
                lo, jam = nm[nm["scenario"] == "highway_low"], nm[nm["scenario"] == "jam_jam"]
                r["density_rise_normal"] = float(np.median(scores(jam, a)) - np.median(scores(lo, a)))
            rows.append(r)
    g = pd.DataFrame(rows)
    g["auc_fit_mean"] = g[["auc_planted_fit", "auc_uah"]].mean(axis=1, skipna=True)
    g.to_csv(os.path.join(DATA_DIR, "proximity_mix_grid.csv"), index=False)
    best = {e: float(d.loc[d["auc_fit_mean"].idxmax(), "alpha"]) for e, d in g.groupby("environment")}
    with open(os.path.join(DATA_DIR, "proximity_mix.json"), "w") as f:
        json.dump({"alpha_metres": best, "criterion": "max mean AUC aggressive vs normal, planted seeds 0-6 and UAH"},
                  f, indent=2)
    with pd.option_context("display.width", 200):
        print(g.round(3).to_string(index=False))
    print("alpha (share of metres) per environment:", best)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
