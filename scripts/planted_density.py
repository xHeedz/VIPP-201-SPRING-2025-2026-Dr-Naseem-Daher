"""
Density sweep on the planted SUMO highway (low / medium / jam): median window score per true driver
type and mean proximity points, gap in metres and time headway.

    python scripts/planted_density.py      # writes data/planted_density_headway.csv
"""
import glob
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.sumo_log import windows_from_per_second  # noqa: E402
from model.aggressiveness_model import THRESHOLDS, headway_features, original_score  # noqa: E402

FEATS = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    rows = []
    for sc, d in [("highway", "low"), ("highway", "medium"), ("jam", "jam")]:
        for f in sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", f"{sc}_{d}_s*.per_second.csv.gz"))):
            ps = pd.read_csv(f)
            for name, feat in [("metres", None), ("headway", headway_features)]:
                w = windows_from_per_second(ps, sc, "highway", features=feat)
                w["s"] = original_score(w[FEATS].to_numpy())
                for c in ["conservative", "normal", "aggressive"]:
                    g = w[w["behavior"] == c]
                    rows.append({"density": d, "prox": name, "type": c, "median": g["s"].median(),
                                 "prox_points": 100 * 0.8 * g["phi_prox"].mean(), "share_aggressive": (g["s"] >= THRESHOLDS[1]).mean()})
    t = pd.DataFrame(rows).groupby(["prox", "type", "density"], sort=False)[["median", "prox_points", "share_aggressive"]].mean().unstack("density")
    t.to_csv(os.path.join(DATA_DIR, "planted_density_headway.csv"))
    print(t.round(3).to_string())
