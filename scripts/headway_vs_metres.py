"""
Proximity as gap in metres (reference) against time headway (gap / speed), on UAH-DriveSet
with leave one driver out: AUC of normal vs aggressive windows of the held out driver, for the
hand-set index and for a DynamicWeightAgent trained on the other five drivers.

    python scripts/headway_vs_metres.py [--root ../28:9:2026/UAH-DRIVESET-v1] [--epochs 600]

Outputs data/headway_vs_metres.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.uah import load_windows  # noqa: E402
from model.aggressiveness_model import headway_features, index_features  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_uah import evaluate, train  # noqa: E402


def main(root, epochs):
    rows = []
    for name, feats in [("metres", index_features), ("headway", headway_features)]:
        win, _ = load_windows(root, features=feats)
        binary = win[win["behavior"].isin(["normal", "aggressive"])]
        for d in sorted(binary["driver"].unique()):
            test = binary[binary["driver"] == d]
            agent = train(binary[binary["driver"] != d], epochs)
            rows.append({"proximity": name, "held_out_driver": d,
                         "original_auc": evaluate(test, "original")["auc"],
                         "agent_auc": evaluate(test, agent)["auc"],
                         "mean_phi_prox_normal": float(test.loc[test["behavior"] == "normal", "phi_prox"].mean()),
                         "mean_phi_prox_aggressive": float(test.loc[test["behavior"] == "aggressive", "phi_prox"].mean())})
            print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(DATA_DIR, "headway_vs_metres.csv"), index=False)
    s = r.groupby("proximity")[["original_auc", "agent_auc"]].agg(["mean", "std"])
    print(s.round(3).to_string())
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "UAH-DRIVESET-v1")))
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()
    main(a.root, a.epochs)
