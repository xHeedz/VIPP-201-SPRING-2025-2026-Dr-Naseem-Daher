"""
UAH labels per window instead of per trip (bug 6): every window of an aggressive trip is labelled
aggressive, even calm cruising. SEMANTIC_ONLINE.txt holds, once per second, DriveSafe's own
"ratio aggressive WINDOW" (column 14, the last 60 s). Two label schemes, leave one driver out:

    trip    the trip label (as before)
    window  aggressive = window of an aggressive trip whose mean ratio aggressive >= 0.5,
            normal     = window of a normal trip whose mean ratio aggressive < 0.5, others dropped

Caveat: the DriveSafe ratio is an algorithm's estimate from the phone's own signals (accelerations,
brakings, car following, weaving), not a human label, so it partly measures what the index measures.

    python scripts/uah_relabel.py
Outputs data/uah_relabel.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.uah import find_trips, load_windows  # noqa: E402
from model.aggressiveness_model import headway_features, index_features  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_uah import evaluate, train  # noqa: E402

ROOT = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "UAH-DRIVESET-v1"))


def ratio_aggressive(win, root=ROOT):
    paths = {t["trip"]: t["path"] for t in find_trips(root)}
    out = np.full(len(win), np.nan)
    for trip, ix in win.groupby("trip").indices.items():
        s = pd.read_csv(os.path.join(paths[trip], "SEMANTIC_ONLINE.txt"), sep=r"\s+", header=None)
        t, r = s[0].to_numpy(), s[13].to_numpy()
        for i in ix:
            t0 = win["t0"].iloc[i]
            m = (t >= t0) & (t < t0 + 10)
            out[i] = r[m].mean() if m.any() else np.nan
    return out


def scheme(win, name):
    w = win[win["behavior"].isin(["normal", "aggressive"])]
    if name == "trip":
        return w
    keep = ((w["behavior"] == "aggressive") & (w["ratio_aggr"] >= 0.5)) | ((w["behavior"] == "normal") & (w["ratio_aggr"] < 0.5))
    return w[keep]


def main(epochs=600):
    rows = []
    for prox, feats in [("metres", index_features), ("headway", headway_features)]:
        win, _ = load_windows(ROOT, features=feats)
        win["ratio_aggr"] = ratio_aggressive(win)
        for sch in ["trip", "window"]:
            w = scheme(win, sch)
            for d in sorted(w["driver"].unique()):
                test = w[w["driver"] == d]
                agent = train(w[w["driver"] != d], epochs)
                rows.append({"proximity": prox, "labels": sch, "held_out_driver": d,
                             "n_aggressive": int((test["behavior"] == "aggressive").sum()),
                             "n_normal": int((test["behavior"] == "normal").sum()),
                             "original_auc": evaluate(test, "original")["auc"], "agent_auc": evaluate(test, agent)["auc"]})
            print(prox, sch, "kept", len(w), "of", int(win["behavior"].isin(["normal", "aggressive"]).sum()), flush=True)
        if prox == "metres":
            b = win[win["behavior"].isin(["normal", "aggressive"])].groupby("behavior")["ratio_aggr"]
            print("mean DriveSafe ratio aggressive per window:", b.mean().round(3).to_dict(),
                  " share >= 0.5:", b.apply(lambda s: (s >= 0.5).mean()).round(3).to_dict())
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(DATA_DIR, "uah_relabel.csv"), index=False)
    print(r.groupby(["proximity", "labels"])[["original_auc", "agent_auc", "n_aggressive", "n_normal"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
