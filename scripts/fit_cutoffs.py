"""
Fit the label cut offs of the index on labelled data instead of the hand-set 35 / 70.

    python scripts/fit_cutoffs.py

aggressive cut off: maximises Youden J (TPR - FPR) of aggressive vs normal
conservative cut off: maximises Youden J of conservative vs normal (score below the cut off); SUMO only,
UAH has no conservative class
Data and held out tests:
    UAH windows, leave one driver out (fit on five drivers, test on the sixth)
    SUMO planted drivers (calibrated tau), vehicle level mean window score, split by seed (fit 0 to 4, test 5 to 9,
    and the reverse)
Both proximity variants (gap in metres, time headway). Reported on the held out part: aggressive recall,
share of normal drivers labelled aggressive, balanced accuracy, against the same numbers for 35 / 70.

Outputs data/cutoffs.csv (held out results) and data/cutoffs_fitted.json (cut offs fitted on all data).
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.sumo_log import windows_from_per_second  # noqa: E402
from datasets.uah import load_windows  # noqa: E402
from model.aggressiveness_model import headway_features, index_features, original_score  # noqa: E402

UAH = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "UAH-DRIVESET-v1"))
FEATS = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]


def youden(pos, neg, higher=True):
    """cut off maximising TPR - FPR; positives are above it (higher=True) or below it"""
    cand = np.unique(np.concatenate([pos, neg]))
    if not higher:
        pos, neg, cand = -pos, -neg, -cand
    tpr = np.array([(pos >= c).mean() for c in cand])
    fpr = np.array([(neg >= c).mean() for c in cand])
    c = cand[np.argmax(tpr - fpr)]
    return float(c if higher else -c)


def metrics(score, truth, c_cons, c_aggr):
    pred = np.where(score >= c_aggr, "aggressive", np.where(score < c_cons, "conservative", "normal"))
    classes = [k for k in ("conservative", "normal", "aggressive") if (truth == k).any()]
    recall = {k: float((pred[truth == k] == k).mean()) for k in classes}
    return {"recall_aggressive": recall.get("aggressive", np.nan), "recall_normal": recall.get("normal", np.nan),
            "recall_conservative": recall.get("conservative", np.nan),
            "normal_labelled_aggressive": float((pred[truth == "normal"] == "aggressive").mean()),
            "balanced_accuracy": float(np.mean(list(recall.values())))}


def uah(prox, feats):
    w, _ = load_windows(UAH, features=feats)
    w = w[w["behavior"].isin(["normal", "aggressive"])].reset_index(drop=True)
    s = original_score(w[FEATS].to_numpy())
    rows = []
    for d in sorted(w["driver"].unique()):
        tr, te = (w["driver"] != d).to_numpy(), (w["driver"] == d).to_numpy()
        c = youden(s[tr & (w["behavior"] == "aggressive").to_numpy()], s[tr & (w["behavior"] == "normal").to_numpy()])
        for name, cc, ca in [("fitted", -np.inf, c), ("35/70", -np.inf, 70.0)]:
            rows.append({"data": "uah", "proximity": prox, "cutoffs": name, "fold": d, "c_aggressive": ca,
                         **metrics(s[te], w.loc[te, "behavior"].to_numpy(), cc, ca)})
    full = youden(s[(w["behavior"] == "aggressive").to_numpy()], s[(w["behavior"] == "normal").to_numpy()])
    return rows, {"aggressive": full}


def sumo_vehicles(feats):
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", "*.per_second.csv.gz"))):
        sc, dens, seed = os.path.basename(f).replace(".per_second.csv.gz", "").rsplit("_", 2)
        w = windows_from_per_second(pd.read_csv(f), sc, "x", features=feats)
        w = w.assign(score=original_score(w[FEATS].to_numpy()))
        v = w.groupby("trip").agg(score=("score", "mean"), truth=("behavior", "first")).reset_index()
        frames.append(v.assign(scenario=sc, density=dens, seed=int(seed[1:])))
    return pd.concat(frames, ignore_index=True)


def sumo(prox, feats):
    v = sumo_vehicles(feats)
    rows = []
    for half in (0, 1):
        tr = ((v["seed"] < 5) == (half == 0)).to_numpy()
        te = ~tr
        s, t = v["score"].to_numpy(), v["truth"].to_numpy()
        ca = youden(s[tr & (t == "aggressive")], s[tr & (t == "normal")])
        cc = youden(s[tr & (t == "conservative")], s[tr & (t == "normal")], higher=False)
        for name, c1, c2 in [("fitted", cc, ca), ("35/70", 35.0, 70.0)]:
            rows.append({"data": "sumo", "proximity": prox, "cutoffs": name, "fold": f"seeds {'5-9' if half == 0 else '0-4'}",
                         "c_conservative": c1, "c_aggressive": c2, **metrics(s[te], t[te], c1, c2)})
    s, t = v["score"].to_numpy(), v["truth"].to_numpy()
    full = {"conservative": youden(s[t == "conservative"], s[t == "normal"], higher=False),
            "aggressive": youden(s[t == "aggressive"], s[t == "normal"])}
    return rows, full


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    rows, fitted = [], {}
    for prox, feats in [("metres", index_features), ("headway", headway_features)]:
        r, f = uah(prox, feats)
        rows += r
        fitted[f"uah_{prox}"] = f
        r, f = sumo(prox, feats)
        rows += r
        fitted[f"sumo_{prox}"] = f
        print(prox, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DATA_DIR, "cutoffs.csv"), index=False)
    with open(os.path.join(DATA_DIR, "cutoffs_fitted.json"), "w") as fh:
        json.dump(fitted, fh, indent=2)
    cols = ["c_conservative", "c_aggressive", "recall_conservative", "recall_normal", "recall_aggressive",
            "normal_labelled_aggressive", "balanced_accuracy"]
    with pd.option_context("display.width", 200):
        print(res.groupby(["data", "proximity", "cutoffs"])[cols].mean().round(3).to_string())
    print(json.dumps(fitted, indent=1))
