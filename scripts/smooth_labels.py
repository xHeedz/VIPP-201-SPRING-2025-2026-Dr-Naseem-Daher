"""
Three ways to soften the hard cut-off (aggressive from 27.5) on the UAH 10 s windows, and what they do to the two audit
groups (label aggressive / formula calm, label normal / formula aggressive).

    python scripts/smooth_labels.py [--band 2.5] [--hold 2]

1. probability: logistic P(aggressive trip | score), fitted leave-one-driver-out (each driver scored by a fit on the others)
2. borderline band: score within +-band of the cut-off -> 'borderline' (the agent keeps watching instead of deciding)
3. hysteresis: per trip, in time order, switch to aggressive only after `hold` windows in a row at or above cut + band,
   back to normal only after `hold` windows in a row below cut - band (a two-state machine, like a thermostat)

writes data/smoothing_windows.csv (per window) and data/smoothing_summary.csv (per audit group and method)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.windows import TERM_COLS  # noqa: E402
from model.aggressiveness_model import REF_CUTOFFS, reference_score  # noqa: E402

CUT = REF_CUTOFFS[1]


def logistic(x, y, iters=50):
    """one input logistic regression by Newton's method, returns (b0, b1)"""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        beta += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
    return beta


def probability(w):
    p = np.full(len(w), np.nan)
    y = (w["label"] == "aggressive").to_numpy(float)
    x = w["score"].to_numpy()
    for drv in w["driver"].unique():
        test = (w["driver"] == drv).to_numpy()
        b0, b1 = logistic(x[~test], y[~test])
        p[test] = 1 / (1 + np.exp(-(b0 + b1 * x[test])))
    b0, b1 = logistic(x, y)
    return p, -b0 / b1


def hysteresis(scores, upper, lower, hold):
    state, run, out = "normal", 0, []
    for s in scores:
        if state == "normal":
            run = run + 1 if s >= upper else 0
            if run >= hold:
                state, run = "aggressive", 0
        else:
            run = run + 1 if s < lower else 0
            if run >= hold:
                state, run = "normal", 0
        out.append(state)
    return out


def main(band, hold):
    w = pd.read_csv(os.path.join(DATA_DIR, "clean", "windows_uah.csv.gz"))
    w = w[w["label"].isin(["normal", "aggressive"])].sort_values(["vid", "t0"]).reset_index(drop=True)
    w["score"] = reference_score(w[TERM_COLS].to_numpy())
    w["driver"] = w["vid"].str.extract(r"-(D\d)-")[0]
    w["hard"] = np.where(w["score"] >= CUT, "aggressive", "normal")
    w["prob"], p50 = probability(w)
    w["band"] = np.where(w["score"] >= CUT + band, "aggressive",
                         np.where(w["score"] < CUT - band, "normal", "borderline"))
    w["hyst"] = np.concatenate([hysteresis(d["score"].to_numpy(), CUT + band, CUT - band, hold)
                                for _, d in w.groupby("vid", sort=False)])
    w["audit_group"] = np.select([(w["label"] == "aggressive") & (w["hard"] == "normal"),
                                  (w["label"] == "normal") & (w["hard"] == "aggressive"),
                                  (w["label"] == "aggressive")], ["label aggressive, formula calm",
                                                                  "label normal, formula aggressive", "agree: aggressive"],
                                 "agree: normal")
    rows = []
    for g, d in w.groupby("audit_group"):
        rows.append({"group": g, "windows": len(d),
                     "hard_aggressive": (d["hard"] == "aggressive").mean(),
                     "prob_median": d["prob"].median(), "prob_over_0.5": (d["prob"] >= 0.5).mean(),
                     "band_aggressive": (d["band"] == "aggressive").mean(), "band_borderline": (d["band"] == "borderline").mean(),
                     "band_normal": (d["band"] == "normal").mean(),
                     "hyst_aggressive": (d["hyst"] == "aggressive").mean()})
    s = pd.DataFrame(rows)
    y = w["label"] == "aggressive"
    sure = w["band"] != "borderline"
    overall = {"hard": ((w["hard"] == "aggressive") == y).mean(), "prob>=0.5": ((w["prob"] >= 0.5) == y).mean(),
               "band (decided only)": ((w.loc[sure, "band"] == "aggressive") == y[sure]).mean(),
               "band share decided": sure.mean(), "hysteresis": ((w["hyst"] == "aggressive") == y).mean()}
    w.drop(columns=TERM_COLS).to_csv(os.path.join(DATA_DIR, "smoothing_windows.csv"), index=False)
    s.to_csv(os.path.join(DATA_DIR, "smoothing_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print(f"score where P(aggressive) = 0.5: {p50:.1f}")
    print({k: round(float(v), 3) for k, v in overall.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--band", type=float, default=2.5)
    p.add_argument("--hold", type=int, default=2)
    a = p.parse_args()
    main(a.band, a.hold)
