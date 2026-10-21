"""
The reference formula on every cleaned source: label shares, AUC where labels exist, and where every point comes
from (mean points of each term, how often a term is unknown), overall and per density quartile.

    python scripts/rerun_formula_all.py

Reads data/clean/<source>.csv.gz (scripts/build_clean_tables.py), weights and cut offs from
data/reference_weights.json. Points of a term = 100 x weight x term / (sum of the weights of the known terms):
the points add up to the score of the window.
Outputs data/rerun_formula_summary.csv, data/rerun_formula_density.csv.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.windows import TERM_COLS, windows  # noqa: E402
from model.aggressiveness_model import REF_CUTOFFS, REF_WEIGHTS  # noqa: E402

CLEAN = os.path.join(DATA_DIR, "clean")
NAMES = ["speed", "accel2", "prox2", "wave"]


def points(w):
    T = w[TERM_COLS].to_numpy()
    k = ~np.isnan(T)
    wt = np.asarray(REF_WEIGHTS)
    den = (k * wt).sum(axis=1, keepdims=True)
    P = 100 * np.where(k, T, 0) * wt / np.where(den > 0, den, np.nan)
    return pd.DataFrame(P, columns=[f"pts_{n}" for n in NAMES], index=w.index)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    rows, dens = [], []
    lo, hi = REF_CUTOFFS
    for f in sorted(glob.glob(os.path.join(CLEAN, "*.csv.gz"))):
        if os.path.basename(f).startswith("windows_"):
            continue
        name = os.path.basename(f)[:-7]
        wpath = os.path.join(CLEAN, f"windows_{name}.csv.gz")
        w = pd.read_csv(wpath) if os.path.exists(wpath) else windows(pd.read_csv(f))
        if not os.path.exists(wpath):
            w.to_csv(wpath, index=False)
        P = points(w)
        w = pd.concat([w, P], axis=1)
        w["score"] = P.sum(axis=1)
        for lab, g in [("all", w)] + [(l, g) for l, g in w.groupby("label") if l != "unlabelled"]:
            r = {"source": name, "label": lab, "windows": len(g), "vehicles": g["vid"].nunique(),
                 "median_speed_kmh": g["speed_kmh"].median(), "median_score": g["score"].median(),
                 "share_conservative": (g["score"] < lo).mean(), "share_normal": ((g["score"] >= lo) & (g["score"] < hi)).mean(),
                 "share_aggressive": (g["score"] >= hi).mean(), "leader_share": g["leader_share"].mean()}
            for n in NAMES:
                r[f"pts_{n}"] = g[f"pts_{n}"].mean()
                r[f"unknown_{n}"] = g[f"t_{n}"].isna().mean()
            rows.append(r)
        labs = set(w["label"])
        if {"aggressive", "normal"} <= labs:
            rows.append({"source": name, "label": "AUC aggressive vs normal",
                         "median_score": auc(w.loc[w.label == "aggressive", "score"], w.loc[w.label == "normal", "score"])})
        if {"normal", "conservative"} <= labs:
            rows.append({"source": name, "label": "AUC normal vs conservative",
                         "median_score": auc(w.loc[w.label == "normal", "score"], w.loc[w.label == "conservative", "score"])})
        if w["leader_share"].notna().any():
            q = pd.qcut(w["leader_share"].rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
            for b, g in w.groupby(q):
                dens.append({"source": name, "car_ahead_quartile": b, "leader_share": g["leader_share"].mean(),
                             "median_score": g["score"].median(), "share_aggressive": (g["score"] >= hi).mean(),
                             **{f"pts_{n}": g[f"pts_{n}"].mean() for n in NAMES}})
        print(f"{name}: {len(w)} windows, median score {w['score'].median():.1f}, aggressive {(w['score'] >= hi).mean():.1%}",
              flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "rerun_formula_summary.csv"), index=False)
    pd.DataFrame(dens).to_csv(os.path.join(DATA_DIR, "rerun_formula_density.csv"), index=False)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
