"""
Extended formula: the four reference terms plus TTC, speed relative to the limit, speed relative to the traffic and
the IDM residual (datasets/signals.py), weights refitted the same way as scripts/fit_reference_weights.py.

    python scripts/fit_extended_weights.py [--step 0.1]

Grid: every vector of 8 non negative weights on a --step grid summing to 1. Criterion: mean of UAH aggressive vs
normal, planted aggressive vs normal and planted normal vs conservative AUC (planted seeds 0 to 6, sampled).
Tests: UAH leave one driver out (weights refitted without the held out driver), planted seeds 7 to 9 per scenario.
Also: AUC of every single term on UAH and on planted test seeds (how much each new signal is worth on its own),
and the reference formula on the same windows.
Outputs data/extended_weights.json, data/extended_fit.csv, data/extended_single_terms.csv;
windows cached in data/clean/windows_ext_*.csv.gz.
"""
import argparse
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402
from datasets.signals import EXT_COLS  # noqa: E402
from datasets.windows import TERM_COLS, windows  # noqa: E402
from fit_reference_weights import auc_cols, scores_all, youden  # noqa: E402
from model.aggressiveness_model import REF_WEIGHTS  # noqa: E402

CLEAN = os.path.join(DATA_DIR, "clean")
COLS = TERM_COLS + EXT_COLS
NAMES = ["speed", "accel2", "gap2", "lane", "ttc", "rel_limit", "rel_traffic", "idm"]
TEST_SEEDS = (7, 8, 9)


def grid(step):
    n = int(round(1 / step))
    return np.array([w for w in itertools.product(range(n + 1), repeat=len(COLS)) if sum(w) == n], dtype=float) / n


def cached(name):
    path = os.path.join(CLEAN, f"windows_ext_{name}.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path)
    w = windows(pd.read_csv(os.path.join(CLEAN, f"{name}.csv.gz")), extended=True)
    w.to_csv(path, index=False)
    return w


def pair(w, pos, neg):
    m = w["label"].isin([pos, neg])
    return w.loc[m, COLS].to_numpy(), (w.loc[m, "label"] == pos).astype(int).to_numpy()


def crit(sets, G, chunk=500):
    out = []
    for i in range(0, len(G), chunk):
        g = G[i:i + chunk]
        out.append(np.nanmean([auc_cols(scores_all(T, g), y) for T, y in sets], axis=0))
    return np.concatenate(out)


def main(step):
    uah = cached("uah")
    uah = uah[uah["label"].isin(["normal", "aggressive"])]
    pl = cached("planted")
    fit = pl[~pl["group"].isin(TEST_SEEDS)]
    test = pl[pl["group"].isin(TEST_SEEDS)]
    rng = np.random.default_rng(0)
    fit = pd.concat([g.iloc[rng.permutation(len(g))[:15000]] for _, g in fit.groupby("label")])
    print("unknown share per term, UAH:", uah[COLS].isna().mean().round(3).to_dict(), flush=True)
    print("unknown share per term, planted:", pl[COLS].isna().mean().round(3).to_dict(), flush=True)

    single = []
    for name, (T, y) in [("UAH A vs N", pair(uah, "aggressive", "normal")),
                         ("planted test A vs N", pair(test, "aggressive", "normal")),
                         ("planted test N vs C", pair(test, "normal", "conservative"))]:
        single.append({"test": name, **{n: float(auc_cols(T[:, [i]], y)[0]) for i, n in enumerate(NAMES)}})
    single = pd.DataFrame(single)
    single.to_csv(os.path.join(DATA_DIR, "extended_single_terms.csv"), index=False)
    print(single.round(3).to_string(index=False), flush=True)

    G = grid(step)
    sets = [pair(uah, "aggressive", "normal"), pair(fit, "aggressive", "normal"), pair(fit, "normal", "conservative")]
    c = crit(sets, G)
    best = G[int(np.nanargmax(c))]
    print(f"{len(G)} weight vectors; best", dict(zip(NAMES, best)), f"criterion {np.nanmax(c):.3f}", flush=True)
    ref = np.r_[np.asarray(REF_WEIGHTS), np.zeros(len(EXT_COLS))]
    print(f"reference formula on the same criterion: {crit(sets, ref[None])[0]:.3f}", flush=True)

    s_uah = scores_all(uah[COLS].to_numpy(), best[None])[:, 0]
    m = fit["label"].isin(["aggressive", "normal"]).to_numpy()
    s_fit = scores_all(fit[COLS].to_numpy(), best[None])[:, 0]
    c_agg = float(np.mean([youden(s_uah, uah["label"] == "aggressive"), youden(s_fit[m], fit.loc[m, "label"] == "aggressive")]))
    m2 = fit["label"].isin(["normal", "conservative"]).to_numpy()
    c_con = float(youden(s_fit[m2], fit.loc[m2, "label"] == "normal"))

    rows = []
    for drv in sorted(uah["group"].unique()):
        tr, te = uah[uah["group"] != drv], uah[uah["group"] == drv]
        w = G[int(np.nanargmax(crit([pair(tr, "aggressive", "normal")] + sets[1:], G)))]
        T, y = pair(te, "aggressive", "normal")
        rows.append({"test": "uah lodo", "held_out": drv, "auc_extended": float(auc_cols(scores_all(T, w[None]), y)[0]),
                     "auc_reference": float(auc_cols(scores_all(T, ref[None]), y)[0]),
                     "weights": " / ".join(f"{x:.1f}" for x in w)})
        print(rows[-1], flush=True)
    for scen, g in test.groupby("scenario"):
        for pos, neg in [("aggressive", "normal"), ("normal", "conservative")]:
            T, y = pair(g, pos, neg)
            rows.append({"test": f"planted test {pos} vs {neg}", "held_out": scen,
                         "auc_extended": float(auc_cols(scores_all(T, best[None]), y)[0]),
                         "auc_reference": float(auc_cols(scores_all(T, ref[None]), y)[0]), "weights": ""})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DATA_DIR, "extended_fit.csv"), index=False)
    summary = res.groupby("test")[["auc_extended", "auc_reference"]].mean()
    print(summary.round(3).to_string(), flush=True)
    with open(os.path.join(DATA_DIR, "extended_weights.json"), "w") as f:
        json.dump({"terms": NAMES, "weights": [float(x) for x in best], "cutoffs": [round(c_con, 1), round(c_agg, 1)],
                   "grid_step": step, "fit_criterion": float(np.nanmax(c)),
                   "tests": {k: {"extended": round(float(v["auc_extended"]), 4), "reference": round(float(v["auc_reference"]), 4)}
                             for k, v in summary.iterrows()}}, f, indent=2)
    print("written data/extended_weights.json")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=float, default=0.1)
    main(p.parse_args().step)
