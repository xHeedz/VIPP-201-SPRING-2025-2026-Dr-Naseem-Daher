"""
Fit the weights and cut offs of the reference formula (speed linear, acceleration squared, proximity mix, unknown
terms left out) on labelled windows, and test them on data never used for fitting.

    python scripts/fit_reference_weights.py

Candidates: every weight vector on a 0.05 grid with four non negative weights summing to 1 (1,771).
Criterion: mean of three AUCs on the fit data: UAH aggressive vs normal, planted aggressive vs normal, planted
normal vs conservative (planted fit = seeds 0 to 6, at most 40,000 windows per class, fixed random sample).
Cut offs: aggressive = Youden J of aggressive vs normal (mean of the UAH and planted cut), conservative =
Youden J of normal vs conservative (planted only; UAH has no conservative class).
Tests: UAH leave one driver out (weights refitted without the held out driver), planted test seeds 7 to 9.
Outputs data/reference_weights.json, data/reference_fit.csv; windows cached in data/clean/windows_*.csv.gz.
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.windows import TERM_COLS, windows  # noqa: E402

CLEAN = os.path.join(DATA_DIR, "clean")
TEST_SEEDS = (7, 8, 9)
GRID = np.array([w for w in itertools.product(range(21), repeat=4) if sum(w) == 20], dtype=float) / 20.0


def cached_windows(name):
    path = os.path.join(CLEAN, f"windows_{name}.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path)
    w = windows(pd.read_csv(os.path.join(CLEAN, f"{name}.csv.gz")))
    w.to_csv(path, index=False)
    return w


def scores_all(T, weights):
    """(n, 4) terms with NaN, (c, 4) weights -> (n, c) scores, unknown terms left out."""
    known = ~np.isnan(T)
    num = np.where(known, T, 0.0) @ weights.T
    den = known.astype(float) @ weights.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100.0 * num / np.where(den > 0, den, np.nan)


def auc_cols(S, y):
    """AUC of every column of S for binary y (1 = positive); rows with NaN score dropped per column."""
    out = np.empty(S.shape[1])
    for j in range(S.shape[1]):
        s = S[:, j]
        ok = ~np.isnan(s)
        s, yy = s[ok], y[ok]
        r = pd.Series(s).rank().to_numpy()
        npos = yy.sum()
        nneg = len(yy) - npos
        out[j] = (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else np.nan
    return out


def youden(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    best = (-1, None)
    for c in np.arange(5, 95, 0.5):
        j = (s[y == 1] >= c).mean() - (s[y == 0] >= c).mean()
        if j > best[0]:
            best = (j, c)
    return best[1]


def pair(w, pos, neg):
    m = w["label"].isin([pos, neg])
    return w.loc[m, TERM_COLS].to_numpy(), (w.loc[m, "label"] == pos).astype(int).to_numpy()


def criterion(sets, weights):
    return np.nanmean([auc_cols(scores_all(T, weights), y) for T, y in sets], axis=0)


def main():
    uah = cached_windows("uah")
    uah = uah[uah["label"].isin(["normal", "aggressive"])]
    pl = cached_windows("planted")
    fit = pl[~pl["group"].isin(TEST_SEEDS)]
    test = pl[pl["group"].isin(TEST_SEEDS)]
    rng = np.random.default_rng(0)
    fit = pd.concat([g.iloc[rng.permutation(len(g))[:40000]] for _, g in fit.groupby("label")])
    print(f"UAH {len(uah)} windows, planted fit {len(fit)} (sampled), planted test {len(test)}", flush=True)

    sets = [pair(uah, "aggressive", "normal"), pair(fit, "aggressive", "normal"), pair(fit, "normal", "conservative")]
    crit = criterion(sets, GRID)
    best = GRID[int(np.nanargmax(crit))]
    print("best weights (speed, accel^2, prox^2 mix, wave):", best, f"criterion {np.nanmax(crit):.3f}", flush=True)

    # cut offs on the fit data
    s_uah = scores_all(uah[TERM_COLS].to_numpy(), best[None])[:, 0]
    s_fit = scores_all(fit[TERM_COLS].to_numpy(), best[None])[:, 0]
    c_agg = float(np.mean([youden(s_uah, uah["label"] == "aggressive"),
                           youden(s_fit[fit["label"].isin(["aggressive", "normal"]).to_numpy()],
                                  fit.loc[fit["label"].isin(["aggressive", "normal"]), "label"] == "aggressive")]))
    m = fit["label"].isin(["normal", "conservative"]).to_numpy()
    c_con = float(youden(s_fit[m], fit.loc[m, "label"] == "normal"))
    print(f"cut offs: conservative below {c_con:.1f}, aggressive from {c_agg:.1f}", flush=True)

    # honest tests
    rows = []
    for drv in sorted(uah["group"].unique()):
        tr, te = uah[uah["group"] != drv], uah[uah["group"] == drv]
        w = GRID[int(np.nanargmax(criterion([pair(tr, "aggressive", "normal")] + sets[1:], GRID)))]
        T, y = pair(te, "aggressive", "normal")
        rows.append({"test": "uah lodo", "held_out": drv, "auc": float(auc_cols(scores_all(T, w[None]), y)[0]),
                     "weights": " / ".join(f"{x:.2f}" for x in w)})
    for scen, g in test.groupby("scenario"):
        for pos, neg in [("aggressive", "normal"), ("normal", "conservative")]:
            T, y = pair(g, pos, neg)
            rows.append({"test": f"planted test {pos} vs {neg}", "held_out": scen,
                         "auc": float(auc_cols(scores_all(T, best[None]), y)[0]), "weights": ""})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DATA_DIR, "reference_fit.csv"), index=False)
    summary = res.groupby("test")["auc"].mean()
    print(summary.round(3).to_string())
    with open(os.path.join(DATA_DIR, "reference_weights.json"), "w") as f:
        json.dump({"terms": ["speed", "accel^2", "prox^2 (mix per environment)", "wave"],
                   "weights": [float(x) for x in best], "cutoffs": [round(c_con, 1), round(c_agg, 1)],
                   "criterion": "max mean AUC: UAH A vs N, planted A vs N, planted N vs C (seeds 0-6)",
                   "fit_criterion": float(np.nanmax(crit)),
                   "tests": {k: round(float(v), 4) for k, v in summary.items()}}, f, indent=2)
    print("written data/reference_weights.json")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
