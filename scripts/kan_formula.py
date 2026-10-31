"""
Let a Kolmogorov-Arnold Network learn the aggressiveness formula: one learned curve per input, then a readable equation.

    python scripts/kan_formula.py [--grid 5] [--degree 3] [--epochs 400]

Inputs: the unsquared window means of datasets/windows.py raw_inputs (speed, |accel|, gap closeness in metres and in
seconds, lane offset, TTC closeness, speed vs limit, speed vs traffic, IDM residual); unknown inputs are masked.
Model: model/kan.py AdditiveKAN, P(aggressive) = sigmoid(sum of curves), balanced cross entropy, small L2 on the
spline coefficients. Tests (aggressive vs normal AUC):
  planted SUMO drivers: train seeds 0 to 6 (sampled), test seeds 7 to 9
  UAH: leave one driver out
against the reference formula and the extended formula (data/extended_weights.json) on the same windows.
Per input: the learned curve (sampled), the best simple shape (x, x^2, x^3, sqrt, exp, |x|) and its R^2.
Outputs data/kan_results.csv, data/kan_curves.csv, data/kan_formula.json, results/figures/kan_curves.png.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, FIG_DIR  # noqa: E402
from datasets.signals import EXT_COLS  # noqa: E402
from datasets.windows import RAW_COLS, TERM_COLS, windows  # noqa: E402
from fit_reference_weights import auc_cols, scores_all  # noqa: E402
from model.aggressiveness_model import REF_WEIGHTS  # noqa: E402
from model.kan import AdditiveKAN, symbolic_fit  # noqa: E402

CLEAN = os.path.join(DATA_DIR, "clean")
NAMES = ["speed", "accel", "gap (m)", "headway", "lane", "TTC", "speed vs limit", "speed vs traffic", "IDM residual"]
TEST_SEEDS = (7, 8, 9)


def cached(name):
    path = os.path.join(CLEAN, f"windows_raw_{name}.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path)
    w = windows(pd.read_csv(os.path.join(CLEAN, f"{name}.csv.gz")), extended=True, raw=True)
    w.to_csv(path, index=False)
    return w


def train_kan(X, y, grid, degree, epochs, seed=0, l2=1e-3):
    torch.manual_seed(seed)
    m = AdditiveKAN(X.shape[1], grid_size=grid, degree=degree)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    pos = yt.mean().clamp(0.05, 0.95)
    wt = torch.where(yt > 0.5, 0.5 / pos, 0.5 / (1 - pos))
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(m(Xt), yt, reduction="none") * wt).mean()
        loss = loss + l2 * (m.coef ** 2).sum()
        loss.backward()
        opt.step()
    return m


def score(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).numpy()


def auc(s, y):
    return float(auc_cols(np.asarray(s, float)[:, None], np.asarray(y))[0])


def formulas(w):
    ref = scores_all(w[TERM_COLS].to_numpy(), np.asarray(REF_WEIGHTS)[None])[:, 0]
    path = os.path.join(DATA_DIR, "extended_weights.json")
    ext = None
    if os.path.exists(path):
        we = np.asarray(json.load(open(path))["weights"])
        ext = scores_all(w[TERM_COLS + EXT_COLS].to_numpy(), we[None])[:, 0]
    return ref, ext


def main(grid, degree, epochs):
    pl = cached("planted")
    pl = pl[pl["label"].isin(["aggressive", "normal"])]
    uah = cached("uah")
    uah = uah[uah["label"].isin(["aggressive", "normal"])]
    rng = np.random.default_rng(0)
    tr = pl[~pl["group"].isin(TEST_SEEDS)]
    tr = pd.concat([g.iloc[rng.permutation(len(g))[:20000]] for _, g in tr.groupby("label")])
    te = pl[pl["group"].isin(TEST_SEEDS)]
    rows, curves, eqs = [], [], {}

    m = train_kan(tr[RAW_COLS].to_numpy(), (tr["label"] == "aggressive").to_numpy(float), grid, degree, epochs)
    y = (te["label"] == "aggressive").to_numpy(int)
    ref, ext = formulas(te)
    rows.append({"data": "planted test seeds 7-9", "held_out": "", "kan": auc(score(m, te[RAW_COLS].to_numpy()), y),
                 "reference": auc(ref, y), "extended": auc(ext, y) if ext is not None else np.nan})
    print(rows[-1], flush=True)
    for i, n in enumerate(NAMES):
        x, c = m.curve_points(i)
        lo, hi = np.nanpercentile(tr[RAW_COLS[i]], [1, 99]) if tr[RAW_COLS[i]].notna().any() else (0, 1)
        keep = (x >= lo) & (x <= hi)
        curves += [{"data": "planted", "input": n, "x": float(a), "phi": float(b), "in_data_range": bool(k)} for a, b, k in zip(x, c, keep)]
        if keep.sum() > 5:
            eqs.setdefault("planted", {})[n] = symbolic_fit(x[keep], c[keep])

    for drv in sorted(uah["group"].unique()):
        a, b = uah[uah["group"] != drv], uah[uah["group"] == drv]
        mu = train_kan(a[RAW_COLS].to_numpy(), (a["label"] == "aggressive").to_numpy(float), grid, degree, epochs)
        yb = (b["label"] == "aggressive").to_numpy(int)
        ref, ext = formulas(b)
        rows.append({"data": "uah lodo", "held_out": drv, "kan": auc(score(mu, b[RAW_COLS].to_numpy()), yb),
                     "reference": auc(ref, yb), "extended": auc(ext, yb) if ext is not None else np.nan})
        print(rows[-1], flush=True)
    mu = train_kan(uah[RAW_COLS].to_numpy(), (uah["label"] == "aggressive").to_numpy(float), grid, degree, epochs)
    for i, n in enumerate(NAMES):
        x, c = mu.curve_points(i)
        col = uah[RAW_COLS[i]]
        lo, hi = np.nanpercentile(col, [1, 99]) if col.notna().any() else (np.nan, np.nan)
        keep = (x >= lo) & (x <= hi) if np.isfinite(lo) else np.zeros_like(x, bool)
        curves += [{"data": "uah", "input": n, "x": float(a), "phi": float(b), "in_data_range": bool(k)} for a, b, k in zip(x, c, keep)]
        if keep.sum() > 5:
            eqs.setdefault("uah", {})[n] = symbolic_fit(x[keep], c[keep])

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DATA_DIR, "kan_results.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(DATA_DIR, "kan_curves.csv"), index=False)
    out = {"grid_size": grid, "degree": degree, "epochs": epochs,
           "equations": {d: {n: {"shape": e[0], "a": round(e[1], 3), "c": round(e[2], 3), "r2": round(e[3], 3)}
                             for n, e in v.items()} for d, v in eqs.items()},
           "auc": res.groupby("data")[["kan", "reference", "extended"]].mean().round(4).to_dict()}
    json.dump(out, open(os.path.join(DATA_DIR, "kan_formula.json"), "w"), indent=2)
    print(res.groupby("data")[["kan", "reference", "extended"]].mean().round(3).to_string())
    for d, v in eqs.items():
        print(d, "score =", " + ".join(f"{e[1]:+.2f}·{e[0].replace('x', n)}" for n, e in v.items() if abs(e[1]) > 0.05))
    plot(pd.DataFrame(curves))


def plot(c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), dpi=110)
    for ax, n in zip(axes.flat, NAMES):
        for d, col in (("planted", "#7A1F2B"), ("uah", "#4A6FA5")):
            g = c[(c["input"] == n) & (c["data"] == d) & c["in_data_range"]]
            if len(g):
                ax.plot(g["x"], g["phi"] - g["phi"].iloc[0], color=col, lw=2.2, label="SUMO" if d == "planted" else "UAH")
        ax.set_title(n, fontsize=11)
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("Learned curve of each input (KAN): how much each value pushes toward 'aggressive'", fontsize=12)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "kan_curves.png"))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--epochs", type=int, default=400)
    a = p.parse_args()
    main(a.grid, a.degree, a.epochs)
