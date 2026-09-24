"""
Noise robustness sweep: how much each kind of sensor error hurts the
agent-side assessment, measured as category agreement with the exact
(ground-truth) assessment in the three SUMO validation scenarios.

    python scripts/noise_robustness.py                  # all kinds, levels 0.5 1 2 4
    python scripts/noise_robustness.py --kinds gaussian spikes --levels 1 2

Outputs data/noise_robustness.csv and results/figures/noise_robustness.png.
"""
import argparse
import contextlib
import csv
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import sumo_runner as sr  # noqa: E402

from model.noise import KINDS, noise_suite  # noqa: E402

SCENARIOS = [
    # label, cfg key, n_steps, kwargs (same settings as sumo_runner.main)
    ("highway", "hw", 200, dict(friction=1.0, speed_ref=38.0, gap_ref=25.0)),
    ("urban", "int", 250, dict(friction=1.0, speed_ref=22.0, gap_ref=17.0, thresh_aggr=50)),
    ("weather", "wthr", 200, dict(friction=0.3, speed_ref=25.0, gap_ref=35.0)),
]


def build_configs():
    hw_net, int_net = sr.generate_highway_network(), sr.generate_intersection_network()
    cfg = {}
    for key, net, writer in [("hw", hw_net, lambda p: sr.write_highway_routes(p, "highway")),
                             ("wthr", hw_net, lambda p: sr.write_highway_routes(p, "weather")),
                             ("int", int_net, sr.write_intersection_routes)]:
        rou = os.path.join(sr.NET_DIR, f"noise_{key}.rou.xml")
        writer(rou)
        cfg[key] = os.path.join(sr.NET_DIR, f"noise_{key}.sumocfg")
        sr.write_sumocfg(cfg[key], net, rou)
    return cfg


def run(kind, level, cfg, seed=0):
    out = {}
    for label, key, steps, kw in SCENARIOS:
        np.random.seed(seed)
        suite = None if kind == "original" else noise_suite(kind, level, seed=seed)
        with contextlib.redirect_stdout(io.StringIO()):
            r = sr.run_scenario(label, cfg[key], "ego", n_steps=steps, noise=suite,
                                noise_scale=level if kind == "original" else 1.0, **kw)
        out[label] = r["accuracy"]
    return out


def plot(rows, levels, kinds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.full((len(kinds), len(levels)), np.nan)
    for r in rows:
        if r["kind"] in kinds:
            grid[kinds.index(r["kind"]), levels.index(r["level"])] = r["mean"]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(kinds) + 1.6))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
    ax.set_xticks(range(len(levels)), [f"x{l:g}" for l in levels])
    ax.set_yticks(range(len(kinds)), kinds)
    for i in range(len(kinds)):
        for j in range(len(levels)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("noise severity")
    ax.set_title("category agreement with ground truth (%), mean of 3 SUMO scenarios", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--levels", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = build_configs()
    rows = []
    base = run("original", 1.0, cfg, a.seed)
    print(f"original gaussian (noise_scale 1): {base}")
    for kind in a.kinds:
        for level in a.levels:
            res = run(kind, level, cfg, a.seed)
            mean = sum(res.values()) / len(res)
            rows.append({"kind": kind, "level": level, "mean": mean, **res})
            print(f"{kind:15s} x{level:<4g} mean {mean:5.1f}%  " + "  ".join(f"{k} {v:5.1f}" for k, v in res.items()))
    out_csv = os.path.join(DATA_DIR, "noise_robustness.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "level", "mean", "highway", "urban", "weather"])
        w.writeheader()
        w.writerows(rows)
    plot(rows, a.levels, a.kinds, os.path.join(FIG_DIR, "noise_robustness.png"))
    print(f"wrote {out_csv}")
