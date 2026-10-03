"""
Sensor noise on real drivers: how much does imperfect sensing hurt the trained
agent's ability to tell aggressive from normal driving?

    python scripts/noise_uah.py --root /path/to/UAH-DRIVESET-v1

Run scripts/train_uah.py first. It saves one agent per held-out driver in
data/uah_lodo_agents/, each trained without that driver. Here every held-out
driver's trips are scored again by that same agent, but the signals first pass
through each of the 9 noise kinds in model/noise.py at 1x, 2x and 4x strength
(datasets/uah.py apply_noise). The original hand-set index is scored on the same
noisy signals for comparison. Level 0 ("clean") uses the same sensor processing
with no noise added, so it can differ slightly from the train_uah.py numbers.

Outputs data/uah_noise_robustness.csv (one row per kind, level and driver) and
results/figures/uah_noise_robustness.png.
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, FIG_DIR  # noqa: E402
from datasets.uah import apply_noise, find_trips, load_trip, windows  # noqa: E402
from model.dynamic_weight_agent import DynamicWeightAgent  # noqa: E402
from model.noise import KINDS, noise_suite  # noqa: E402
import train_uah as tu  # noqa: E402   evaluate() and LODO_DIR are reused from Step 1

LEVELS = (1.0, 2.0, 4.0)
MIN_SPEED_KMH = 5.0          # same filter as datasets/uah.py load_windows


def driver_windows(per_second, trips, suite=None):
    """Window table for one driver's trips, as recorded (suite=None) or through sensors."""
    frames = []
    for trip in trips:
        ps = per_second[trip["trip"]]
        if suite is not None:
            ps = apply_noise(ps, suite, key=trip["trip"])
        w = windows(ps, trip)
        if len(w):
            frames.append(w[w["speed_kmh"] >= MIN_SPEED_KMH])
    return pd.concat(frames, ignore_index=True)


def main(root, kinds, levels):
    trips = [t for t in find_trips(root) if t["behavior"] in ("normal", "aggressive")]
    per_second = {t["trip"]: load_trip(t) for t in trips}
    rows = []
    for d in sorted({t["driver"] for t in trips}):
        path = os.path.join(tu.LODO_DIR, f"agent_without_{d}.json")
        if not os.path.exists(path):
            sys.exit(f"{path} is missing: run scripts/train_uah.py first")
        agent = DynamicWeightAgent.load(path)
        mine = [t for t in trips if t["driver"] == d]
        # "clean" goes through the same sensor processing with zero noise, so the only
        # difference between it and the other rows is the noise itself
        cases = [("clean", 0.0)] + [(k, lv) for k in kinds for lv in levels]
        for seed, (kind, level) in enumerate(cases):
            suite = noise_suite("gaussian", 0.0) if kind == "clean" else noise_suite(kind, level, seed=seed)
            w = driver_windows(per_second, mine, suite)
            new, base = tu.evaluate(w, agent), tu.evaluate(w, "original")
            rows.append({"kind": kind, "level": level, "driver": d,
                         "agent_auc": new["auc"], "agent_window_acc": new["window_acc"],
                         "original_auc": base["auc"], "original_window_acc": base["window_acc"]})
        print(f"  {d} done")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DATA_DIR, "uah_noise_robustness.csv"), index=False)

    mean = res.groupby(["kind", "level"])[["agent_auc", "original_auc"]].mean().reset_index()
    clean = mean[mean["kind"] == "clean"].iloc[0]
    print(f"clean: agent AUC {clean['agent_auc']:.3f}, original AUC {clean['original_auc']:.3f}")
    table = mean[mean["kind"] != "clean"].pivot(index="kind", columns="level", values="agent_auc").round(3)
    print("agent AUC under noise (mean over the 6 held-out drivers), columns = noise strength:")
    print(table.to_string())
    plot(mean, clean)


def plot(mean, clean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = [k for k in mean["kind"].unique() if k != "clean"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, col, title in zip(axs, ["agent_auc", "original_auc"],
                              ["dynamic weight agent trained on UAH", "original hand-set index"]):
        for k in kinds:
            m = mean[mean["kind"] == k].sort_values("level")
            ax.plot([0.0] + list(m["level"]), [clean[col]] + list(m[col]), marker="o", ms=3, label=k)
        ax.axhline(0.5, color="k", lw=0.6, ls="--")
        ax.set_xlabel("noise strength (x base sensor error)")
        ax.set_title(title, fontsize=10)
    axs[0].set_ylabel("AUC on held-out drivers")
    axs[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "uah_noise_robustness.png"), dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing D1 ... D6")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument("--levels", nargs="+", type=float, default=list(LEVELS))
    a = ap.parse_args()
    main(a.root, a.kinds, a.levels)
