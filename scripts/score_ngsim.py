"""
Score real NGSIM traffic: Aggressiveness Index and shockwave factor for every
vehicle, using the weights learned on UAH-DriveSet when available.

NGSIM has no behavior labels, so nothing is trained here. The point is to see
how the index trained on labeled data behaves on thousands of real drivers,
and which drivers disturb the traffic behind them.

    python scripts/score_ngsim.py --file /path/to/ngsim.csv --location us-101 --minutes 5
    python scripts/score_ngsim.py --file trajectories-0750am-0805am.txt --minutes 5

Outputs data/ngsim_vehicles.csv and results/figures/ngsim_*.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR  # noqa: E402
from datasets.ngsim import context_for, read_raw, trajectories  # noqa: E402
from model.calibrated_index import CalibratedIndex, original_ai, phi  # noqa: E402
from model.shockwave import shockwave_factor  # noqa: E402


def main(path, location, minutes, context=None):
    traj = trajectories(read_raw(path, location, minutes))
    ctx = context or context_for(location)
    f = phi(traj["speed"] * 3.6, traj["accel"], traj["gap"], traj["wave"])
    traj["ai_original"] = original_ai(f)
    cal_path = os.path.join(DATA_DIR, "calibrated_index.json")
    calibrated = os.path.exists(cal_path)
    if calibrated:
        model = CalibratedIndex.load(cal_path)
        traj["ai_calibrated"] = model.ai(f, ctx)
    one_hz = traj[np.isclose(traj["t"] % 1.0, 0.0, atol=0.05)]      # SF at 1 Hz keeps it fast
    sf = shockwave_factor(one_hz[["t", "vehicle_id", "lane", "pos", "speed", "accel"]])
    cols = ["ai_original"] + (["ai_calibrated"] if calibrated else [])
    per = traj.groupby("vehicle_id").agg(mean_speed=("speed", "mean"), **{c: (c, "mean") for c in cols}).join(sf)
    key = "ai_calibrated" if calibrated else "ai_original"
    thr = model.thresholds[ctx] if calibrated else 70.0
    per["category"] = np.where(per[key] >= thr, "aggressive", np.where(per[key] < 35, "conservative", "normal"))
    per.to_csv(os.path.join(DATA_DIR, "ngsim_vehicles.csv"))
    print(f"{len(per)} vehicles, {traj['t'].max() / 60:.1f} min, context {ctx}, "
          f"{'calibrated' if calibrated else 'original'} index")
    print(per["category"].value_counts(normalize=True).round(3).to_string())
    print(per[["mean_speed"] + cols + ["shockwave_factor"]].describe().round(3).to_string())
    plots(traj, per, key)


def plots(traj, per, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].hist(per[key], bins=40, range=(0, 100), color="#840032")
    axs[0].set_xlabel("mean Aggressiveness Index per vehicle")
    axs[0].set_title("real drivers, NGSIM", fontsize=10)
    axs[1].scatter(per[key], per["shockwave_factor"], s=6, alpha=0.5, color="#2a9d8f")
    axs[1].set_xlabel("mean Aggressiveness Index")
    axs[1].set_ylabel("shockwave factor")
    axs[1].set_title("own aggressiveness against disturbance caused upstream", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ngsim_ai_and_shockwave.png"), dpi=140)

    lane = traj["lane"].mode().iloc[0]
    d = traj[(traj["lane"] == lane) & np.isclose(traj["t"] % 0.5, 0.0, atol=0.05)]
    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(d["t"], d["pos"], c=d["speed"], s=0.4, cmap="RdYlGn", vmin=0, vmax=30)
    fig.colorbar(sc, label="speed (m/s)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position (m)")
    ax.set_title(f"time-space diagram, lane {lane}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ngsim_timespace.png"), dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--location", default=None, help="us-101, i-80, lankershim or peachtree (combined CSV only)")
    ap.add_argument("--minutes", type=float, default=5.0, help="first N minutes only; NGSIM files are large")
    ap.add_argument("--context", choices=["motorway", "secondary"], default=None)
    a = ap.parse_args()
    main(a.file, a.location, a.minutes, a.context)
