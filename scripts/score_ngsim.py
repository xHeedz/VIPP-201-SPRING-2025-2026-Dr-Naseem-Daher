"""
Score real NGSIM traffic: Aggressiveness Index and shockwave factor for every
vehicle, using the dynamic weight agent trained on UAH-DriveSet when available.

NGSIM has no behavior labels, so nothing is trained here. The point is to see
how the dynamic weight agent trained on labeled data behaves on thousands of real drivers,
and which drivers disturb the traffic behind them.

Traffic context: dense stop-and-go traffic makes every vehicle brake and follow
closely, so the index is also given relative to the traffic within 100 m
(model/context.py). Both are reported; the vehicle categories use the context score.
Acceleration for the index is measured the way UAH-DriveSet measures it
(accel_1hz in datasets/ngsim.py).

Sensor noise: every vehicle is then scored a second time through imperfect
sensors (model/noise.py, each of the 9 kinds at 1x, 2x and 4x strength), and the
share of vehicles whose category (conservative, normal, aggressive) stays the
same is reported. The clean recording plays the ground truth and the noisy copy
plays what an onboard sensor suite would see.

    python scripts/score_ngsim.py --file /path/to/ngsim.csv --location us-101 --minutes 5
    python scripts/score_ngsim.py --file trajectories-0750am-0805am.txt --minutes 5

Outputs data/ngsim_vehicles.csv, data/ngsim_noise_agreement.csv and
results/figures/ngsim_*.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR  # noqa: E402
from datasets.ngsim import environment_for, read_raw, trajectories  # noqa: E402
from model.aggressiveness_model import index_features, original_score  # noqa: E402
from model.context import context_adjusted, scorer_for  # noqa: E402
from model.shockwave import shockwave_factor  # noqa: E402
from model.ellipses import category  # noqa: E402
from model.noise import KINDS, noise_suite, sensor_view  # noqa: E402

NOISE_LEVELS = (1.0, 2.0, 4.0)


def row_scores(df, score, normal_mean=None):
    """Score of every row (vehicle and time); with normal_mean, the context score."""
    f = index_features(df["speed"] * 3.6, df["accel"], df["gap"], df["wave"])
    own = score(f)
    return own if normal_mean is None else context_adjusted(df, own, normal_mean)


def noise_agreement(one_hz, score, threshold, normal_mean, kinds=KINDS, levels=NOISE_LEVELS):
    """Share of vehicles whose (context) category is unchanged when the signals pass through
    noisy sensors. Acceleration is re-derived from the noisy 1 Hz speed, as in UAH."""
    def through_sensors(suite):
        parts = []
        for vid, g in one_hz.groupby("vehicle_id"):
            v, a, gap, lat = sensor_view(g["t"], g["speed"], g["accel"], g["gap"], g["wave"], suite, key=vid)
            parts.append(pd.DataFrame({"t": g["t"].to_numpy(), "pos": g["pos"].to_numpy(), "vehicle_id": vid,
                                       "speed": v, "accel": a, "gap": gap, "wave": lat}))
        rows = pd.concat(parts, ignore_index=True)
        return pd.Series(row_scores(rows, score, normal_mean), index=rows.index).groupby(rows["vehicle_id"]).mean()

    # the clean reference goes through the same sensor processing with zero noise, so
    # the only difference between the two is the noise itself
    clean = through_sensors(noise_suite("gaussian", 0.0))
    clean_cat = clean.apply(lambda v: category(v, threshold))
    rows = []
    for seed, (kind, level) in enumerate([(k, lv) for k in kinds for lv in levels]):
        noisy = through_sensors(noise_suite(kind, level, seed=seed))
        noisy_cat = noisy.apply(lambda v: category(v, threshold))
        rows.append({"kind": kind, "level": level, "agreement": float((noisy_cat == clean_cat).mean()),
                     "mean_abs_change": float((noisy - clean).abs().mean())})
    return pd.DataFrame(rows)


def main(path, location, minutes, environment=None):
    traj = trajectories(read_raw(path, location, minutes))
    env = environment or environment_for(location)
    score, thr, normal_mean, name = scorer_for(env, DATA_DIR)
    idx = traj[["t", "pos", "vehicle_id", "speed", "gap", "wave"]].assign(accel=traj["accel_1hz"])
    traj["ai_original"] = original_score(index_features(idx["speed"] * 3.6, idx["accel"], idx["gap"], idx["wave"]))
    traj["ai"] = row_scores(idx, score)
    traj["ai_context"] = row_scores(idx, score, normal_mean)
    one_hz = traj[np.isclose(traj["t"] % 1.0, 0.0, atol=0.05)]      # SF at 1 Hz keeps it fast
    sf = shockwave_factor(one_hz[["t", "vehicle_id", "lane", "pos", "speed", "accel"]])
    cols = ["ai_original", "ai", "ai_context"]
    per = traj.groupby("vehicle_id").agg(mean_speed=("speed", "mean"), **{c: (c, "mean") for c in cols}).join(sf)
    per["category_absolute"] = per["ai"].apply(lambda v: category(v, thr))
    per["category"] = per["ai_context"].apply(lambda v: category(v, thr))
    per.to_csv(os.path.join(DATA_DIR, "ngsim_vehicles.csv"))
    print(f"{len(per)} vehicles, {traj['t'].max() / 60:.1f} min, environment {env}, {name}, "
          f"aggressive from {thr:.1f}, normal UAH driver scores {normal_mean:.1f} on average")
    shares = pd.DataFrame({"absolute": per["category_absolute"].value_counts(normalize=True),
                           "with traffic context": per["category"].value_counts(normalize=True)}).fillna(0.0)
    print("share of vehicles per category:")
    print(shares.round(3).to_string())
    print(per[["mean_speed"] + cols + ["shockwave_factor"]].describe().round(3).to_string())
    plots(traj, per, "ai_context")

    na = noise_agreement(one_hz.drop(columns="accel").rename(columns={"accel_1hz": "accel"}), score, thr, normal_mean)
    na.to_csv(os.path.join(DATA_DIR, "ngsim_noise_agreement.csv"), index=False)
    print("share of vehicles whose category is unchanged under sensor noise (columns = noise strength):")
    print(na.pivot(index="kind", columns="level", values="agreement").round(3).to_string())
    plot_noise(na)


def plots(traj, per, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].hist(per[key], bins=40, range=(0, 100), color="#840032")
    axs[0].set_xlabel("mean Aggressiveness Index per vehicle, relative to nearby traffic")
    axs[0].set_title("real drivers, NGSIM", fontsize=10)
    axs[1].scatter(per[key], per["shockwave_factor"], s=6, alpha=0.5, color="#2a9d8f")
    axs[1].set_xlabel("mean Aggressiveness Index, relative to nearby traffic")
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


def plot_noise(na):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, g in na.groupby("kind"):
        g = g.sort_values("level")
        ax.plot([0.0] + list(g["level"]), [1.0] + list(g["agreement"]), marker="o", ms=3, label=k)
    ax.set_xlabel("noise strength (x base sensor error)")
    ax.set_ylabel("vehicles keeping their category")
    ax.set_title("real US traffic (NGSIM): clean recording against noisy sensors", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ngsim_noise_agreement.png"), dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--location", default=None, help="us-101, i-80, lankershim or peachtree (combined CSV only)")
    ap.add_argument("--minutes", type=float, default=5.0, help="first N minutes only; NGSIM files are large")
    ap.add_argument("--environment", choices=["highway", "urban"], default=None)
    a = ap.parse_args()
    main(a.file, a.location, a.minutes, a.environment)
