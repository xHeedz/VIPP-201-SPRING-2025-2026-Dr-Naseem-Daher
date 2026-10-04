"""
Unlabelled trajectory sets through the same per second -> 10 s window -> index path as UAH:
score histograms per road type and per traffic density bin, gap in metres and time headway.

    python scripts/score_unlabelled.py

Sources (next to the repo in ../28:9:2026/):
    NGSIM US-101 (freeway, 5 min extract), I-80 (freeway, first 15 min period),
    Lankershim and Peachtree (arterials, through traffic on road sections only:
    Int_ID 0, Movement 1, Direction north or south), pNEUMA (downtown Athens, one drone slice,
    cars, taxis, buses and trucks; three terms, no lane offset)
Density: NGSIM vehicles within +-100 m along the road in the same direction, per km per lane
(lanes per direction: US-101 5, I-80 6, Lankershim 3, Peachtree 2); pNEUMA vehicles within 50 m.
UAH windows (normal and aggressive) are the labelled reference.

Outputs data/unlabelled_windows.csv.gz, data/unlabelled_summary.csv,
results/figures/unlabelled_scores.png, results/figures/unlabelled_density.png
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR  # noqa: E402
from datasets.sumo_log import windows_from_per_second  # noqa: E402
from model.aggressiveness_model import THRESHOLDS, WEIGHTS, headway_features, original_score  # noqa: E402

EXT = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026"))
FEATS = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]
LANES = {"us-101": 5, "i-80": 6, "lankershim": 3, "peachtree": 2}
SITES = [  # name, file, location, minutes, road type, planar
    ("ngsim us-101", "ngsim_us-101_5min.csv", "us-101", None, "freeway", False),
    ("ngsim i-80", "ngsim_i-80.csv", "i-80", 15, "freeway", False),
    ("ngsim lankershim", "ngsim_lankershim.csv", "lankershim", None, "arterial", True),
    ("ngsim peachtree", "ngsim_peachtree.csv", "peachtree", None, "arterial", True),
]
DENSITY_BINS = [0, 10, 20, 40, 60, np.inf]
DENSITY_LABELS = ["<10", "10-20", "20-40", "40-60", ">60"]


def ngsim_per_second(file, location, minutes, planar):
    from datasets.ngsim import per_second, read_raw, trajectories
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = trajectories(read_raw(os.path.join(EXT, file), location, minutes), planar=planar)
    ps = per_second(tr)
    if planar:
        ps = ps[(ps["Int_ID"] == 0) & (ps["Movement"] == 1) & ps["Direction"].isin([2, 4])]
        group = ["t", "Direction"]
    else:
        group = ["t"]
    dens = np.zeros(len(ps))
    pos = ps["pos"].to_numpy()
    for _, ix in ps.groupby(group).indices.items():
        p = pos[ix]
        o = np.sort(p)
        dens[ix] = np.searchsorted(o, p + 100.0, side="right") - np.searchsorted(o, p - 100.0, side="left") - 1
    ps = ps.assign(density=dens / 0.2 / LANES[location])               # veh/km/lane
    return ps


def pneuma_per_second():
    from datasets.pneuma import SCORED, per_second, read_raw
    cache = os.path.join(EXT, "pneuma", "20181024_d1_0830_0900.per_second.csv.gz")
    ps = pd.read_csv(cache) if os.path.exists(cache) else per_second(read_raw(cache.replace(".per_second.csv.gz", ".csv")))
    return ps[ps["type"].isin(SCORED)].rename(columns={"density_50m": "density"})


def window_density(w, ps):
    """density of the vehicle at the window centre second"""
    key = ps.assign(tc=ps["t"].round().astype(int)).drop_duplicates(["vehicle_id", "tc"]).set_index(["vehicle_id", "tc"])["density"]
    k = pd.MultiIndex.from_arrays([w["trip"].to_numpy(), (w["t0"] + 5).round().astype(int).to_numpy()])
    return key.reindex(k).to_numpy()


def score_site(name, road, ps):
    w = windows_from_per_second(ps, road, "highway" if road == "freeway" else "urban")
    wh = windows_from_per_second(ps, road, "highway" if road == "freeway" else "urban", features=headway_features)
    w = w.assign(site=name, road_type=road, score=original_score(w[FEATS].to_numpy()),
                 score_headway=original_score(wh[FEATS].to_numpy()), density=window_density(w, ps))
    for i, t in enumerate(["speed", "accel", "prox", "wave"]):
        w[f"pts_{t}"] = 100 * WEIGHTS[i] * w[FEATS[i]]
    return w


def uah_reference():
    from datasets.uah import load_windows
    root = os.path.join(EXT, "UAH-DRIVESET-v1")
    if not os.path.isdir(root):
        return None
    win, _ = load_windows(root)
    win = win[win["behavior"].isin(["normal", "aggressive"])]
    wh, _ = load_windows(root, features=headway_features)
    wh = wh[wh["behavior"].isin(["normal", "aggressive"])]
    return win.assign(site="uah " + win["behavior"], road_type=win["road"], score=original_score(win[FEATS].to_numpy()),
                      score_headway=original_score(wh[FEATS].to_numpy()))


def summary(all_w):
    rows = []
    for site, g in all_w.groupby("site", sort=False):
        rows.append({"site": site, "road_type": g["road_type"].iloc[0], "windows": len(g),
                     "vehicles": g["trip"].nunique(), "speed_kmh_median": g["speed_kmh"].median(),
                     "score_median": g["score"].median(), "score_p95": g["score"].quantile(0.95),
                     "share_aggressive": (g["score"] >= THRESHOLDS[1]).mean(),
                     "headway_score_median": g["score_headway"].median(), "headway_share_aggressive": (g["score_headway"] >= THRESHOLDS[1]).mean(),
                     **{f"pts_{t}_mean": g[f"pts_{t}"].mean() for t in ["speed", "accel", "prox", "wave"] if f"pts_{t}" in g}})
    return pd.DataFrame(rows)


def density_table(all_w):
    ng = all_w[all_w["site"].str.startswith("ngsim")].copy()
    ng["density_bin"] = pd.cut(ng["density"], DENSITY_BINS, labels=DENSITY_LABELS, right=False)
    t = ng.groupby(["road_type", "density_bin"], observed=True).agg(
        windows=("score", "size"), speed_kmh=("speed_kmh", "median"), score=("score", "median"),
        score_headway=("score_headway", "median"), prox_pts=("pts_prox", "mean"),
        share_aggressive=("score", lambda s: (s >= THRESHOLDS[1]).mean()))
    pn = all_w[all_w["site"] == "pneuma athens"].copy()
    if len(pn):
        pn["density_bin"] = pd.qcut(pn["density"], 4, duplicates="drop").astype(str)
        tp = pn.groupby("density_bin").agg(windows=("score", "size"), speed_kmh=("speed_kmh", "median"),
                                           score=("score", "median"), score_headway=("score_headway", "median"),
                                           prox_pts=("pts_prox", "mean"), share_aggressive=("score", lambda s: (s >= THRESHOLDS[1]).mean()))
        tp.index = pd.MultiIndex.from_product([["pneuma (vehicles within 50 m)"], tp.index])
        t = pd.concat([t, tp])
    t.index.names = ["road_type", "density_bin"]
    return t


def plots(all_w, dens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sites = list(dict.fromkeys(all_w["site"]))
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, col, title in [(axs[0], "score", "gap in metres (reference)"), (axs[1], "score_headway", "time headway variant")]:
        data = [all_w.loc[all_w["site"] == s, col].to_numpy() for s in sites]
        ax.boxplot(data, vert=False, showfliers=False, whis=(5, 95))
        ax.set_yticks(range(1, len(sites) + 1), sites, fontsize=8)
        ax.axvline(THRESHOLDS[1], color="#9e2a2b", lw=0.8, ls="--")
        ax.set_title(title, fontsize=9)
    axs[1].set_xlabel("window score (box 25 to 75%, whiskers 5 to 95%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "unlabelled_scores.png"), dpi=140)
    plt.close(fig)

    d = dens.reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for road, col in [("freeway", "#3a6ea5"), ("arterial", "#56876d")]:
        g = d[d["road_type"] == road]
        ax.plot(g["density_bin"].astype(str), g["score"], marker="o", color=col, label=f"{road}, metres")
        ax.plot(g["density_bin"].astype(str), g["score_headway"], marker="s", ls="--", color=col, label=f"{road}, headway")
    ax.set_xlabel("density (veh/km/lane)")
    ax.set_ylabel("median window score")
    ax.set_title("NGSIM: index against traffic density", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "unlabelled_density.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    frames = []
    for name, file, loc, minutes, road, planar in SITES:
        if os.path.exists(os.path.join(EXT, file)):
            ps = ngsim_per_second(file, loc, minutes, planar)
            frames.append(score_site(name, road, ps))
            print(f"  {name}: {len(frames[-1])} windows", flush=True)
    if os.path.exists(os.path.join(EXT, "pneuma")):
        frames.append(score_site("pneuma athens", "urban downtown", pneuma_per_second()))
        print(f"  pneuma: {len(frames[-1])} windows", flush=True)
    u = uah_reference()
    all_w = pd.concat(frames + ([u] if u is not None else []), ignore_index=True)
    all_w.to_csv(os.path.join(DATA_DIR, "unlabelled_windows.csv.gz"), index=False)
    s = summary(all_w)
    s.to_csv(os.path.join(DATA_DIR, "unlabelled_summary.csv"), index=False)
    dens = density_table(all_w)
    dens.to_csv(os.path.join(DATA_DIR, "unlabelled_density.csv"))
    plots(all_w, dens)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(s.round(3).to_string(index=False))
        print()
        print(dens.round(3).to_string())
