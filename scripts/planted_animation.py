"""
Animation of the calibrated planted drivers at the lane drop in the jam (scripts/planted_drivers.py),
drawn three times: true driver type, label from the index with the gap in metres, label from the
time headway variant (cut offs 29 / 42). Each label uses the mean index over the vehicle's last 10 s,
computed the UAH way (datasets/sumo_log.py per_second, then index_features).

    python scripts/planted_animation.py      # writes results/figures/planted_jam_animation.gif
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import planted_drivers as P  # noqa: E402
from paths import FIG_DIR  # noqa: E402
from model.aggressiveness_model import THRESHOLDS, headway_features, index_features, original_score  # noqa: E402

T0, T1, DT = 420.0, 480.0, 0.5            # animated interval (s) and frame step
X0, X1 = 1950.0, 2280.0                   # road section around the lane drop at 2200 m
COL = {"conservative": "#2E9E4F", "normal": "#E0B020", "aggressive": "#C8323C", None: "#9A9A9A"}


def simulate(seed=3):
    import traci.constants as tc
    nets = {"highway": P.networks()["highway"]}
    rou = os.path.join(P.NET_DIR, "anim.rou.xml")
    P.write_routes(rou, "jam", "jam")
    P.traci.start([P.SUMO_BIN, "-n", nets["highway"], "-r", rou, "--step-length", "0.1", "--seed", str(seed),
                   "--lateral-resolution", "0.8", "--no-step-log", "true", "--no-warnings", "true"], label="anim")
    con = P.traci.getConnection("anim")
    vars_ = [tc.VAR_SPEED, tc.VAR_ACCELERATION, tc.VAR_LANEPOSITION_LAT, tc.VAR_LANE_INDEX, tc.VAR_ROAD_ID,
             tc.VAR_POSITION, tc.VAR_ANGLE]
    rows = []
    try:
        while con.simulation.getTime() < T1:
            con.simulationStep()
            for vid in con.simulation.getDepartedIDList():
                con.vehicle.subscribe(vid, vars_)
                con.vehicle.subscribeLeader(vid, 100.0)
            t = round(con.simulation.getTime(), 1)
            if t < T0 - 40:
                continue
            for vid, r in con.vehicle.getAllSubscriptionResults().items():
                lead = r.get(tc.VAR_LEADER)
                gap = float(lead[1]) + P.MIN_GAP if lead and lead[0] else 0.0
                x, y = r[tc.VAR_POSITION]
                rows.append((t, vid, vid.split("_")[0], r[tc.VAR_SPEED], r[tc.VAR_ACCELERATION], gap,
                             r[tc.VAR_LANEPOSITION_LAT], r[tc.VAR_LANE_INDEX], r[tc.VAR_ROAD_ID], x, y))
    finally:
        con.close()
    return pd.DataFrame(rows, columns=["t", "vehicle_id", "label", "speed_ms", "accel_ms2", "gap_m", "lat_m", "lane",
                                       "edge", "x", "y"])


def labels(raw):
    """label per vehicle and whole second: mean per second index over the last 10 s"""
    from datasets.sumo_log import per_second
    out = []
    for vid, g in raw.groupby("vehicle_id"):
        ps = per_second(g)
        if len(ps) < 3:
            continue
        for name, feats in [("metres", index_features), ("headway", headway_features)]:
            s = pd.Series(original_score(feats(ps["speed_kmh"], ps["accel"], ps["gap_m"], ps["wave_m"])))
            m = s.rolling(10, min_periods=5).mean()
            for t, v in zip(ps["t"], m):
                if np.isfinite(v):
                    cat = "conservative" if v < THRESHOLDS[0] else ("normal" if v < THRESHOLDS[1] else "aggressive")
                    out.append((vid, t, name, cat))
    return pd.DataFrame(out, columns=["vehicle_id", "sec", "prox", "cat"])


def render(raw, lab, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    lab = lab.set_index(["vehicle_id", "sec", "prox"])["cat"]
    yc = raw.groupby("lane")["y"].median().sort_index()          # lane centres (lane 0 is the rightmost)
    half = 1.6
    ylo, yhi = yc.min() - half - 0.6, yc.max() + half + 1.6
    frames = []
    rows = [("true driver type", None), ("index, gap in metres", "metres"), ("index, time headway", "headway")]
    for t in np.arange(T0, T1 + 1e-9, DT):
        f = raw[np.isclose(raw["t"], t) & (raw["x"] >= X0 - 5) & (raw["x"] <= X1 + 5)]
        fig, axs = plt.subplots(3, 1, figsize=(9.6, 4.2), dpi=100)
        fig.patch.set_facecolor("#FBF8F6")
        for ax, (title, prox) in zip(axs, rows):
            ax.set_facecolor("#2B2B2B")
            ax.set_xlim(X0, X1)
            ax.set_ylim(ylo, yhi)
            ax.set_aspect("auto")
            lanes = list(yc.values)
            for a, b in zip(lanes[:-1], lanes[1:]):
                ax.plot([X0, X1], [(a + b) / 2] * 2, color="#BBBBBB", lw=0.7, ls=(0, (6, 6)))
            top = max(lanes)
            ax.add_patch(Rectangle((2200, top - half), X1 - 2200, 2 * half + 3, color="#4A4A4A", lw=0))
            for _, v in f.iterrows():
                cat = v["label"] if prox is None else lab.get((v["vehicle_id"], round(v["t"] - 0.5), prox), None)
                ax.add_patch(Rectangle((v["x"] - 4.5, v["y"] - 0.9), 4.5, 1.8, color=COL[cat], lw=0))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(X0 + 3, yhi - 0.9, title, color="white", fontsize=9, va="center", family="sans-serif")
        n = len(f)
        fig.text(0.01, 0.01, f"t = {t:5.1f} s   {n} vehicles on 330 m before the lane drop (grey: the closed lane)   "
                 "green conservative, yellow normal, red aggressive", fontsize=8, color="#5E5456")
        fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.07, hspace=0.06)
        fig.canvas.draw()
        frames.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB"))
        plt.close(fig)
    pal = [fr.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for fr in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(DT * 1000 / 2), loop=0, optimize=True)
    return len(frames)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    raw = simulate()
    lab = labels(raw)
    path = os.path.join(FIG_DIR, "planted_jam_animation.gif")
    n = render(raw, lab, path)
    sec = raw[(raw["t"] >= T0) & (raw["x"] >= X0) & (raw["x"] <= 2200)]
    print(f"{n} frames -> {path} ({os.path.getsize(path) / 1e6:.1f} MB); upstream of the drop: median speed "
          f"{sec['speed_ms'].median() * 3.6:.1f} km/h")
    shares = lab[lab["sec"].between(T0, T1)].merge(raw.groupby("vehicle_id")["label"].first().rename("truth"),
                                                   left_on="vehicle_id", right_index=True)
    print(shares.groupby(["prox", "truth"])["cat"].value_counts(normalize=True).unstack().round(2).to_string())
