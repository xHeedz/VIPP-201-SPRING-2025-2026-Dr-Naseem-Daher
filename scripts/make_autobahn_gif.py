"""
Top-down animation of the simulated German autobahn (seed 0, 10 Hz log): every car as a rectangle in its lane, coloured by
its planted type, trucks grey; the speed limit sign changes where the unlimited section starts.

    python scripts/make_autobahn_gif.py [--start 400] [--seconds 30] [--out results/figures/autobahn.gif]

The 10 Hz log has speed, lane and edge but no longitudinal position, so the position along the road is the running sum
of speed x 0.1 s per vehicle (exact on a straight road), offset by the edge (abl starts at 0 m, abf at 2000 m).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR  # noqa: E402

COL = {"conservative": "#4A6FA5", "normal": "#D9A441", "aggressive": "#7A1F2B", "truck": "#9A9A9A"}


def positions(raw):
    raw = raw.sort_values(["vehicle_id", "t"]).copy()
    raw["x"] = raw.groupby("vehicle_id")["speed_ms"].cumsum() * 0.1
    first_edge = raw.groupby("vehicle_id")["edge"].transform("first")
    raw.loc[first_edge == "abf", "x"] += 2000.0
    return raw


def main(start, seconds, out, fps=10):
    raw = pd.read_csv(os.path.join(DATA_DIR, "sumo_planted_de", "autobahn_medium_s0.raw10hz.csv.gz"))
    raw = positions(raw[raw["edge"].isin(["abl", "abf"]) | raw["edge"].str.startswith(":")])
    frames = np.round(np.arange(start, start + seconds, 1 / fps * 2), 1)        # every 0.2 s of simulation
    fig, ax = plt.subplots(figsize=(13, 3.2), dpi=90)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.2)
    fig.patch.set_facecolor("white")

    def draw(t):
        ax.cla()
        ax.set_xlim(1650, 2350)
        ax.set_ylim(-0.6, 2.6)
        ax.axis("off")
        for y in (-0.5, 0.5, 1.5, 2.5):
            ax.axhline(y, color="#BBBBBB", lw=1.2, ls="-" if y in (-0.5, 2.5) else (0, (6, 6)))
        ax.axvline(2000, color="#231A1C", lw=1.2, ls=":")
        ax.text(1990, 2.62, "120 km/h", ha="right", va="bottom", fontsize=11, color="#231A1C")
        ax.text(2010, 2.62, "no speed limit", ha="left", va="bottom", fontsize=11, color="#7A1F2B", weight="bold")
        f = raw[np.isclose(raw["t"], t)]
        for _, r in f.iterrows():
            length = 22.0 if r["label"] == "truck" else 11.0          # drawn 2.5x longer to be visible
            ax.add_patch(Rectangle((r["x"] - length, r["lane"] - 0.28), length, 0.56, color=COL.get(r["label"], "#999")))
        ax.text(1650, -0.85, f"t = {t:.0f} s   ·   lane 0 = right lane   ·   blue conservative · gold normal · red aggressive · "
                f"grey trucks (80 km/h)", fontsize=10, color="#555555", va="top")
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("written", out, len(frames), "frames")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=float, default=400.0)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--out", default=os.path.join(FIG_DIR, "autobahn.gif"))
    a = p.parse_args()
    main(a.start, a.seconds, a.out)
