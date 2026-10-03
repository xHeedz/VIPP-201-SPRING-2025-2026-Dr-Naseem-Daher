"""
Replay NGSIM trajectories inside SUMO, so recorded vehicles appear as SUMO
cars (real length and width) instead of simulated traffic.

    python scripts/ngsim_replay_sumo.py --file ngsim.csv --location us-101 --minutes 2 --gui
    python scripts/ngsim_replay_sumo.py --file ngsim.csv --location us-101 --minutes 2 --gui --ellipses

A straight road with the same number of lanes is built. Every 0.1 s each
recorded vehicle is placed at its recorded position with TraCI moveToXY, so
SUMO draws the real traffic, and anything added on top (the RL ego car, the
noisy assessor, the shockwave factor) sees real vehicles around it.

With --ellipses every vehicle is drawn as two ellipses (model/ellipses.py)
instead of a SUMO car: its body, and a see-through influence zone that grows
with speed and with the vehicle's Aggressiveness Index, coloured green,
yellow or red, plus a purple ring while it disturbs the traffic behind it.
The index is the trained dynamic weight agent (data/dynamic_weight_agent.json,
from scripts/train_uah.py) when it exists, otherwise the original hand-set
index, averaged over the last 10 s (the window length the agent was trained on).

--color-by chooses what the colours mean:
  context   (default) the index relative to the traffic within 100 m (model/context.py):
            a vehicle that moves like the jam around it is normal
  absolute  the index on its own, as trained on UAH-DriveSet
  rank      display only: the top 10 percent at each moment red, the bottom 10 percent green

Lanes: NGSIM numbers lanes from the left (1 = median side); SUMO numbers them
from the right (0 = shoulder side). Auxiliary and ramp lanes are kept as
extra lanes on the right.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from paths import DATA_DIR, sumo_binary_name  # noqa: E402
from datasets.ngsim import environment_for, read_raw, trajectories  # noqa: E402
from model.aggressiveness_model import index_features  # noqa: E402
from model import ellipses  # noqa: E402
from model.context import context_adjusted, rank_categories, scorer_for  # noqa: E402
from model.shockwave import disruption_table  # noqa: E402


def _sumo_home():
    home = os.environ.get("SUMO_HOME")
    if home:
        return home
    import sumo
    return sumo.SUMO_HOME


SUMO_HOME = _sumo_home()
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci  # noqa: E402

LANE_W = 3.66     # 12 ft, the NGSIM lane width
DISRUPTIVE_FROM = 0.12   # 10 s mean disruption above which the purple ring is shown


def build(work, length_m, n_lanes):
    nb = os.path.join(SUMO_HOME, "bin", "netconvert" + (".exe" if os.name == "nt" else ""))
    with open(os.path.join(work, "r.nod.xml"), "w") as f:
        f.write(f'<nodes><node id="a" x="-50" y="0"/><node id="b" x="{length_m + 50:.1f}" y="0"/></nodes>')
    with open(os.path.join(work, "r.edg.xml"), "w") as f:
        f.write(f'<edges><edge id="road" from="a" to="b" numLanes="{n_lanes}" speed="40" width="{LANE_W}"/></edges>')
    net = os.path.join(work, "r.net.xml")
    subprocess.run([nb, "-n", os.path.join(work, "r.nod.xml"), "-e", os.path.join(work, "r.edg.xml"), "-o", net],
                   check=True, capture_output=True)
    rou = os.path.join(work, "r.rou.xml")
    with open(rou, "w") as f:
        f.write('<routes><vType id="rec" color="0,0.6,1"/><route id="r" edges="road"/></routes>')
    cfg = os.path.join(work, "r.sumocfg")
    with open(cfg, "w") as f:
        f.write(f'<configuration><input><net-file value="{net}"/><route-files value="{rou}"/></input>'
                '<time><step-length value="0.1"/></time>'
                '<processing><collision.action value="none"/></processing>'
                '<report><no-step-log value="true"/><no-warnings value="true"/></report></configuration>')
    return cfg


def add_scores(traj, location, color_by="context", smooth_s=10.0):
    """Adds these columns, all averaged over the last smooth_s seconds of each vehicle
    (10 s by default, the window length the agent was trained on in train_uah.py):
      ai          the score that sizes the influence ellipse: the context score for
                  --color-by context and rank, the plain index for absolute
      disruption  how much the vehicle is disturbing the traffic behind it right now
                  (model/shockwave.py; its mean over a whole run is the shockwave factor)
      cat         conservative, normal or aggressive, which sets the colour
    Returns (traj, name of the scorer used)."""
    env = environment_for(location)
    score, threshold, normal_mean, name = scorer_for(env, DATA_DIR)
    f = index_features(traj["speed"] * 3.6, traj["accel_1hz"], traj["gap"], traj["wave"])
    own = score(f)
    traj["ai"] = own if color_by == "absolute" else context_adjusted(traj, own, normal_mean)
    d = disruption_table(traj[["t", "vehicle_id", "lane", "pos", "speed", "accel"]])
    traj = traj.merge(d[["t", "vehicle_id", "disruption"]], on=["t", "vehicle_id"], how="left")
    traj["disruption"] = traj["disruption"].fillna(0.0)
    win = max(1, int(round(smooth_s / 0.1)))
    traj = traj.sort_values(["vehicle_id", "t"])
    for col in ("ai", "disruption"):
        traj[col] = traj.groupby("vehicle_id")[col].transform(lambda s: s.rolling(win, min_periods=1).mean())
    traj = traj.sort_values(["t", "vehicle_id"]).reset_index(drop=True)
    if color_by == "rank":
        traj["cat"] = rank_categories(traj, "ai")
    else:
        traj["cat"] = traj["ai"].apply(lambda v: ellipses.category(v, threshold))
    return traj, f"{name}, colours by {color_by}, aggressive from {threshold:.1f}"


def draw_ellipses(vid, x_front, y, length, width, speed, ai, disruption, cat, new, ringed):
    """Adds (new=True) or moves the polygons that stand for vehicle vid: its influence
    zone and its body, plus a purple ring only while the vehicle is disturbing the
    traffic behind it (disruption >= DISRUPTIVE_FROM). ringed is the set of vehicle
    ids that currently have a ring; it is updated here."""
    zone = ellipses.influence_ellipse(x_front, y, length, width, speed, ai)
    body = ellipses.body_ellipse(x_front, y, length, width)
    zone_color = ellipses.color_of(cat, alpha=70)
    body_color = ellipses.color_of(cat, alpha=255)
    if new:
        traci.polygon.add(f"zone_{vid}", zone, zone_color, fill=True, layer=4)
        traci.polygon.add(f"body_{vid}", body, body_color, fill=True, layer=5)
    else:
        for name, shape, color in (("zone", zone, zone_color), ("body", body, body_color)):
            traci.polygon.setShape(f"{name}_{vid}", shape)
            traci.polygon.setColor(f"{name}_{vid}", color)
    # the ring is added and removed rather than made transparent: SUMO draws
    # a fully transparent outline in white
    ring = ellipses.body_ellipse(x_front + 0.3 * length, y, 1.6 * length, 2.2 * width)
    if disruption >= DISRUPTIVE_FROM:
        if vid in ringed:
            traci.polygon.setShape(f"ring_{vid}", ring)
        else:
            traci.polygon.add(f"ring_{vid}", ring, ellipses.PURPLE + (255,), fill=False, layer=6, lineWidth=0.5)
            ringed.add(vid)
    elif vid in ringed:
        traci.polygon.remove(f"ring_{vid}")
        ringed.discard(vid)


def replay(traj, raw, gui=False, max_steps=None, draw=False, delay_ms=50, screenshot=None):
    lanes = sorted(traj["lane"].unique())
    n_lanes = int(max(lanes))
    work = tempfile.mkdtemp(prefix="ngsim_replay_")
    cfg = build(work, float(traj["pos"].max()), n_lanes)
    binary = os.path.join(SUMO_HOME, "bin", sumo_binary_name(gui))
    gui_args = ["--start", "--delay", str(delay_ms)] + (["--quit-on-end", "--window-size", "1600,600"] if screenshot else [])
    try:
        # sumo-gui can take a while to open its window on a Mac (XQuartz), so wait up to 60 s
        traci.start([binary, "-c", cfg] + (gui_args if gui else []), numRetries=60)
    except traci.exceptions.FatalTraCIError:
        sys.exit("SUMO did not start. With --gui on a Mac this usually means XQuartz is missing: "
                 "install it from https://www.xquartz.org, log out and back in, then try again.")
    lane_y = {}
    for i in range(n_lanes):
        shape = traci.lane.getShape(f"road_{i}")
        lane_y[n_lanes - i] = shape[0][1]                 # NGSIM lane L -> SUMO index n_lanes - L
    if gui:
        # start zoomed in on a 250 m stretch in the middle, so the vehicles are big enough to see
        mid, ys = 0.5 * float(traj["pos"].max()), list(lane_y.values())
        traci.gui.setBoundary("View #0", mid - 125, min(ys) - 20, mid + 125, max(ys) + 20)
    centre = traj.groupby("lane")["x_lat"].median()
    size = raw.drop_duplicates("Vehicle_ID").set_index("Vehicle_ID")
    frames = traj.groupby("t")
    active, ringed, placed = set(), set(), 0
    try:
        for step, (t, rows) in enumerate(frames):
            if max_steps and step >= max_steps:
                break
            present = set()
            for r in rows.itertuples():
                vid = str(r.vehicle_id)
                present.add(vid)
                if vid not in active:
                    traci.vehicle.add(vid, "r", typeID="rec", departSpeed="0")
                    s = size.loc[r.vehicle_id]
                    traci.vehicle.setLength(vid, float(s["v_Length"]) * 0.3048)
                    if "v_Width" in s:
                        traci.vehicle.setWidth(vid, float(s["v_Width"]) * 0.3048)
                    traci.vehicle.setSpeedMode(vid, 0)
                    if draw:
                        traci.vehicle.setColor(vid, (0, 0, 0, 0))    # hide the SUMO car; the ellipses replace it
                    placed += 1
                y = lane_y.get(int(r.lane), 0.0) - (r.x_lat - centre.get(r.lane, r.x_lat))
                traci.vehicle.moveToXY(vid, "road", -1, float(r.pos), float(y), angle=90, keepRoute=2)
                traci.vehicle.setSpeed(vid, float(r.speed))
                if draw:
                    s = size.loc[r.vehicle_id]
                    draw_ellipses(vid, float(r.pos), float(y), float(s["v_Length"]) * 0.3048,
                                  float(s.get("v_Width", 6.0)) * 0.3048, float(r.speed), float(r.ai),
                                  float(r.disruption), r.cat, vid not in active, ringed)
                active.add(vid)
            for vid in active - present:
                traci.vehicle.remove(vid)
                if draw:
                    traci.polygon.remove(f"zone_{vid}")
                    traci.polygon.remove(f"body_{vid}")
                    if vid in ringed:
                        traci.polygon.remove(f"ring_{vid}")
                        ringed.discard(vid)
            active &= present
            traci.simulationStep()
        if gui and screenshot:
            # picture of a 120 m stretch centred on the vehicles of the last frame, e.g. for slides
            mid = float(rows["pos"].median())
            ys = list(lane_y.values())
            traci.gui.setBoundary("View #0", mid - 60, min(ys) - 8, mid + 60, max(ys) + 8)
            traci.gui.screenshot("View #0", os.path.abspath(screenshot))
            traci.simulationStep()
    finally:
        traci.close()
    return placed, step + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--location", default=None)
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--ellipses", action="store_true", help="draw body and influence ellipses coloured by the index")
    ap.add_argument("--color-by", choices=["context", "absolute", "rank"], default="context",
                    help="context: relative to nearby traffic (default); absolute: the index alone; rank: display only")
    ap.add_argument("--delay", type=int, default=50, help="milliseconds between steps in the SUMO window")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many 0.1 s steps")
    ap.add_argument("--screenshot", default=None, help="with --gui: save a PNG of the SUMO view at the end")
    a = ap.parse_args()
    raw = read_raw(a.file, a.location, a.minutes)
    traj = trajectories(raw)
    if a.ellipses:
        traj, described = add_scores(traj, a.location, a.color_by)
        print(f"ellipses: {described}")
        print("share of vehicle-moments per colour:", traj["cat"].value_counts(normalize=True).round(3).to_dict())
    n, steps = replay(traj, raw, a.gui, a.max_steps, a.ellipses, a.delay, a.screenshot)
    print(f"replayed {n} recorded vehicles over {steps} steps ({steps / 10:.0f} s)")
