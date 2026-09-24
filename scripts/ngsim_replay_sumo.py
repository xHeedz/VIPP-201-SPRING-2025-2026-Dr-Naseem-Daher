"""
Replay NGSIM trajectories inside SUMO, so recorded vehicles appear as SUMO
cars (real length and width) instead of simulated traffic.

    python scripts/ngsim_replay_sumo.py --file ngsim.csv --location us-101 --minutes 2 --gui

A straight road with the same number of lanes is built. Every 0.1 s each
recorded vehicle is placed at its recorded position with TraCI moveToXY, so
SUMO draws the real traffic, and anything added on top (the RL ego car, the
noisy assessor, the shockwave factor) sees real vehicles around it.

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
from paths import sumo_binary_name  # noqa: E402
from datasets.ngsim import read_raw, trajectories  # noqa: E402


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


def replay(traj, raw, gui=False, max_steps=None):
    lanes = sorted(traj["lane"].unique())
    n_lanes = int(max(lanes))
    work = tempfile.mkdtemp(prefix="ngsim_replay_")
    cfg = build(work, float(traj["pos"].max()), n_lanes)
    binary = os.path.join(SUMO_HOME, "bin", sumo_binary_name(gui))
    traci.start([binary, "-c", cfg] + (["--start", "--delay", "50"] if gui else []))
    lane_y = {}
    for i in range(n_lanes):
        shape = traci.lane.getShape(f"road_{i}")
        lane_y[n_lanes - i] = shape[0][1]                 # NGSIM lane L -> SUMO index n_lanes - L
    centre = traj.groupby("lane")["x_lat"].median()
    size = raw.drop_duplicates("Vehicle_ID").set_index("Vehicle_ID")
    frames = traj.groupby("t")
    active, placed = set(), 0
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
                    active.add(vid)
                    placed += 1
                y = lane_y.get(int(r.lane), 0.0) - (r.x_lat - centre.get(r.lane, r.x_lat))
                traci.vehicle.moveToXY(vid, "road", -1, float(r.pos), float(y), angle=90, keepRoute=2)
                traci.vehicle.setSpeed(vid, float(r.speed))
            for vid in active - present:
                traci.vehicle.remove(vid)
            active &= present
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
    a = ap.parse_args()
    raw = read_raw(a.file, a.location, a.minutes)
    traj = trajectories(raw)
    n, steps = replay(traj, raw, a.gui)
    print(f"replayed {n} recorded vehicles over {steps} steps ({steps / 10:.0f} s)")
