"""
Planted driver types in SUMO: every NPC is conservative, normal or aggressive by
construction (one vType each, mix 20 / 60 / 20, label in the vehicle id), so the
index can be scored against a true label.

    python scripts/planted_drivers.py                      # every run (scenarios x densities x seeds)
    python scripts/planted_drivers.py --only highway --seeds 1 --workers 1

Scenarios (networks generated under env/sumo_scenarios/planted/):
    highway     3 lanes, 3 km, 120 km/h, lane drop 3 -> 2 at 2.2 km; densities low 1200 / medium 2400 / jam
                4500 veh/h (measured: 0%, 5% and 58% of rows before the drop below 30 km/h)
    jam         = highway at jam density (demand above the 2 lane capacity: queue, stop and go)
    urban       4 arm traffic light intersection, 2 lanes, 50 km/h
    merge       2 lane urban motorway (100 km/h) with an on ramp and a 300 m acceleration lane
    roundabout  single lane roundabout, 4 arms, 50 km/h
    weather     highway at medium density, every type with speedFactor x 0.8 and decel x 0.7
    us101       US-101 like weaving section: 5 lanes at 65 mph, on ramp onto an auxiliary lane, off ramp
                400 m later (US101_DEMAND veh/h on the main road and the ramp)

Driver types: IDM car following, SL2015 sublane lane changing (--lateral-resolution 0.8);
tau / accel / decel / speedFactor / lcAssertive / sigma from the plan table. Lateral
imprecision (lcSigma 0.2) is the SAME for every type, so wave can only separate types
through lane change behaviour, not through a planted lateral parameter.

Every vehicle is logged at 10 Hz (speed, SUMO acceleration, gap to the leader + minGap,
offset from the lane centre, lane, edge). Per run the per second table (datasets/sumo_log.py)
is saved to data/sumo_planted/; the raw 10 Hz log is kept for seed 0.
"""
import argparse
import os
import subprocess
import sys
import textwrap
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, SUMO_DIR, sumo_binary_name  # noqa: E402

SUMO_HOME = os.environ.get("SUMO_HOME", "")
if not SUMO_HOME:
    import sumo
    SUMO_HOME = sumo.SUMO_HOME
    os.environ["SUMO_HOME"] = SUMO_HOME
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
import traci  # noqa: E402
import traci.constants as tc  # noqa: E402

SUMO_BIN = os.path.join(SUMO_HOME, "bin", sumo_binary_name())
NETCONV = os.path.join(SUMO_HOME, "bin", "netconvert")
NET_DIR = os.path.join(SUMO_DIR, "planted")
OUT_DIR = os.path.join(DATA_DIR, "sumo_planted")

TYPES = {   # plan table (part 2) with tau x 0.7: calibrated on the time headway of NGSIM US-101
    # (KS distance 0.09 to 0.11 against 0.28 to 0.31 for the table values; scripts/planted_tau_sweep.py)
    "conservative": dict(tau=1.26, accel=1.5, decel=3.0, speedFactor=0.9, lcAssertive=0.5, sigma=0.2),
    "normal":       dict(tau=0.84, accel=2.6, decel=4.5, speedFactor=1.0, lcAssertive=1.0, sigma=0.5),
    "aggressive":   dict(tau=0.42, accel=3.5, decel=6.0, speedFactor=1.2, lcAssertive=3.0, sigma=0.7),
}
MIX = {"conservative": 0.2, "normal": 0.6, "aggressive": 0.2}
LC_SIGMA = 0.2
MIN_GAP = 2.5
SIM_END, WARMUP, STEP = 900.0, 120.0, 0.1

# scenario -> (network, demand level -> veh/h per route group, environment of the UAH agent)
DENSITY = {"low": 1200, "medium": 2400, "jam": 4500}
US101_DEMAND = (7000, 1200)      # veh/h on the main road and the on ramp (us101 scenario)       # highway, veh/h entering
RUNS = ([("highway", d) for d in DENSITY] +
        [("urban", "medium"), ("merge", "medium"), ("roundabout", "medium"), ("weather", "medium")])
ENVIRONMENT = {"highway": "highway", "jam": "highway", "merge": "highway", "weather": "weather", "us101": "highway",
               "urban": "urban", "roundabout": "urban"}


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).strip() + "\n")


def _netconvert(name, nodes, edges, extra=()):
    os.makedirs(NET_DIR, exist_ok=True)
    nod, edg, net = (os.path.join(NET_DIR, f"{name}.{x}.xml") for x in ("nod", "edg", "net"))
    _write(nod, f'<?xml version="1.0"?>\n<nodes>\n{nodes}\n</nodes>')
    _write(edg, f'<?xml version="1.0"?>\n<edges>\n{edges}\n</edges>')
    subprocess.run([NETCONV, "--node-files", nod, "--edge-files", edg, "--no-turnarounds", "true",
                    "--output-file", net, *extra], check=True, capture_output=True)
    return net


def networks():
    nets = {}
    nets["highway"] = _netconvert("highway", """
  <node id="start" x="0" y="0" type="priority"/>
  <node id="drop" x="2200" y="0" type="priority"/>
  <node id="end" x="3000" y="0" type="priority"/>""", """
  <edge id="hw1" from="start" to="drop" numLanes="3" speed="33.33"/>
  <edge id="hw2" from="drop" to="end" numLanes="2" speed="33.33"/>""")
    arms = {"n": (0, 400), "s": (0, -400), "e": (400, 0), "w": (-400, 0)}
    nets["urban"] = _netconvert("urban", '  <node id="c" x="0" y="0" type="traffic_light"/>\n' + "\n".join(
        f'  <node id="{k}" x="{x}" y="{y}" type="priority"/>' for k, (x, y) in arms.items()), "\n".join(
        f'  <edge id="{k}_in" from="{k}" to="c" numLanes="2" speed="13.89"/>\n'
        f'  <edge id="{k}_out" from="c" to="{k}" numLanes="2" speed="13.89"/>' for k in arms), ["--tls.guess", "true"])
    nets["merge"] = _netconvert("merge", """
  <node id="a" x="0" y="0" type="priority"/>
  <node id="r" x="700" y="-150" type="priority"/>
  <node id="m" x="1200" y="0" type="priority"/>
  <node id="m2" x="1500" y="0" type="priority"/>
  <node id="b" x="2500" y="0" type="priority"/>""", """
  <edge id="main1" from="a" to="m" numLanes="2" speed="27.78"/>
  <edge id="ramp" from="r" to="m" numLanes="1" speed="22.22"/>
  <edge id="acc" from="m" to="m2" numLanes="3" speed="27.78"/>
  <edge id="main2" from="m2" to="b" numLanes="2" speed="27.78"/>""")
    R, L = 30.0, 330.0
    ring = {"e": (R, 0), "n": (0, R), "w": (-R, 0), "s": (0, -R)}
    outer = {"e": (L, 0), "n": (0, L), "w": (-L, 0), "s": (0, -L)}
    order = ["e", "n", "w", "s"]                       # counterclockwise (right-hand traffic)
    nodes = "\n".join(f'  <node id="r{k}" x="{x}" y="{y}" type="priority"/>' for k, (x, y) in ring.items()) + "\n" + \
        "\n".join(f'  <node id="o{k}" x="{x}" y="{y}" type="priority"/>' for k, (x, y) in outer.items())
    edges = []
    for i, k in enumerate(order):
        nk = order[(i + 1) % 4]
        a0, a1 = np.deg2rad(90 * i), np.deg2rad(90 * (i + 1))
        shape = " ".join(f"{R * np.cos(a):.2f},{R * np.sin(a):.2f}" for a in np.linspace(a0, a1, 7))
        edges.append(f'  <edge id="r_{k}{nk}" from="r{k}" to="r{nk}" numLanes="1" speed="8.33" shape="{shape}"/>')
        edges.append(f'  <edge id="{k}_in" from="o{k}" to="r{k}" numLanes="1" speed="13.89"/>')
        edges.append(f'  <edge id="{k}_out" from="r{k}" to="o{k}" numLanes="1" speed="13.89"/>')
    edges.append('  <roundabout nodes="re rn rw rs" edges="r_en r_nw r_ws r_se"/>')
    nets["roundabout"] = _netconvert("roundabout", nodes, "\n".join(edges))
    # US-101 like weaving section: 5 lanes at 65 mph, on ramp joining an auxiliary lane, off ramp 400 m later
    nets["us101"] = _netconvert("us101", """
  <node id="a" x="0" y="0" type="priority"/>
  <node id="r" x="800" y="-150" type="priority"/>
  <node id="m" x="1000" y="0" type="priority"/>
  <node id="m2" x="1400" y="0" type="priority"/>
  <node id="x" x="1600" y="-150" type="priority"/>
  <node id="b" x="2600" y="0" type="priority"/>""", """
  <edge id="main1" from="a" to="m" numLanes="5" speed="29.06"/>
  <edge id="ramp" from="r" to="m" numLanes="1" speed="22.22"/>
  <edge id="aux" from="m" to="m2" numLanes="6" speed="29.06"/>
  <edge id="main2" from="m2" to="b" numLanes="5" speed="29.06"/>
  <edge id="off" from="m2" to="x" numLanes="1" speed="22.22"/>""")
    return nets


def routes(scenario):
    """[(route_id, edges, share of the scenario demand)]"""
    if scenario in ("highway", "jam", "weather"):
        return [("r0", "hw1 hw2", 1.0)]
    if scenario == "merge":
        return [("main", "main1 acc main2", 0.8), ("onramp", "ramp acc main2", 0.2)]
    if scenario == "us101":      # shares of US101_DEMAND (main, ramp); 10% of each leaves at the off ramp
        m, r = US101_DEMAND
        tot = m + r
        return [("thru", "main1 aux main2", 0.9 * m / tot), ("exit", "main1 aux off", 0.1 * m / tot),
                ("rthru", "ramp aux main2", 0.9 * r / tot), ("rexit", "ramp aux off", 0.1 * r / tot)]
    arms = ["n", "e", "s", "w"]
    out = []
    if scenario == "urban":
        for a in arms:
            for b in arms:
                if a != b:
                    out.append((f"{a}{b}", f"{a}_in {b}_out", 1 / 12))
    if scenario == "roundabout":
        order = ["e", "n", "w", "s"]
        for a in order:
            for b in order:
                if a == b:
                    continue
                i, path = order.index(a), [f"{a}_in"]
                while order[i] != b:
                    path.append(f"r_{order[i]}{order[(i + 1) % 4]}")
                    i = (i + 1) % 4
                out.append((f"{a}{b}", " ".join(path + [f"{b}_out"]), 1 / 12))
    return out


def demand(scenario, density):
    if scenario in ("highway", "weather"):
        return DENSITY[density]
    if scenario == "us101":
        return sum(US101_DEMAND)
    return {"jam": DENSITY["jam"], "urban": 1200, "merge": 3000, "roundabout": 1000}[scenario]


def write_routes(path, scenario, density):
    scale = dict(speedFactor=0.8, decel=0.7) if scenario == "weather" else {}
    lines = ['<?xml version="1.0"?>', "<routes>"]
    for name, p in TYPES.items():
        q = dict(p)
        for k, f in scale.items():
            q[k] = round(q[k] * f, 3)
        attrs = " ".join(f'{k}="{v}"' for k, v in q.items())
        lines.append(f'  <vType id="{name}" carFollowModel="IDM" laneChangeModel="SL2015" lcSigma="{LC_SIGMA}" '
                     f'minGap="{MIN_GAP}" speedDev="0.1" {attrs}/>')
    total = demand(scenario, density)
    for rid, edges, share in routes(scenario):
        lines.append(f'  <route id="{rid}" edges="{edges}"/>')
        for name, mix in MIX.items():
            vph = total * share * mix
            # highway family: random start lane, so traffic has to merge at the lane drop ("best" would put
            # every car in the lanes that continue and cap insertion at their capacity)
            lane = "random" if scenario in ("highway", "jam", "weather") or rid == "thru" else "best"
            if rid == "exit":       # cars bound for the off ramp start on the right, as drivers who plan ahead:
                lane = "0"          # starting anywhere made them stop at the end of the auxiliary lane and deadlock it
            lines.append(f'  <flow id="{name}_{rid}" type="{name}" route="{rid}" begin="0" end="{SIM_END - 120}" '
                         f'vehsPerHour="{vph:.1f}" departLane="{lane}" departSpeed="desired"/>')
    lines.append("</routes>")
    _write(path, "\n".join(lines))


def run(job):
    scenario, density, seed, nets = job
    net = nets["highway" if scenario in ("jam", "weather") else scenario]
    tag = f"{scenario}_{density}_s{seed}"
    rou = os.path.join(NET_DIR, f"{tag}.rou.xml")
    write_routes(rou, scenario, density)
    traci.start([SUMO_BIN, "-n", net, "-r", rou, "--step-length", str(STEP), "--seed", str(seed),
                 "--lateral-resolution", "0.8", "--no-step-log", "true", "--no-warnings", "true",
                 "--collision.action", "warn", "--time-to-teleport", "300"], label=tag)
    con = traci.getConnection(tag)
    vars_ = [tc.VAR_SPEED, tc.VAR_ACCELERATION, tc.VAR_LANEPOSITION_LAT, tc.VAR_LANE_INDEX, tc.VAR_ROAD_ID]
    rows = []
    try:
        while con.simulation.getTime() < SIM_END:
            con.simulationStep()
            for vid in con.simulation.getDepartedIDList():
                con.vehicle.subscribe(vid, vars_)
                con.vehicle.subscribeLeader(vid, 100.0)
            t = round(con.simulation.getTime(), 1)
            if t < WARMUP:
                continue
            for vid, r in con.vehicle.getAllSubscriptionResults().items():
                lead = r.get(tc.VAR_LEADER)
                gap = float(lead[1]) + MIN_GAP if lead and lead[0] else 0.0
                rows.append((t, vid, vid.split("_")[0], r[tc.VAR_SPEED], r[tc.VAR_ACCELERATION], gap,
                             r[tc.VAR_LANEPOSITION_LAT], r[tc.VAR_LANE_INDEX], r[tc.VAR_ROAD_ID]))
    finally:
        con.close()
    raw = pd.DataFrame(rows, columns=["t", "vehicle_id", "label", "speed_ms", "accel_ms2", "gap_m", "lat_m",
                                      "lane", "edge"])
    from datasets.sumo_log import per_second
    ps = pd.concat([per_second(g).assign(vehicle_id=v, label=g["label"].iloc[0])
                    for v, g in raw.groupby("vehicle_id", sort=False) if len(g) >= 30], ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    ps.to_csv(os.path.join(OUT_DIR, f"{tag}.per_second.csv.gz"), index=False)
    if seed == 0:
        raw.to_csv(os.path.join(OUT_DIR, f"{tag}.raw10hz.csv.gz"), index=False)
    return tag, len(raw), raw["vehicle_id"].nunique()


def jobs(only=None, seeds=10):
    nets = networks()
    out = []
    for scenario, density in RUNS:
        if scenario == "highway" and density == "jam":
            scenario = "jam"
        if only and scenario not in only:
            continue
        out += [(scenario, density, s, nets) for s in range(seeds)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    js = jobs(a.only, a.seeds)
    print(f"{len(js)} runs")
    with Pool(a.workers) as pool:
        for tag, n, nv in pool.imap_unordered(run, js):
            print(f"  {tag}: {n:,} rows at 10 Hz, {nv} vehicles", flush=True)
