"""
Shockwave experiment: do aggressive and overly conservative drivers both
create upstream disturbances, and how much?

A 3 km motorway narrows from 2 lanes to 1 for its last 500 m, and demand is
set close to the capacity of that single lane, which is where waves form. The share of
aggressive or conservative drivers is raised from 0 to 30 percent (the rest
are normal), with two random seeds per setting. For each run the script
records every vehicle every 0.5 s and computes:
  per driver    Aggressiveness Index (model/aggressiveness_model.py) and
                shockwave factor (model/shockwave.py)
  per run       mean speed, speed spread, throughput, stop-and-go vehicles,
                and how fast slowdowns travel upstream

    python scripts/shockwave_experiment.py            # full sweep, about 5 minutes
    python scripts/shockwave_experiment.py --quick    # one seed, fewer settings

Outputs data/shockwave_runs.csv, data/shockwave_drivers.csv and three figures
in results/figures/.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from paths import DATA_DIR, FIG_DIR, sumo_binary_name  # noqa: E402
from model.aggressiveness_model import AggressivenessModel  # noqa: E402
from model.shockwave import shockwave_factor, traffic_metrics  # noqa: E402


def _sumo_home():
    home = os.environ.get("SUMO_HOME")
    if home:
        return home
    import sumo
    return sumo.SUMO_HOME


SUMO_HOME = _sumo_home()
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci  # noqa: E402

ROAD_M = 3000.0
DT = 0.5
SIM_S = 480.0
WARMUP_S = 120.0
DEMAND_VEH_H = 2800   # above the single-lane capacity, so queues and waves form
BOTTLENECK_M = 500.0   # the last 500 m narrow from 2 lanes to 1

# Krauss car-following. sigma is driver imperfection: random, unnecessary slowdowns.
VTYPES = {
    "normal": 'accel="2.6" decel="4.5" sigma="0.5" tau="1.2" minGap="2.5" speedFactor="1.0" speedDev="0.1"',
    "aggressive": ('accel="4.0" decel="7.0" sigma="0.5" tau="0.6" minGap="1.0" speedFactor="1.2" speedDev="0.1" '
                   'lcSpeedGain="3" lcAssertive="3" lcCooperative="0.2"'),
    "conservative": ('accel="1.5" decel="4.5" sigma="0.9" tau="2.2" minGap="4.0" speedFactor="0.75" speedDev="0.05" '
                     'lcSpeedGain="0.2"'),
}


def build(work, shares, seed):
    nb = os.path.join(SUMO_HOME, "bin", "netconvert" + (".exe" if os.name == "nt" else ""))
    with open(os.path.join(work, "n.nod.xml"), "w") as f:
        f.write(f'<nodes><node id="a" x="0" y="0"/><node id="m" x="{ROAD_M - BOTTLENECK_M}" y="0"/>'
                f'<node id="b" x="{ROAD_M}" y="0"/></nodes>')
    with open(os.path.join(work, "n.edg.xml"), "w") as f:
        f.write('<edges><edge id="hw" from="a" to="m" numLanes="2" speed="33.33"/>'
                '<edge id="neck" from="m" to="b" numLanes="1" speed="33.33"/></edges>')
    net = os.path.join(work, "n.net.xml")
    subprocess.run([nb, "-n", os.path.join(work, "n.nod.xml"), "-e", os.path.join(work, "n.edg.xml"), "-o", net],
                   check=True, capture_output=True)
    types = "\n".join(f'  <vType id="{k}" carFollowModel="Krauss" {v}/>' for k, v in VTYPES.items())
    flows = "\n".join(f'  <flow id="f_{k}" type="{k}" route="r" begin="0" end="{SIM_S}" '
                      f'vehsPerHour="{DEMAND_VEH_H * s:.1f}" departLane="random" departSpeed="desired"/>'
                      for k, s in shares.items() if s > 0)
    rou = os.path.join(work, "n.rou.xml")
    with open(rou, "w") as f:
        f.write(f'<routes>\n{types}\n  <route id="r" edges="hw neck"/>\n{flows}\n</routes>')
    cfg = os.path.join(work, "n.sumocfg")
    with open(cfg, "w") as f:
        f.write(f'<configuration><input><net-file value="{net}"/><route-files value="{rou}"/></input>'
                f'<time><step-length value="{DT}"/></time>'
                f'<report><no-step-log value="true"/><no-warnings value="true"/></report></configuration>')
    return cfg


def simulate(shares, seed):
    work = tempfile.mkdtemp(prefix="shockwave_")
    cfg = build(work, shares, seed)
    traci.start([os.path.join(SUMO_HOME, "bin", sumo_binary_name()), "-c", cfg, "--seed", str(seed)],
                label=f"sw{seed}{id(shares)}")
    c = traci.getConnection(f"sw{seed}{id(shares)}")
    rows = []
    model = AggressivenessModel()
    try:
        while c.simulation.getTime() < SIM_S:
            c.simulationStep()
            t = c.simulation.getTime()
            for vid in c.vehicle.getIDList():
                spd, acc = c.vehicle.getSpeed(vid), c.vehicle.getAcceleration(vid)
                lead = c.vehicle.getLeader(vid, 100.0)
                gap = lead[1] if lead and lead[0] else 0.0     # 0 = no leader, as in AggressivenessModel
                wave = abs(c.vehicle.getLateralLanePosition(vid))
                ai, _ = model.get_ai_score(spd * 3.6, acc, gap, wave)
                rows.append((t, vid, c.vehicle.getLaneIndex(vid), c.vehicle.getPosition(vid)[0],
                             spd, acc, c.vehicle.getTypeID(vid), ai))
    finally:
        c.close()
    return pd.DataFrame(rows, columns=["t", "vehicle_id", "lane", "pos", "speed", "accel", "type", "ai"])


def conditions(quick):
    fracs = [0.2] if quick else [0.1, 0.2, 0.3]
    out = [("baseline", 0.0, {"normal": 1.0})]
    for kind in ("aggressive", "conservative"):
        for f in fracs:
            out.append((kind, f, {"normal": 1.0 - f, kind: f}))
    return out


def run_all(quick=False, seeds=(1, 2)):
    run_rows, driver_rows, keep = [], [], {}
    for kind, frac, shares in conditions(quick):
        for seed in (seeds[:1] if quick else seeds):
            df = simulate(shares, seed)
            m = traffic_metrics(df, ROAD_M, warmup_s=WARMUP_S)
            live = df[df["t"] >= WARMUP_S]
            sf = shockwave_factor(live, v_ref=33.33)
            per = live.groupby("vehicle_id").agg(type=("type", "first"), ai=("ai", "mean")).join(sf)
            per = per[per["n_steps"] >= 20]
            by_type = per.groupby("type")["shockwave_factor"].mean().to_dict()
            run_rows.append({"condition": kind, "share": frac, "seed": seed, **m,
                             **{f"sf_{k}": v for k, v in by_type.items()}})
            per = per.assign(condition=kind, share=frac, seed=seed)
            driver_rows.append(per.reset_index())
            if seed == seeds[0] and frac in (0.0, 0.2):
                keep[kind] = df
            print(f"{kind:12s} {frac:4.0%} seed {seed}: speed {m['mean_speed']:5.1f} m/s  "
                  f"std {m['speed_std']:4.1f}  flow {m['throughput_veh_h']:6.0f} veh/h  "
                  f"stop-go {m['stop_and_go_vehicles']:3d}  "
                  + "  ".join(f"SF[{k}] {v:.3f}" for k, v in sorted(by_type.items())))
    return pd.DataFrame(run_rows), pd.concat(driver_rows, ignore_index=True), keep


def plots(runs, drivers, keep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"normal": "#5b8def", "aggressive": "#d1495b", "conservative": "#2a9d8f"}

    # 1. driver profiles: AI against shockwave factor
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = drivers[drivers["share"].isin([0.0, 0.2])]
    for typ, g in sub.groupby("type"):
        ax.scatter(g["ai"], g["shockwave_factor"], s=10, alpha=0.5, color=colors[typ], label=typ)
    ax.set_xlabel("Aggressiveness Index (mean over the run)")
    ax.set_ylabel("shockwave factor")
    ax.set_title("driver profiles: own aggressiveness against disturbance caused upstream", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shockwave_driver_profiles.png"), dpi=140)

    # 2. traffic metrics against share
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    base = runs[runs["condition"] == "baseline"]
    for kind in ("aggressive", "conservative"):
        g = pd.concat([base, runs[runs["condition"] == kind]]).groupby("share").mean(numeric_only=True)
        for ax, col, lab in zip(axs, ["mean_speed", "throughput_veh_h", "stop_and_go_vehicles"],
                                ["mean speed (m/s)", "throughput (veh/h)", "vehicles in stop-and-go"]):
            ax.plot(g.index * 100, g[col], "o-", color=colors[kind], label=kind)
            ax.set_xlabel("share of drivers (%)")
            ax.set_title(lab, fontsize=10)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shockwave_traffic_metrics.png"), dpi=140)

    # 3. time-space diagrams, right lane
    fig, axs = plt.subplots(1, len(keep), figsize=(4.2 * len(keep), 3.8), sharey=True)
    for ax, (kind, df) in zip(np.atleast_1d(axs), keep.items()):
        d = df[(df["lane"] == 0) & (df["t"] >= WARMUP_S)]
        sc = ax.scatter(d["t"], d["pos"], c=d["speed"], s=0.3, cmap="RdYlGn", vmin=0, vmax=36)
        ax.axhline(ROAD_M - BOTTLENECK_M, color="k", lw=0.6, ls="--")
        ax.set_title("baseline" if kind == "baseline" else f"20% {kind}", fontsize=10)
        ax.set_xlabel("time (s)")
    np.atleast_1d(axs)[0].set_ylabel("position along road (m)")
    fig.colorbar(sc, ax=axs, label="speed (m/s)", fraction=0.02)
    fig.savefig(os.path.join(FIG_DIR, "shockwave_timespace.png"), dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--demand", type=float, default=DEMAND_VEH_H, help="vehicles per hour entering the road")
    a = ap.parse_args()
    DEMAND_VEH_H = a.demand
    runs, drivers, keep = run_all(a.quick)
    runs.to_csv(os.path.join(DATA_DIR, "shockwave_runs.csv"), index=False)
    drivers.to_csv(os.path.join(DATA_DIR, "shockwave_drivers.csv"), index=False)
    plots(runs, drivers, keep)
    print("summary: mean shockwave factor by driver type across all runs")
    print(drivers.groupby("type")["shockwave_factor"].describe()[["count", "mean", "50%", "75%"]].round(3))
