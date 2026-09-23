"""
Run SumoEnv and write per-vehicle telemetry to CSV. The CSV feeds
scripts/train_gate.py and the ROS2 TelemetryPublisherNode replay.

RECREATED BY CLAUDE (Anthropic), September 2026. See ../README.md.

    python scripts/record_telemetry.py --scenario highway --seconds 60
"""
import argparse
import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from env.sumo_env import SumoEnv  # noqa: E402

COLUMNS = ["t", "scenario", "vehicle_id", "vtype", "speed", "accel", "prox", "wave", "jerk", "slip",
           "density", "friction", "road_type", "crashed"]


def record(scenario, seconds, out, seed=0, aggressive_fraction=0.05):
    env = SumoEnv(scenario, seed=seed, aggressive_fraction=aggressive_fraction,
                  max_steps=int(seconds * 15) + 10)
    env.reset()
    t0 = env.conn.simulation.getTime()
    rows = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for _ in range(int(seconds * 15)):
            _, _, term, trunc, info = env.step(1)
            t = env.conn.simulation.getTime() - t0
            colliding = set(env.conn.simulation.getCollidingVehiclesIDList())
            for vid, tel in info["telemetry"].items():
                w.writerow([f"{t:.4f}", scenario, vid, tel["type"], f"{tel['speed']:.4f}", f"{tel['accel']:.4f}",
                            f"{tel['prox']:.3f}", f"{tel['wave']:.4f}", f"{tel['jerk']:.4f}", f"{tel['slip']:.4f}",
                            f"{tel['density']:.3f}", env.friction, env.road_type, int(vid in colliding)])
                rows += 1
            if trunc:
                break
            if term:   # ego left the road; keep recording the rest of the traffic
                pass
    env.close()
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="highway", choices=SumoEnv.SCENARIOS)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", f"telemetry_{a.scenario}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = record(a.scenario, a.seconds, out, a.seed)
    print(f"wrote {n} rows to {out}")
