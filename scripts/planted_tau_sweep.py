"""
Calibration of the planted driver types: a common multiplier on IDM tau (ratios between the types kept)
and the highway demand, two seeds each, 600 s runs; KS distance of speed and time headway (rows before
the lane drop) against NGSIM US-101 (5 min extract). Result: data/planted_tau_sweep.csv.

    python scripts/planted_tau_sweep.py
(the multipliers are applied to the tau values in scripts/planted_drivers.py at the time of the run)
"""
import copy
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from multiprocessing import Pool
import numpy as np, pandas as pd
SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sumo_planted", "tau_sweep")

def ref():
    from datasets.ngsim import read_raw, trajectories
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = trajectories(read_raw(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "28:9:2026", "ngsim_us-101_5min.csv"), "us-101"))
    tr = tr[np.isclose(tr.t % 1.0, 0, atol=0.05)]
    m = (tr.gap > 0) & (tr.speed > 1)
    return tr.speed.to_numpy() * 3.6, (tr.gap / tr.speed)[m].to_numpy()

def job(arg):
    f, vph, seed = arg
    import planted_drivers as P
    P.SIM_END, P.WARMUP = 600.0, 120.0
    P.OUT_DIR = P.NET_DIR = f"{SCR}/f{f}_d{vph}_s{seed}"
    os.makedirs(P.OUT_DIR, exist_ok=True)
    P.TYPES = copy.deepcopy(P.TYPES)
    for t in P.TYPES.values():
        t["tau"] = round(t["tau"] * f, 3)
    P.DENSITY = {"medium": vph}
    nets = {"highway": P.networks()["highway"]}
    base = __import__("planted_drivers").TYPES["normal"]["tau"] / f
    assert base > 0                                                     # fresh process per job (maxtasksperchild=1)
    P.run(("highway", "medium", seed, nets))
    ps = pd.read_csv(os.path.join(P.OUT_DIR, f"highway_medium_s{seed}.per_second.csv.gz"))
    ps = ps[ps.edge == "hw1"]                      # upstream of the drop, where the queue sits
    m = (ps.gap_m > 0) & (ps.speed_kmh > 3.6)
    return f, vph, seed, ps.speed_kmh.to_numpy(), (ps.gap_m / (ps.speed_kmh / 3.6))[m].to_numpy()

if __name__ == "__main__":
    from planted_eval import ks
    rs, rt = ref()
    args = [(f, v, s) for f in (0.5, 0.6, 0.7, 0.85, 1.0) for v in (3600, 4500) for s in (1, 2)]
    with Pool(8, maxtasksperchild=1) as p:
        res = p.map(job, args)
    agg = {}
    for f, v, s, sp, th in res:
        a = agg.setdefault((f, v), ([], []))
        a[0].append(sp); a[1].append(th)
    rows = []
    for (f, v), (sps, ths) in agg.items():
        sp, th = np.concatenate(sps), np.concatenate(ths)
        rows.append({"tau_factor": f, "demand": v, "speed_med": np.median(sp), "ks_speed": ks(sp, rs)[0],
                     "thw_med": np.median(th), "thw_p10": np.percentile(th, 10), "ks_thw": ks(th, rt)[0]})
    r = pd.DataFrame(rows); r["ks_sum"] = r.ks_speed + r.ks_thw
    print("NGSIM US-101: speed median %.1f, thw median %.2f, p10 %.2f" % (np.median(rs), np.median(rt), np.percentile(rt, 10)))
    print(r.sort_values("ks_sum").round(3).to_string(index=False))
    r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "planted_tau_sweep.csv"), index=False)
