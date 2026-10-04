"""
Demand sweep for the us101 scenario (scripts/planted_drivers.py): main road and on ramp demand, two seeds,
600 s; KS distance of speed and time headway (main road before the merge and the weaving section)
against NGSIM US-101 (5 min extract). Result: data/us101_sweep.csv.

    python scripts/us101_sweep.py
"""
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
SCR = os.path.join(HERE, "..", "data", "sumo_planted", "us101_sweep")
NGSIM = os.path.join(HERE, "..", "..", "28:9:2026", "ngsim_us-101_5min.csv")
MAINS = [int(x) for x in os.environ.get("US101_MAINS", "3000,4000,5000,6000").split(",")]
RAMPS = [int(x) for x in os.environ.get("US101_RAMPS", "500,1000").split(",")]


def ref():
    from datasets.ngsim import read_raw, trajectories
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = trajectories(read_raw(NGSIM, "us-101"))
    tr = tr[np.isclose(tr.t % 1.0, 0, atol=0.05)]
    m = (tr.gap > 0) & (tr.speed > 1)
    return tr.speed.to_numpy() * 3.6, (tr.gap / tr.speed)[m].to_numpy()


def job(arg):
    main, ramp, seed = arg
    import planted_drivers as P
    P.SIM_END, P.WARMUP = 600.0, 120.0
    P.OUT_DIR = P.NET_DIR = os.path.join(SCR, f"m{main}_r{ramp}_s{seed}")
    os.makedirs(P.OUT_DIR, exist_ok=True)
    P.US101_DEMAND = (main, ramp)
    P.run(("us101", "medium", seed, {"us101": P.networks()["us101"]}))
    ps = pd.read_csv(os.path.join(P.OUT_DIR, f"us101_medium_s{seed}.per_second.csv.gz"))
    ps = ps[ps["edge"].isin(["main1", "aux"])]
    m = (ps.gap_m > 0) & (ps.speed_kmh > 3.6)
    return main, ramp, seed, ps.speed_kmh.to_numpy(), (ps.gap_m / (ps.speed_kmh / 3.6))[m].to_numpy()


if __name__ == "__main__":
    from planted_eval import ks
    rs, rt = ref()
    args = [(m, r, s) for m in MAINS for r in RAMPS for s in (1, 2)]
    with Pool(8, maxtasksperchild=1) as p:
        res = p.map(job, args)
    agg = {}
    for m, r, s, sp, th in res:
        a = agg.setdefault((m, r), ([], []))
        a[0].append(sp)
        a[1].append(th)
    rows = []
    for (m, r), (sps, ths) in agg.items():
        sp, th = np.concatenate(sps), np.concatenate(ths)
        rows.append({"main": m, "ramp": r, "speed_med": np.median(sp), "speed_p25": np.percentile(sp, 25),
                     "speed_p75": np.percentile(sp, 75), "ks_speed": ks(sp, rs)[0], "thw_med": np.median(th), "ks_thw": ks(th, rt)[0]})
    out = pd.DataFrame(rows)
    out["ks_sum"] = out.ks_speed + out.ks_thw
    print("NGSIM US-101: speed median %.1f (p25 %.1f, p75 %.1f), headway median %.2f" %
          (np.median(rs), np.percentile(rs, 25), np.percentile(rs, 75), np.median(rt)))
    print(out.sort_values("ks_sum").round(3).to_string(index=False))
    path = os.path.join(HERE, "..", "data", "us101_sweep.csv")
    if os.path.exists(path):
        out = pd.concat([pd.read_csv(path), out]).drop_duplicates(["main", "ramp"], keep="last")
    out.to_csv(path, index=False)
