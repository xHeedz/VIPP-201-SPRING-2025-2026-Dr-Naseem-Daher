"""
10 s windows (step 5 s) from a cleaned per second table (data/clean/*.csv.gz), the same way for every source.

Seconds below 5 km/h are dropped first (as before). Per window: mean of each reference term over its seconds
(model/aggressiveness_model.py reference_terms), a term is unknown (NaN) when more than half of its seconds are
unknown; plus mean speed, share of seconds with a car seen ahead, number of seconds, label, group,
environment, scenario. A window needs at least 8 seconds.
"""
import numpy as np
import pandas as pd

from model.aggressiveness_model import REF_TERMS, reference_terms

TERM_COLS = [f"t_{t}" for t in REF_TERMS]


def per_second_terms(d):
    """cleaned table -> same rows with the four term columns added (stopped seconds dropped)."""
    d = d[d["speed_kmh"] >= 5.0].copy()
    T = reference_terms(d["speed_kmh"], d["accel"], d["gap_m"], d["wave_m"], d["environment"].to_numpy())
    for i, c in enumerate(TERM_COLS):
        d[c] = T[:, i]
    return d


def windows(d, length_s=10, step_s=5, min_seconds=8):
    d = per_second_terms(d)
    d = d.sort_values(["vid", "t"])
    t0 = d.groupby("vid")["t"].transform("min")
    block = np.floor((d["t"] - t0) / step_s).astype(int)
    d = d.assign(block=block, has_leader=d["leader"])
    agg = {c: ["sum", "count"] for c in TERM_COLS + ["has_leader"]}
    agg["speed_kmh"] = ["sum", "count"]
    b = d.groupby(["vid", "block"]).agg(agg)
    b.columns = [f"{a}_{s}" for a, s in b.columns]
    b = b.reset_index()
    k = length_s // step_s                                  # blocks per window (2)
    parts = []
    for j in range(k):
        x = b.copy()
        x["start"] = x["block"] - j
        parts.append(x)
    w = pd.concat(parts).groupby(["vid", "start"]).sum(numeric_only=True)
    n = w["speed_kmh_count"]
    out = pd.DataFrame({"seconds": n, "speed_kmh": w["speed_kmh_sum"] / n,
                        "leader_share": w["has_leader_sum"] / n})      # seconds with a car seen ahead (unknown = not seen)
    for c in TERM_COLS:
        m = w[f"{c}_sum"] / w[f"{c}_count"].replace(0, np.nan)
        out[c] = m.where(w[f"{c}_count"] >= n / 2)
    out = out[out["seconds"] >= min_seconds].reset_index()
    meta = d.groupby("vid")[["source", "group", "label", "environment", "scenario"]].first()
    out = out.join(meta, on="vid")
    tmin = d.groupby("vid")["t"].min()
    out["t0"] = out["vid"].map(tmin) + out["start"] * step_s
    return out.drop(columns="start")
