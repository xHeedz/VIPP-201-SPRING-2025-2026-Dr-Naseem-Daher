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


RAW_COLS = ["r_speed", "r_accel", "r_gap_m", "r_headway", "r_lane", "r_ttc", "r_rel_limit", "r_rel_traffic", "r_idm"]


def raw_inputs(d):
    """unsquared, roughly 0 to 1 (signed where the sign means something) inputs for models that learn each term's shape
    (model/kan.py); NaN stays unknown"""
    v = d["speed_kmh"] / 3.6
    gap = d["gap_m"]
    with np.errstate(invalid="ignore", divide="ignore"):
        thw = np.where(gap > 0, gap / np.maximum(v, 0.5), np.inf)
    d["r_speed"] = d["speed_kmh"] / 150.0
    d["r_accel"] = (d["accel"].abs() / 5.0).clip(upper=1.0)
    d["r_gap_m"] = np.where(gap.isna(), np.nan, np.clip(1 - gap.fillna(1e9) / 50.0, 0, 1))
    d["r_headway"] = np.where(gap.isna(), np.nan, np.clip(1 - thw / 3.0, 0, 1))
    d["r_lane"] = (d["wave_m"].abs() / 1.5).clip(upper=1.0)
    d["r_ttc"] = np.where(np.isnan(d["ttc_s"]), np.nan, np.clip(1 - d["ttc_s"] / 4.0, 0, 1))
    d["r_rel_limit"] = d["rel_limit"].clip(-0.5, 0.5)
    d["r_rel_traffic"] = d["rel_traffic"].clip(-0.5, 0.5)
    d["r_idm"] = (d["idm_res"] / 3.0).clip(-1, 1)
    return d


def windows(d, length_s=10, step_s=5, min_seconds=8, extended=False, raw=False):
    """extended: also the terms of datasets/signals.py (TTC, speed vs limit, speed vs traffic, IDM residual);
    raw: also the unsquared inputs of raw_inputs"""
    d = per_second_terms(d)
    cols = list(TERM_COLS)
    if extended or raw:
        from datasets.signals import EXT_COLS, add_signals
        d = add_signals(d)
        if extended:
            cols += EXT_COLS
        if raw:
            d = raw_inputs(d)
            cols += RAW_COLS
    d = d.sort_values(["vid", "t"])
    t0 = d.groupby("vid")["t"].transform("min")
    block = np.floor((d["t"] - t0) / step_s).astype(int)
    d = d.assign(block=block, has_leader=d["leader"])
    agg = {c: ["sum", "count"] for c in cols + ["has_leader"]}
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
    for c in cols:
        m = w[f"{c}_sum"] / w[f"{c}_count"].replace(0, np.nan)
        out[c] = m.where(w[f"{c}_count"] >= n / 2)
    out = out[out["seconds"] >= min_seconds].reset_index()
    meta = d.groupby("vid")[["source", "group", "label", "environment", "scenario"]].first()
    out = out.join(meta, on="vid")
    tmin = d.groupby("vid")["t"].min()
    out["t0"] = out["vid"].map(tmin) + out["start"] * step_s
    return out.drop(columns="start")
