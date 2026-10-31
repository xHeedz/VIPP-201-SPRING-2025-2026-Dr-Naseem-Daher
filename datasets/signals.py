"""
Extra per second signals for the extended formula and the agents' state, computed the same way for every cleaned
source (data/clean/*.csv.gz, with speed_limit_kmh and traffic_kmh from scripts/build_clean_tables.py).

  closing speed  -(gap(t+1) - gap(t-1)) / 2, consecutive seconds with a car ahead; a gap jump above 10 m/s means the
                 car ahead changed: unknown
  TTC            gap / closing speed while closing in (above 0.1 m/s); no car ahead or opening gap: no collision course
  rel_limit      (speed - limit) / limit;  rel_traffic  (speed - traffic speed) / max(traffic speed, 10 km/h)
  IDM residual   actual acceleration - IDM acceleration (Treiber, Hennecke, Helbing 2000) of a calm driver in the
                 same situation: a_max 1.5 m/s2, b 2.0 m/s2, T 1.5 s, s0 2 m, delta 4, desired speed = speed limit
                 (traffic speed, else 130 km/h, when there is no limit); free road term only without a car ahead
Terms between 0 and 1 (NaN = unknown, left out of the score like the other terms):
  t_ttc          (1 - TTC / 4 s)^2 below 4 s, else 0
  t_rel_limit    rel_limit / 0.3 clipped to [0, 1] (30% over the limit = 1, at or under the limit = 0)
  t_rel_traffic  rel_traffic / 0.3 clipped to [0, 1]
  t_idm          |IDM residual| / 3 m/s2 clipped to [0, 1]
"""
import numpy as np
import pandas as pd

EXT_COLS = ["t_ttc", "t_rel_limit", "t_rel_traffic", "t_idm"]
IDM = dict(a=1.5, b=2.0, T=1.5, s0=2.0, delta=4.0)
TTC_MAX_S, REL_MAX, IDM_MAX = 4.0, 0.3, 3.0


def idm_accel(v, gap, dv, v0):
    """IDM acceleration (m/s2); v, dv (closing speed), v0 in m/s, gap in m (NaN or 0 = free road)"""
    free = IDM["a"] * (1.0 - (v / np.maximum(v0, 1.0)) ** IDM["delta"])
    with np.errstate(invalid="ignore", divide="ignore"):
        s_star = IDM["s0"] + np.maximum(0.0, v * IDM["T"] + v * dv / (2.0 * np.sqrt(IDM["a"] * IDM["b"])))
        inter = IDM["a"] * (s_star / gap) ** 2
    follow = np.isfinite(gap) & (gap > 0) & np.isfinite(dv)
    return np.clip(np.where(follow, free - inter, free), -9.0, IDM["a"])


def add_signals(d):
    """per second table (sorted by vid, t) -> same rows with closing_ms, ttc_s, rel_limit, rel_traffic, idm_res and the
    four extended terms"""
    d = d.sort_values(["vid", "t"]).copy()
    v = d["speed_kmh"].to_numpy(float) / 3.6
    gap = d["gap_m"].to_numpy(float)
    vid = d["vid"].to_numpy()
    t = d["t"].to_numpy(float)
    nxt, prv = np.r_[np.arange(1, len(d)), len(d) - 1], np.r_[0, np.arange(len(d) - 1)]
    ok = (vid[nxt] == vid) & (vid[prv] == vid) & np.isclose(t[nxt] - t[prv], 2.0, atol=0.5)
    ok &= (gap[nxt] > 0) & (gap[prv] > 0) & (gap > 0)
    closing = np.where(ok, -(gap[nxt] - gap[prv]) / np.maximum(t[nxt] - t[prv], 1e-6), np.nan)
    closing = np.where(np.abs(closing) > 10.0, np.nan, closing)
    lead = d["leader"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ttc = np.where(closing > 0.1, gap / closing, np.inf)
    ttc = np.where(lead == 0, np.inf, ttc)                       # known: no car ahead
    ttc = np.where((lead == 1) & np.isnan(closing), np.nan, ttc)  # car ahead, closing speed unknown
    ttc = np.where(np.isnan(lead), np.nan, ttc)                   # car ahead unknown
    lim = d["speed_limit_kmh"].to_numpy(float) if "speed_limit_kmh" in d else np.full(len(d), np.nan)
    trf = d["traffic_kmh"].to_numpy(float) if "traffic_kmh" in d else np.full(len(d), np.nan)
    sp = d["speed_kmh"].to_numpy(float)
    rel_limit = (sp - lim) / lim
    rel_traffic = (sp - trf) / np.maximum(trf, 10.0)
    v0 = np.where(np.isfinite(lim), lim, np.where(np.isfinite(trf), trf, 130.0)) / 3.6
    a_idm = idm_accel(v, np.where(lead == 1, gap, np.nan), np.nan_to_num(closing, nan=0.0), v0)
    a_idm = np.where((lead == 1) & np.isnan(closing), np.nan, a_idm)
    a_idm = np.where(np.isnan(lead), np.nan, a_idm)
    res = d["accel"].to_numpy(float) - a_idm
    d["closing_ms"], d["ttc_s"], d["rel_limit"], d["rel_traffic"], d["idm_res"] = closing, ttc, rel_limit, rel_traffic, res
    with np.errstate(invalid="ignore"):
        d["t_ttc"] = np.where(np.isnan(ttc), np.nan, np.where(ttc < TTC_MAX_S, (1.0 - ttc / TTC_MAX_S) ** 2, 0.0))
        d["t_rel_limit"] = np.clip(rel_limit / REL_MAX, 0.0, 1.0)
        d["t_rel_traffic"] = np.clip(rel_traffic / REL_MAX, 0.0, 1.0)
        d["t_idm"] = np.clip(np.abs(res) / IDM_MAX, 0.0, 1.0)
    for c in ("t_rel_limit", "t_rel_traffic", "t_idm"):
        src = {"t_rel_limit": rel_limit, "t_rel_traffic": rel_traffic, "t_idm": res}[c]
        d[c] = d[c].where(np.isfinite(src))
    return d
