"""
One cleaning step for every per second table (scripts/audit_datasets.py found the problems; numbers there).

Input columns: t, speed_kmh, accel, gap_m, wave_m (+ lane_valid where the source has it). Output adds `leader`
(1 car ahead, 0 known no car ahead, NaN unknown) and replaces unknown values by NaN instead of 0:

  rule 1  accel above 8 m/s2 or at the +-9 clip: row dropped (not a car's acceleration)
  rule 2  wave unknown: source has no lanes (pNEUMA, inD, rounD, uniD), lane not detected (lane_valid False),
          or above 2 m (outside a lane: lane assignment error, SUMO invalid code 2^30)
  rule 3  gap impossible while moving above 20 km/h: below 2 m or below 0.3 s headway -> gap unknown
  rule 4  gap 0 means "no car ahead" only where the source sees every car (drones, SUMO, NGSIM, DriveDNA radar);
          for UAH (phone camera) gap 0 means "no car detected": gap and leader unknown

Unknown values are left out of a window's mean and out of the score (model/aggressiveness_model.py
reference_score rescales the weights of the known terms), so missing data no longer reads as calm driving.
"""
import numpy as np
import pandas as pd

NO_LANES = {"pneuma", "inD", "rounD", "uniD"}
CAMERA_GAP = {"uah"}           # gap 0 = not detected, not "no car ahead"
ACCEL_MAX = 8.0
WAVE_MAX = 2.0
GAP_MIN_M, THW_MIN_S, MOVING_KMH = 2.0, 0.3, 20.0


def clean(ps, source):
    """(cleaned table, log dict with the number of rows each rule touched)."""
    d = ps.copy()
    log = {"source": source, "rows_in": len(d)}
    bad_a = (d["accel"].abs() > ACCEL_MAX) | np.isclose(d["accel"].abs(), 9.0)
    log["rule1_accel_dropped"] = int(bad_a.sum())
    d = d[~bad_a].copy()

    wave = d["wave_m"].astype(float)
    if source in NO_LANES:
        unknown_w = pd.Series(True, index=d.index)
    else:
        unknown_w = wave.isna() | (wave > WAVE_MAX)
        if "lane_valid" in d:
            unknown_w |= ~d["lane_valid"].astype(bool)
    log["rule2_wave_unknown"] = int(unknown_w.sum())
    d["wave_m"] = wave.mask(unknown_w)

    gap = d["gap_m"].astype(float)
    v = d["speed_kmh"] / 3.6
    moving = d["speed_kmh"] > MOVING_KMH
    impossible = (gap > 0) & moving & ((gap < GAP_MIN_M) | (gap / v.clip(lower=0.5) < THW_MIN_S))
    log["rule3_gap_impossible"] = int(impossible.sum())
    leader = pd.Series(np.where(gap > 0, 1.0, 0.0), index=d.index)
    if source in CAMERA_GAP:
        not_seen = gap <= 0
        log["rule4_gap_not_detected"] = int(not_seen.sum())
        leader = leader.mask(not_seen)
        gap = gap.mask(not_seen)
    else:
        log["rule4_gap_not_detected"] = 0
    gap = gap.mask(impossible)
    leader = leader.mask(impossible)
    d["gap_m"] = gap
    d["leader"] = leader
    log["rows_out"] = len(d)
    return d, log
