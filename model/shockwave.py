"""
Shockwave factor: how much a driver disturbs the traffic behind it.

The Aggressiveness Index (AI) describes a driver's own motion. It cannot see
that an overly cautious driver, who scores low on AI, can still trigger waves
of braking and slowdowns upstream. The shockwave factor (SF) measures that
effect for every driver, aggressive or conservative.

For every vehicle i at every time step:

  condition(v) = 0.5 * min(1, braking_v / A_REF) + 0.5 * speed_deficit_v
      braking_v       = max(0, -acceleration)             in m/s^2
      speed_deficit_v = max(0, v_ref - speed) / v_ref      how far below free-flow speed

  upstream(i)   = distance-weighted mean condition of up to 3 followers
                  within HORIZON metres behind i in the same lane
  downstream(i) = condition of i's own leader within HORIZON metres (0 if none)

  disruption(i) = max(0, upstream(i) - downstream(i))

The subtraction is the attribution step. A car stuck in a queue has followers
that are slow and braking, but so is its own leader, so it passes the
disturbance along rather than creating it and scores near zero. A slow car on
an open road, or a car that brakes harder than its leader did, creates the
disturbance and scores high.

SF_i is the mean disruption over the time steps where i had followers, in
[0, 1]. Input is a long table with columns t, vehicle_id, lane, pos, speed,
accel (SI units), which both SUMO telemetry and NGSIM trajectories provide.
"""
import numpy as np
import pandas as pd

A_REF = 5.0
HORIZON_M = 100.0
N_FOLLOWERS = 3


def _condition(speed, accel, v_ref):
    braking = np.clip(-accel, 0.0, None) / A_REF
    deficit = np.clip((v_ref - speed) / v_ref, 0.0, 1.0)
    return 0.5 * np.minimum(braking, 1.0) + 0.5 * deficit


def disruption_table(df, v_ref=None, horizon_m=HORIZON_M, n_followers=N_FOLLOWERS):
    """Adds columns condition, upstream, downstream, disruption to a copy of df."""
    d = df.sort_values(["t", "lane", "pos"]).reset_index(drop=True)
    if v_ref is None:
        v_ref = float(np.nanpercentile(d["speed"], 85))
    v_ref = max(v_ref, 1.0)
    d["condition"] = _condition(d["speed"].to_numpy(), d["accel"].to_numpy(), v_ref)

    t, lane, pos, cond = d["t"].to_numpy(), d["lane"].to_numpy(), d["pos"].to_numpy(), d["condition"].to_numpy()
    n = len(d)
    num = np.zeros(n)
    den = np.zeros(n)
    for k in range(1, n_followers + 1):          # rows k places earlier = k-th car behind
        idx = np.arange(n) - k
        ok = idx >= 0
        j = np.where(ok, idx, 0)
        gap = pos - pos[j]
        ok &= (t[j] == t) & (lane[j] == lane) & (gap > 0) & (gap <= horizon_m)
        w = np.where(ok, 1.0 - gap / horizon_m, 0.0)
        num += w * cond[j]
        den += w
    upstream = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    j = np.minimum(np.arange(n) + 1, n - 1)       # next row = car ahead
    gap = pos[j] - pos
    has_leader = (np.arange(n) + 1 < n) & (t[j] == t) & (lane[j] == lane) & (gap > 0) & (gap <= horizon_m)
    downstream = np.where(has_leader, cond[j], 0.0)

    d["upstream"] = upstream
    d["downstream"] = downstream
    d["disruption"] = np.where(np.isnan(upstream), np.nan, np.clip(upstream - downstream, 0.0, None))
    return d


def shockwave_factor(df, **kw):
    """Per-vehicle SF: DataFrame indexed by vehicle_id with shockwave_factor and n_steps."""
    d = disruption_table(df, **kw)
    g = d.groupby("vehicle_id")["disruption"]
    return pd.DataFrame({"shockwave_factor": g.mean(), "n_steps": g.count()})


def traffic_metrics(df, road_length_m, warmup_s=0.0, detector_spacing_m=200.0, dt=None):
    """Network-level indicators: mean and spread of speed, throughput, stop-and-go
    events, and the speed at which slowdowns travel upstream (negative = upstream)."""
    d = df[df["t"] >= warmup_s]
    dur_h = max((d["t"].max() - d["t"].min()) / 3600.0, 1e-9)
    exit_x = road_length_m - 100.0
    passed = d[d["pos"] >= exit_x].groupby("vehicle_id").size().shape[0]

    # stop-and-go: vehicle drops below 5 m/s after having been above 15 m/s
    sg = 0
    for _, v in d.sort_values("t").groupby("vehicle_id")["speed"]:
        s = v.to_numpy()
        fast = np.maximum.accumulate(s > 15.0)
        sg += int(np.any(fast & (s < 5.0)))

    wave_speed = _wave_speed(d, road_length_m, detector_spacing_m)
    return {
        "mean_speed": float(d["speed"].mean()),
        "speed_std": float(d["speed"].std()),
        "throughput_veh_h": passed / dur_h,
        "stop_and_go_vehicles": sg,
        "wave_speed_kmh": wave_speed,
    }


def _wave_speed(d, road_length_m, spacing):
    """Median lag of speed drops between neighbouring virtual detectors."""
    t_bins = np.arange(d["t"].min(), d["t"].max() + 1.0, 1.0)
    xs = np.arange(spacing, road_length_m - spacing, spacing)
    series = []
    for x in xs:
        near = d[(d["pos"] >= x - 15) & (d["pos"] < x + 15)]
        s = near.groupby(np.digitize(near["t"], t_bins))["speed"].mean()
        full = pd.Series(np.nan, index=range(1, len(t_bins) + 1))
        full.loc[s.index] = s.values
        series.append(full.interpolate(limit_direction="both").to_numpy())
    speeds = []
    for a, b in zip(series[1:], series[:-1]):        # a is downstream of b
        if np.nanstd(a) < 0.5 or np.nanstd(b) < 0.5:
            continue
        a0, b0 = a - np.nanmean(a), b - np.nanmean(b)
        # only lags that correspond to plausible upstream wave speeds (5 to 60 km/h)
        lags = range(max(1, int(spacing / (60 / 3.6))), int(spacing / (5 / 3.6)) + 1)
        lags = [L for L in lags if L < len(a) - 10]
        if not lags:
            continue
        cc = [np.nansum(a0[:-L] * b0[L:]) for L in lags]   # b follows a after L seconds
        L = lags[int(np.argmax(cc))]
        if max(cc) > 0:
            speeds.append(-spacing / L * 3.6)
    return float(np.median(speeds)) if speeds else float("nan")


def driver_profile(ai, sf, sf_threshold):
    """AI category plus a 'disruptive' flag when SF exceeds the threshold."""
    cat = "conservative" if ai < 35 else ("normal" if ai < 70 else "aggressive")
    return f"{cat}, disruptive" if sf >= sf_threshold else cat
