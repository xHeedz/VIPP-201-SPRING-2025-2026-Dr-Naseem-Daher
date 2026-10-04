"""
pNEUMA loader (drone swarm over downtown Athens, 25 Hz; Barmpounakis and Geroliminis 2020,
data source: pNEUMA, open-traffic.epfl.ch, DOI 10.5281/zenodo.10491409, CC BY-NC 4.0).

File format (semicolon separated, one row per vehicle):
    track_id; type; traveled_d; avg_speed; then repeated (lat; lon; speed km/h; lon_acc; lat_acc; time s)

Returns the per second table of datasets/uah.py (t, speed_kmh, accel, gap_m, wave_m) plus vehicle_id,
type and local density. How each input is measured here:
    speed   pNEUMA speed sampled once per second, 3 s mean (UAH way); accel = its derivative
    gap     no lanes or leaders in the data: nearest vehicle ahead whose centre is within +-1.6 m
            laterally of the heading line and whose heading is within 30 deg, centre distance minus
            half of both lengths (length by vehicle type); none within 100 m = 0 (no car ahead)
    wave    NOT measurable without a lane map: set to 0, so pNEUMA scores use three terms
    density vehicles (any type) within 50 m of the vehicle at that second
Motorcycles are kept as possible leaders but not scored (they filter between lanes, so a lane cone
gap is meaningless for them).
"""
import numpy as np
import pandas as pd

LENGTH_M = {"Car": 4.5, "Taxi": 4.5, "Medium Vehicle": 6.5, "Heavy Vehicle": 10.0, "Bus": 12.0, "Motorcycle": 2.0}
SCORED = ("Car", "Taxi", "Medium Vehicle", "Heavy Vehicle", "Bus")
EARTH_R = 6371000.0


def read_raw(path):
    """Long table at 25 Hz: vehicle_id, type, t, x, y (m, local), speed_kmh, lon_acc, lat_acc."""
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            p = [x.strip() for x in line.strip().strip(";").split(";")]
            if len(p) < 10:
                continue
            vid, typ = int(p[0]), p[1]
            a = np.array(p[4:4 + (len(p) - 4) // 6 * 6], dtype=float).reshape(-1, 6)
            rows.append(pd.DataFrame({"vehicle_id": vid, "type": typ, "lat": a[:, 0], "lon": a[:, 1],
                                      "speed_kmh": a[:, 2], "lon_acc": a[:, 3], "lat_acc": a[:, 4], "t": a[:, 5]}))
    d = pd.concat(rows, ignore_index=True)
    lat0, lon0 = d["lat"].mean(), d["lon"].mean()
    d["x"] = np.deg2rad(d["lon"] - lon0) * EARTH_R * np.cos(np.deg2rad(lat0))
    d["y"] = np.deg2rad(d["lat"] - lat0) * EARTH_R
    return d


def per_second(raw, cone_lat_m=1.6, heading_tol_deg=30.0, lookahead_m=100.0, density_r_m=50.0):
    """Per second table for every vehicle (see module docstring)."""
    d = raw[np.isclose(raw["t"] % 1.0, 0.0, atol=0.021) | np.isclose(raw["t"] % 1.0, 1.0, atol=0.021)].copy()
    d["t"] = np.round(d["t"])
    d = d.drop_duplicates(["vehicle_id", "t"]).sort_values(["vehicle_id", "t"])
    g = d.groupby("vehicle_id", sort=False)
    d["v_s"] = g["speed_kmh"].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()) / 3.6

    # per vehicle derivative by position (groupby.apply returns a DataFrame when all groups have equal length)
    acc = np.zeros(len(d))
    tt_all, vv_all = d["t"].to_numpy(), d["v_s"].to_numpy()
    for ix in d.groupby("vehicle_id", sort=False).indices.values():
        if len(ix) > 2:
            acc[ix] = np.gradient(vv_all[ix], tt_all[ix])
    d["accel"] = np.clip(acc, -9.0, 9.0)
    # heading from the displacement over +-1 s (kept from the last move when nearly stopped)
    dx = g["x"].shift(-1) - g["x"].shift(1)
    dy = g["y"].shift(-1) - g["y"].shift(1)
    head = np.arctan2(dy, dx).where(np.hypot(dx, dy) > 0.5)
    d["heading"] = head.groupby(d["vehicle_id"]).transform(lambda s: s.ffill().bfill())
    d["length"] = d["type"].map(LENGTH_M).fillna(4.5)

    gap = np.zeros(len(d))
    dens = np.zeros(len(d))
    idx = np.arange(len(d))
    for _, rows in d.groupby("t").indices.items():
        sub = d.iloc[rows]
        x, y, h, L = (sub[c].to_numpy() for c in ("x", "y", "heading", "length"))
        ok = np.isfinite(h)
        DX, DY = x[None, :] - x[:, None], y[None, :] - y[:, None]            # j relative to i
        lon = DX * np.cos(h)[:, None] + DY * np.sin(h)[:, None]
        lat = -DX * np.sin(h)[:, None] + DY * np.cos(h)[:, None]
        dh = np.abs((h[None, :] - h[:, None] + np.pi) % (2 * np.pi) - np.pi)
        cand = (lon > 0) & (lon <= lookahead_m) & (np.abs(lat) <= cone_lat_m) & (dh <= np.deg2rad(heading_tol_deg))
        cand &= ok[:, None] & ok[None, :]
        lon_c = np.where(cand, lon, np.inf)
        j = lon_c.argmin(axis=1)
        has = np.isfinite(lon_c[np.arange(len(sub)), j])
        bumper = lon_c[np.arange(len(sub)), j] - (L + L[j]) / 2
        gap[idx[rows]] = np.where(has, np.clip(bumper, 0.1, None), 0.0)
        dens[idx[rows]] = (np.hypot(DX, DY) <= density_r_m).sum(axis=1) - 1
    d["gap_m"] = gap
    d["density_50m"] = dens
    out = pd.DataFrame({"t": d["t"], "vehicle_id": d["vehicle_id"], "type": d["type"], "speed_kmh": d["v_s"] * 3.6,
                        "accel": d["accel"], "gap_m": d["gap_m"], "wave_m": 0.0, "density_50m": d["density_50m"],
                        "label": "unlabelled"})
    return out.reset_index(drop=True)
