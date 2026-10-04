"""
Hand check of one NGSIM arterial row (Lankershim) and one pNEUMA second, recomputed from the raw
CSV rows with plain Python (no pandas, no numpy, no pipeline code) and compared with the loaders.

    python scripts/hand_check_more.py      # writes docs/hand_checks/hand_check_arterial_pneuma.md
"""
import bisect
import csv
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "..", "28:9:2026")
FT = 0.3048
W = (0.5, 0.2, 0.8, 0.4)
LANK = os.path.join(DATA, "ngsim_lankershim.csv")
PNEUMA = os.path.join(DATA, "pneuma", "20181024_d1_0830_0900.csv")


def terms(speed_kmh, accel, gap, wave):
    ns = min(speed_kmh / 150.0, 1.0)
    na = min(abs(accel) / 5.0, 1.0)
    npx = 1.0 - gap / 50.0 if 0 < gap <= 50 else 0.0
    nw = min(abs(wave) / 1.5, 1.0)
    phi = (ns * ns, na, npx * npx, nw)
    return phi, min(100.0 * sum(w * x for w, x in zip(W, phi)), 100.0)


def grad(t, v, i):
    if i == 0:
        return (v[1] - v[0]) / (t[1] - t[0])
    if i == len(v) - 1:
        return (v[-1] - v[-2]) / (t[-1] - t[-2])
    hl, hr = t[i] - t[i - 1], t[i + 1] - t[i]
    return (hl * hl * v[i + 1] - hr * hr * v[i - 1] + (hr * hr - hl * hl) * v[i]) / (hl * hr * (hl + hr))


def roll(x, before, after):
    return [sum(x[max(0, i - before):i + after + 1]) / len(x[max(0, i - before):i + after + 1]) for i in range(len(x))]


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ── NGSIM Lankershim ────────────────────────────────────────────────────────
def lankershim(pick=None):
    num = lambda s: float(s.replace(",", ""))  # noqa: E731
    rows = []
    with open(LANK) as f:
        for r in csv.DictReader(f):
            rows.append({"vid": int(num(r["Vehicle_ID"])), "frame": int(num(r["Frame_ID"])), "gt": num(r["Global_Time"]),
                         "x": num(r["Local_X"]) * FT, "y": num(r["Local_Y"]) * FT, "len": num(r["v_length"]),
                         "lane": int(num(r["Lane_ID"])), "pre": int(num(r["Preceding"])), "sh": num(r["Space_Headway"]),
                         "int": num(r["Int_ID"]), "sec": num(r["Section_ID"]), "dir": num(r["Direction"]), "mov": num(r["Movement"])})
    # recording period: Global_Time - 100 ms x Frame_ID, 30 s bins; ids become period x 100000 + id
    starts = sorted({round((r["gt"] - 100 * r["frame"]) / 30000.0) for r in rows})
    for r in rows:
        k = starts.index(round((r["gt"] - 100 * r["frame"]) / 30000.0)) + 1
        r["vid"] = k * 100000 + r["vid"]
        r["pre"] = k * 100000 + r["pre"] if r["pre"] > 0 else 0
    t0 = min(r["gt"] for r in rows)
    cars, length = {}, {}
    for r in rows:
        r["t"] = (r["gt"] - t0) / 1000.0
        cars.setdefault(r["vid"], []).append(r)
        length[(r["vid"], r["frame"])] = r["len"]
    lane_lat = {}
    for c in cars.values():
        c.sort(key=lambda r: r["t"])
        xs, ys = roll([r["x"] for r in c], 5, 4), roll([r["y"] for r in c], 5, 4)
        for r, xs_, ys_ in zip(c, xs, ys):
            r["xs"], r["ys"] = xs_, ys_
            lane_lat.setdefault((r["lane"], r["dir"], r["sec"], math.floor(ys_ / 10.0)), []).append(xs_)
    if pick is None:   # a through-traffic car on a section, at a whole second in the middle of its trajectory, with a leader
        for vid in sorted(cars):
            c = cars[vid]
            mid = c[len(c) // 2: len(c) // 2 + 30]
            cand = [r for r in mid if r["int"] == 0 and r["mov"] == 1 and r["dir"] in (2, 4) and r["pre"] > 0
                    and abs(r["t"] - round(r["t"])) < 0.05 and len(c) > 300]
            if cand:
                pick = (vid, round(cand[0]["t"]))
                break
    vid, tc = pick
    c = cars[vid]
    t = [r["t"] for r in c]
    i = min(range(len(c)), key=lambda k: abs(t[k] - tc))
    xs, ys = [r["xs"] for r in c], [r["ys"] for r in c]
    vx, vy = grad(t, xs, i), grad(t, ys, i)
    speed = [max(0.0, math.hypot(grad(t, xs, k), grad(t, ys, k))) for k in range(len(c))]
    secs = {}
    for tk, sp in zip(t, speed):
        secs.setdefault(math.floor(tk), []).append(sp)
    sec = sorted(secs)
    v1 = [sum(secs[q]) / len(secs[q]) for q in sec]
    v1s = roll(v1, 1, 1)
    si = sec.index(math.floor(t[i]))
    a1 = max(-9.0, min(9.0, (v1s[si + 1] - v1s[si - 1]) / (sec[si + 1] - sec[si - 1])))
    r = c[i]
    lead_len = length.get((r["pre"], r["frame"]), 15.0)
    gap = max(0.1, (r["sh"] - lead_len) * FT) if r["pre"] > 0 and r["sh"] > 0 else 0.0
    centre = median(lane_lat[(r["lane"], r["dir"], r["sec"], math.floor(r["ys"] / 10.0))])
    wave = abs(r["xs"] - centre)
    phi, score = terms(speed[i] * 3.6, a1, gap, wave)
    return {"vid": vid, "t": t[i], "frame": r["frame"], "lane": r["lane"], "dir": r["dir"], "sec": r["sec"],
            "vx": vx, "vy": vy, "speed_kmh": speed[i] * 3.6, "v1_prev": v1s[si - 1], "v1_next": v1s[si + 1], "accel": a1,
            "sh_ft": r["sh"], "lead": r["pre"], "lead_len_ft": lead_len, "gap": gap, "xs": r["xs"], "centre": centre,
            "n_centre": len(lane_lat[(r["lane"], r["dir"], r["sec"], math.floor(r["ys"] / 10.0))]), "wave": wave,
            "phi": phi, "score": score}


def lankershim_pipeline(vid, tc):
    from datasets.ngsim import per_second, read_raw, trajectories
    from model.aggressiveness_model import index_features, original_score
    ps = per_second(trajectories(read_raw(LANK, "lankershim"), planar=True))
    row = ps[(ps["vehicle_id"] == vid) & (ps["t"] == round(tc))].iloc[0]
    phi = index_features(row["speed_kmh"], row["accel"], row["gap_m"], row["wave_m"])
    return {"speed_kmh": row["speed_kmh"], "accel": row["accel"], "gap": row["gap_m"], "wave": row["wave_m"],
            "score": float(original_score(phi))}


# ── pNEUMA ──────────────────────────────────────────────────────────────────
LEN = {"Car": 4.5, "Taxi": 4.5, "Medium Vehicle": 6.5, "Heavy Vehicle": 10.0, "Bus": 12.0, "Motorcycle": 2.0}


def pneuma(pick=None):
    veh = {}
    with open(PNEUMA) as f:
        next(f)
        for line in f:
            p = [x.strip() for x in line.strip().strip(";").split(";")]
            if len(p) < 10:
                continue
            n = (len(p) - 4) // 6
            veh[int(p[0])] = (p[1], [tuple(float(v) for v in p[4 + 6 * k:10 + 6 * k]) for k in range(n)])
    allpts = [s for _, ss in veh.values() for s in ss]
    lat0 = sum(s[0] for s in allpts) / len(allpts)
    lon0 = sum(s[1] for s in allpts) / len(allpts)
    R = 6371000.0

    def xy(lat, lon):
        return math.radians(lon - lon0) * R * math.cos(math.radians(lat0)), math.radians(lat - lat0) * R

    per = {}   # vid -> {second: (x, y, speed_kmh)}
    for vid, (typ, ss) in veh.items():
        d = {}
        for lat, lon, spd, _, _, tt in ss:
            fr = tt % 1.0
            if fr < 0.021 or fr > 0.979:
                sec = round(tt)
                if sec not in d:
                    d[sec] = (*xy(lat, lon), spd)
        per[vid] = d
    if pick is None:
        for vid in sorted(per):
            if veh[vid][0] != "Car" or len(per[vid]) < 40:
                continue
            secs = sorted(per[vid])
            pick = (vid, secs[len(secs) // 2])
            break
    vid, sec = pick
    secs = sorted(per[vid])
    k = secs.index(sec)
    v = [per[vid][s][2] / 3.6 for s in secs]
    vs = roll(v, 1, 1)
    accel = max(-9.0, min(9.0, grad([float(s) for s in secs], vs, k)))
    x, y = per[vid][sec][0], per[vid][sec][1]
    dx = per[vid][secs[k + 1]][0] - per[vid][secs[k - 1]][0]
    dy = per[vid][secs[k + 1]][1] - per[vid][secs[k - 1]][1]
    h = math.atan2(dy, dx)
    def heading(d, s):
        """heading from the positions 1 s before and after; a car that moved <= 0.5 m keeps its last
        heading while moving (else the next one), as in datasets/pneuma.py"""
        ss = sorted(d)
        hs = []
        for j in range(len(ss)):
            if 0 < j < len(ss) - 1:
                ddx, ddy = d[ss[j + 1]][0] - d[ss[j - 1]][0], d[ss[j + 1]][1] - d[ss[j - 1]][1]
                hs.append(math.atan2(ddy, ddx) if math.hypot(ddx, ddy) > 0.5 else None)
            else:
                hs.append(None)
        j = ss.index(s)
        back = [x for x in hs[:j + 1] if x is not None]
        fwd = [x for x in hs[j:] if x is not None]
        return back[-1] if back else (fwd[0] if fwd else None)

    best, near, stopped_leader = None, 0, False
    for oid, d in per.items():
        if oid == vid or sec not in d:
            continue
        ox, oy = d[sec][0], d[sec][1]
        if math.hypot(ox - x, oy - y) <= 50.0:
            near += 1
        oh = heading(d, sec)
        if oh is None:
            continue
        lon = (ox - x) * math.cos(h) + (oy - y) * math.sin(h)
        lat = -(ox - x) * math.sin(h) + (oy - y) * math.cos(h)
        dh = abs((oh - h + math.pi) % (2 * math.pi) - math.pi)
        if 0 < lon <= 100 and abs(lat) <= 1.6 and dh <= math.radians(30) and (best is None or lon < best[1]):
            best = (oid, lon, lat)
    gap = max(0.1, best[1] - (LEN[veh[vid][0]] + LEN[veh[best[0]][0]]) / 2) if best else 0.0
    if best:
        d = per[best[0]]
        stopped_leader = d[sec][2] < 1.0
    phi, score = terms(vs[k] * 3.6, accel, gap, 0.0)
    return {"vid": vid, "sec": sec, "type": veh[vid][0], "raw_speeds_kmh": [per[vid][s][2] for s in secs[k - 2:k + 3]],
            "speed_kmh": vs[k] * 3.6, "accel": accel, "heading_deg": math.degrees(h), "leader": best,
            "leader_type": veh[best[0]][0] if best else None, "leader_stopped": stopped_leader, "gap": gap, "density": near, "phi": phi, "score": score}


def pneuma_pipeline(vid, sec):
    import pandas as pd
    from model.aggressiveness_model import index_features, original_score
    ps = pd.read_csv(PNEUMA.replace(".csv", ".per_second.csv.gz"))
    row = ps[(ps["vehicle_id"] == vid) & (ps["t"] == sec)].iloc[0]
    phi = index_features(row["speed_kmh"], row["accel"], row["gap_m"], row["wave_m"])
    return {"speed_kmh": row["speed_kmh"], "accel": row["accel"], "gap": row["gap_m"], "density": row["density_50m"],
            "score": float(original_score(phi))}


def page(lk, lp, pn, pp):
    f = lambda x, d=3: f"{x:.{d}f}"  # noqa: E731
    L = ["# hand check: one NGSIM arterial row and one pNEUMA second", "",
         "hand numbers: `scripts/hand_check_more.py` (plain python on the raw CSV rows); pipeline: `datasets/ngsim.py`",
         "(`trajectories(planar=True)`, `per_second`) and `datasets/pneuma.py` (`per_second`).", "",
         f"## ngsim lankershim, vehicle {lk['vid']} (period {lk['vid'] // 100000}, id {lk['vid'] % 100000}), t = {lk['t']:.1f} s",
         "", f"frame {lk['frame']}, lane {lk['lane']}, direction {lk['dir']:.0f}, section {lk['sec']:.0f}, through movement, not in an intersection.", "",
         f"- speed: 2-D path (arterial). velocity from the 10 frame centred means of Local_X and Local_Y: vx {f(lk['vx'])}, vy {f(lk['vy'])} m/s, "
         f"|v| = **{f(lk['speed_kmh'], 2)} km/h**",
         f"- accel_1hz: 3 s mean of the per second speeds, previous and next second {f(lk['v1_prev'], 4)} and {f(lk['v1_next'], 4)} m/s, "
         f"difference / 2 s = **{f(lk['accel'])} m/s2**",
         f"- gap: Space_Headway {lk['sh_ft']} ft minus leader {lk['lead']} length {lk['lead_len_ft']} ft, x 0.3048 = **{f(lk['gap'], 2)} m**",
         f"- wave: smoothed Local_X {f(lk['xs'])} m, lane centre = median of {lk['n_centre']} smoothed rows in the same lane, direction, "
         f"section and 10 m of road {f(lk['centre'])} m, |difference| = **{f(lk['wave'])} m**",
         f"- features {', '.join(f(x, 4) for x in lk['phi'])}, score **{f(lk['score'], 2)}**", "",
         f"pipeline: speed {f(lp['speed_kmh'], 2)}, accel {f(lp['accel'])}, gap {f(lp['gap'], 2)}, wave {f(lp['wave'])}, "
         f"score **{f(lp['score'], 2)}**; difference {abs(lk['score'] - lp['score']):.2e}.", "",
         f"## pneuma, vehicle {pn['vid']} ({pn['type']}), second {pn['sec']}", "",
         f"- speed: pNEUMA speed at the whole seconds around it {', '.join(f(x, 2) for x in pn['raw_speeds_kmh'])} km/h, "
         f"3 s mean = **{f(pn['speed_kmh'], 2)} km/h**",
         f"- accel: derivative of the 3 s mean speed = **{f(pn['accel'])} m/s2**",
         f"- heading from the positions 1 s before and after: {f(pn['heading_deg'], 1)} deg",
         (f"- gap: nearest vehicle ahead in the cone (+-1.6 m lateral, heading within 30 deg) is {pn['leader'][0]} "
          f"({pn['leader_type']}{', stopped: its heading is the last one while moving' if pn['leader_stopped'] else ''}), {f(pn['leader'][1], 2)} m ahead and {f(pn['leader'][2], 2)} m to the side; minus half of both "
          f"lengths = **{f(pn['gap'], 2)} m**") if pn["leader"] else "- gap: no vehicle in the cone = **0**",
         "- wave: not measurable (no lane map), 0",
         f"- vehicles within 50 m: {pn['density']}",
         f"- features {', '.join(f(x, 4) for x in pn['phi'])}, score **{f(pn['score'], 2)}**", "",
         f"pipeline: speed {f(pp['speed_kmh'], 2)}, accel {f(pp['accel'])}, gap {f(pp['gap'], 2)}, density {pp['density']:.0f}, "
         f"score **{f(pp['score'], 2)}**; difference {abs(pn['score'] - pp['score']):.2e}.", ""]
    path = os.path.join(ROOT, "docs", "hand_checks", "hand_check_arterial_pneuma.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    lk = lankershim()
    lp = lankershim_pipeline(lk["vid"], lk["t"])
    print("lankershim", {k: lk[k] for k in ("vid", "t", "speed_kmh", "accel", "gap", "wave", "score")})
    print("pipeline  ", lp)
    pn = pneuma()
    pp = pneuma_pipeline(pn["vid"], pn["sec"])
    print("pneuma", {k: pn[k] for k in ("vid", "sec", "speed_kmh", "accel", "gap", "density", "score")})
    print("pipeline", pp)
    print("written", page(lk, lp, pn, pp))
