"""
Hand check of the Aggressiveness Index on real data, recomputed from the raw files
with plain Python arithmetic (no pandas, no numpy, no pipeline code) and compared
with what the pipeline produces for the same window.

    python scripts/hand_check.py            # writes docs/hand_checks/hand_check_3_windows.md

Windows (fixed, so the numbers can be checked again later):
    UAH D1 NORMAL motorway,     t0 = 306.88 s (a typical window of the trip)
    UAH D1 AGGRESSIVE motorway, t0 = 688.94 s (the highest scoring window with a leader)
    NGSIM US-101, one car at one second (see NGSIM_CAR below)
"""
import bisect
import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "..", "28:9:2026")
UAH = os.path.join(DATA, "UAH-DRIVESET-v1", "D1")
UAH_WINDOWS = [
    ("UAH D1 normal motorway", "20151111123124-25km-D1-NORMAL-MOTORWAY", 306.88),
    ("UAH D1 aggressive motorway", "20151111125233-24km-D1-AGGRESSIVE-MOTORWAY", 688.94),
]
NGSIM_FILE = os.path.join(DATA, "ngsim_us-101_5min.csv")
NGSIM_CAR = (983, 259.0)           # (Vehicle_ID, t in s from the start of the extract): a car near the 90th
                                   # percentile of mean score, mid trajectory, with a leader
FT = 0.3048

# reference constants written out again by hand on purpose (not imported)
W = (0.5, 0.2, 0.8, 0.4)


def score_terms(speed_kmh, accel, gap, wave):
    ns = min(speed_kmh / 150.0, 1.0)
    na = min(abs(accel) / 5.0, 1.0)
    npx = 1.0 - gap / 50.0 if 0 < gap <= 50 else 0.0
    nw = min(abs(wave) / 1.5, 1.0)
    phi = (ns * ns, na, npx * npx, nw)
    return (ns, na, npx, nw), phi


def lbl(s):
    return "Conservative" if s < 29 else ("Normal" if s < 42 else "Aggressive")


def gradient_at(t, v, i):
    """np.gradient for uneven spacing at an interior point i (second order):
    (hl^2 v[i+1] - hr^2 v[i-1] + (hr^2 - hl^2) v[i]) / (hl hr (hl + hr)), hl = t[i]-t[i-1], hr = t[i+1]-t[i].
    Equal spacing h reduces it to (v[i+1] - v[i-1]) / 2h."""
    hl, hr = t[i] - t[i - 1], t[i + 1] - t[i]
    return (hl * hl * v[i + 1] - hr * hr * v[i - 1] + (hr * hr - hl * hl) * v[i]) / (hl * hr * (hl + hr))


def read_rows(path):
    with open(path) as f:
        return [[float(x) for x in line.split()] for line in f if line.strip()]


# ── UAH ─────────────────────────────────────────────────────────────────────
def uah_window(folder, t0, length=10.0):
    p = os.path.join(UAH, folder)
    gps = {}
    for r in read_rows(os.path.join(p, "RAW_GPS.txt")):
        if 0 <= r[1] < 250 and r[0] not in gps:
            gps[r[0]] = r[1]
    t = sorted(gps)
    raw_kmh = [gps[x] for x in t]
    v = [x / 3.6 for x in raw_kmh]
    vs = [sum(v[max(0, i - 1):i + 2]) / len(v[max(0, i - 1):i + 2]) for i in range(len(v))]   # 3 s centred mean

    lane = [r for r in read_rows(os.path.join(p, "PROC_LANE_DETECTION.txt")) if r[3] > 0 and abs(r[1]) <= 2.0]
    lt = [r[0] for r in lane]
    veh = read_rows(os.path.join(p, "PROC_VEHICLE_DETECTION.txt"))
    vt = [r[0] for r in veh]

    seconds = []
    for i, ti in enumerate(t):
        if not (t0 <= ti < t0 + length):
            continue
        a = max(-9.0, min(9.0, gradient_at(t, vs, i)))
        k = min(bisect.bisect_left(lt, ti), len(lt) - 1)           # first lane row at or after t
        wave = abs(lane[k][1]) if abs(lt[k] - ti) <= 1.0 else 0.0
        j = min(bisect.bisect_left(vt, ti), len(vt) - 1)           # first vehicle row at or after t
        gap = (veh[j][1] if veh[j][1] > 0 else 0.0) if abs(vt[j] - ti) <= 1.5 else 0.0
        n, phi = score_terms(vs[i] * 3.6, a, gap, wave)
        seconds.append({
            "t": ti, "gps_prev": raw_kmh[i - 1], "gps": raw_kmh[i], "gps_next": raw_kmh[i + 1],
            "dt_l": ti - t[i - 1], "dt_r": t[i + 1] - ti,
            "v_ms": vs[i], "speed_kmh": vs[i] * 3.6, "accel": a,
            "lane_t": lt[k], "wave": wave, "veh_t": vt[j], "gap": gap,
            "n": n, "phi": phi, "score": min(100.0 * sum(w * x for w, x in zip(W, phi)), 100.0),
        })
    mean_phi = [sum(s["phi"][q] for s in seconds) / len(seconds) for q in range(4)]
    terms = [w * x for w, x in zip(W, mean_phi)]
    score = min(100.0 * sum(terms), 100.0)
    return {"seconds": seconds, "phi": mean_phi, "terms": terms, "score": score, "label": lbl(score)}


def uah_pipeline(folder, t0):
    from datasets.uah import find_trips, load_trip, windows
    from model.aggressiveness_model import original_score, label
    trip = next(x for x in find_trips(os.path.join(DATA, "UAH-DRIVESET-v1")) if x["trip"] == folder)
    w = windows(load_trip(trip), trip)
    row = w.loc[(w["t0"] - t0).abs() < 1e-6].iloc[0]
    phi = [row["phi_speed"], row["phi_accel"], row["phi_prox"], row["phi_wave"]]
    s = float(original_score(phi))
    return {"phi": phi, "score": s, "label": label(s)}


# ── NGSIM ───────────────────────────────────────────────────────────────────
def _roll10(x):
    """pandas rolling(10, center=True, min_periods=1): rows i-5 .. i+4."""
    return [sum(x[max(0, i - 5):i + 5]) / len(x[max(0, i - 5):i + 5]) for i in range(len(x))]


def ngsim_car(vid, t_check):
    """Features of one car at one 10 Hz row, from the raw columns, the way datasets/ngsim.py defines them:
    position smoothed over 1 s (10 frames, centred), speed = its gradient (clipped at 0);
    accel_1hz = per second mean speed, 3 s centred mean, central difference over seconds;
    gap = (Space_Headway minus the leader's length) in metres;
    wave = |smoothed Local_X minus the lane centre|, lane centre = median smoothed Local_X of all rows in the lane."""
    with open(NGSIM_FILE) as f:
        rows = list(csv.DictReader(f))
    t_min = min(float(r["Global_Time"]) for r in rows)
    seen, cars, length = set(), {}, {}
    for r in rows:
        key = (int(r["Vehicle_ID"]), int(r["Frame_ID"]))
        if key in seen:
            continue
        seen.add(key)
        r["t"] = (float(r["Global_Time"]) - t_min) / 1000.0
        cars.setdefault(key[0], []).append(r)
        length[key] = float(r["v_Length"])
    lane_lat = {}
    for c in cars.values():
        c.sort(key=lambda r: r["t"])
        for r, x in zip(c, _roll10([float(r["Local_X"]) * FT for r in c])):
            r["lat_s"] = x
            lane_lat.setdefault(r["Lane_ID"], []).append(x)

    car = cars[vid]
    t = [r["t"] for r in car]
    pos_s = _roll10([float(r["Local_Y"]) * FT for r in car])
    speed = [max(0.0, (pos_s[1] - pos_s[0]) / (t[1] - t[0]))] + \
            [max(0.0, gradient_at(t, pos_s, i)) for i in range(1, len(t) - 1)] + \
            [max(0.0, (pos_s[-1] - pos_s[-2]) / (t[-1] - t[-2]))]

    secs = {}
    for ti, sp in zip(t, speed):
        secs.setdefault(int(ti // 1), []).append(sp)
    sec = sorted(secs)
    v1 = [sum(secs[q]) / len(secs[q]) for q in sec]
    v1s = [sum(v1[max(0, i - 1):i + 2]) / len(v1[max(0, i - 1):i + 2]) for i in range(len(v1))]

    i = min(range(len(t)), key=lambda k: abs(t[k] - t_check))
    r = car[i]
    si = sec.index(int(t[i] // 1))
    a1 = max(-9.0, min(9.0, (v1s[si + 1] - v1s[si - 1]) / (sec[si + 1] - sec[si - 1])))

    lead_id = int(float(r["Preceding"]))
    lead_len = length.get((lead_id, int(r["Frame_ID"])), 15.0)
    sh = float(r["Space_Headway"])
    gap = max(0.1, (sh - lead_len) * FT) if lead_id > 0 and sh > 0 else 0.0

    ll = sorted(lane_lat[r["Lane_ID"]])
    m = len(ll)
    centre = ll[m // 2] if m % 2 else (ll[m // 2 - 1] + ll[m // 2]) / 2.0
    wave = abs(r["lat_s"] - centre)

    out = {"t": t[i], "frame": int(r["Frame_ID"]), "lane": r["Lane_ID"],
           "local_y_ft": [float(x["Local_Y"]) for x in car[i - 5:i + 6]], "pos_s": (pos_s[i - 1], pos_s[i + 1]),
           "dt": (t[i] - t[i - 1], t[i + 1] - t[i]), "speed_ms": speed[i], "speed_kmh": speed[i] * 3.6,
           "sec_mean_speeds": v1[si - 2:si + 3], "v1s_prev": v1s[si - 1], "v1s_next": v1s[si + 1],
           "accel": a1, "space_headway_ft": sh, "lead_id": lead_id, "lead_len_ft": lead_len, "gap": gap,
           "lat_s": r["lat_s"], "lane_centre": centre, "lane_rows": m, "wave": wave}
    n, phi = score_terms(out["speed_kmh"], a1, gap, wave)
    out["n"], out["phi"] = n, phi
    out["terms"] = [w * x for w, x in zip(W, phi)]
    out["score"] = min(100.0 * sum(out["terms"]), 100.0)
    out["label"] = lbl(out["score"])
    return out


def ngsim_pipeline(vid, t_check):
    from datasets.ngsim import read_raw, trajectories
    from model.aggressiveness_model import index_features, original_score, label
    tr = trajectories(read_raw(NGSIM_FILE, "us-101"))
    car = tr[tr["vehicle_id"] == vid]
    row = car.iloc[int((car["t"] - t_check).abs().argmin())]
    phi = index_features(row["speed"] * 3.6, row["accel_1hz"], row["gap"], row["wave"])
    s = float(original_score(phi))
    return {"speed_kmh": float(row["speed"]) * 3.6, "accel": float(row["accel_1hz"]), "gap": float(row["gap"]),
            "wave": float(row["wave"]), "phi": [float(x) for x in phi], "score": s, "label": label(s)}


def _f(x, d=4):
    return f"{x:.{d}f}"


def write_page(path):
    L = ["# hand check: aggressiveness index on 3 real windows", "",
         "reference: `model/aggressiveness_model.py`. hand numbers: `scripts/hand_check.py` (plain python on the raw",
         "files, no pipeline code). pipeline numbers: `datasets/uah.py` `windows` and `datasets/ngsim.py` `trajectories`.", "",
         "formula: AI = min(100 (0.5 n_s^2 + 0.2 n_a + 0.8 n_p^2 + 0.4 n_w), 100); n_s = min(v_kmh/150, 1),",
         "n_a = min(|a|/5, 1), n_p = 1 - gap/50 if 0 < gap <= 50 else 0, n_w = min(|wave|/1.5, 1);",
         "labels: < 29 conservative, < 42 normal, else aggressive (fitted, scripts/fit_cutoffs.py).", ""]
    for name, folder, t0 in UAH_WINDOWS:
        h, p = uah_window(folder, t0), uah_pipeline(folder, t0)
        L += [f"## {name.lower()}, window t0 = {t0} s, 10 s", "",
              f"trip `{folder}`. one row per GPS second; the window score is the score of the mean features",
              "(equal to the mean of the per second scores while no second reaches the cap of 100).", "",
              "- speed: RAW_GPS col 1 (km/h), / 3.6, 3 s centred mean, x 3.6 back to km/h, / 150, squared",
              "- accel: derivative of the smoothed speed at the GPS timestamps (np.gradient, second order for uneven dt;",
              "  for equal dt it is (v_next - v_prev) / 2 dt), clip to +-9, |a| / 5",
              "- gap: PROC_VEHICLE_DETECTION col 1 (m, to the vehicle ahead seen by the phone camera), first row at or",
              "  after the GPS time and within 1.5 s; <= 0 means no vehicle; 1 - gap/50, squared",
              "- wave: PROC_LANE_DETECTION col 1 (m, car position from the lane centre), rows with road width > 0 and",
              "  |offset| <= 2 m, first row at or after the GPS time and within 1 s; |x| / 1.5", "",
              "| t (s) | gps prev / now / next (km/h) | dt l / r (s) | speed (km/h) | accel (m/s2) | gap (m) | wave (m) | n_s^2 | n_a | n_p^2 | n_w | score |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for q in h["seconds"]:
            L.append(f"| {q['t']:.2f} | {q['gps_prev']} / {q['gps']} / {q['gps_next']} | {q['dt_l']:.2f} / {q['dt_r']:.2f} | "
                     f"{_f(q['speed_kmh'], 2)} | {_f(q['accel'], 3)} | {_f(q['gap'], 2)} | {_f(q['wave'], 3)} | "
                     + " | ".join(_f(x) for x in q["phi"]) + f" | {_f(q['score'], 2)} |")
        L += ["", f"mean features: n_s^2 {_f(h['phi'][0])}, n_a {_f(h['phi'][1])}, n_p^2 {_f(h['phi'][2])}, n_w {_f(h['phi'][3])}",
              f"weighted: 0.5 x {_f(h['phi'][0])} = {_f(h['terms'][0])}; 0.2 x {_f(h['phi'][1])} = {_f(h['terms'][1])}; "
              f"0.8 x {_f(h['phi'][2])} = {_f(h['terms'][2])}; 0.4 x {_f(h['phi'][3])} = {_f(h['terms'][3])}",
              f"sum {_f(sum(h['terms']))}, x 100 = **{_f(h['score'], 2)}**, label **{h['label'].lower()}**", "",
              f"pipeline: features {', '.join(_f(x) for x in p['phi'])}, score **{_f(p['score'], 2)}**, label {p['label'].lower()}. "
              f"difference {abs(h['score'] - p['score']):.2e}.", ""]
    h, p = ngsim_car(*NGSIM_CAR), ngsim_pipeline(*NGSIM_CAR)
    vs = ", ".join(_f(x, 3) for x in h["sec_mean_speeds"])
    L += [f"## ngsim us-101, car {NGSIM_CAR[0]} at t = {h['t']:.1f} s (frame {h['frame']}, lane {h['lane']})", "",
          "one 10 Hz row; the vehicle score in `score_ngsim.py` is the mean of these row scores.", "",
          f"- speed: Local_Y (ft) x 0.3048, 10 frame centred mean, gradient over t. Local_Y rows i-5..i+5: "
          f"{', '.join(str(x) for x in h['local_y_ft'])} ft. smoothed position at i-1 and i+1: "
          f"{_f(h['pos_s'][0], 3)}, {_f(h['pos_s'][1], 3)} m, dt {h['dt'][0]:.2f} / {h['dt'][1]:.2f} s, "
          f"speed {_f(h['speed_ms'])} m/s = **{_f(h['speed_kmh'], 2)} km/h**",
          f"- accel (accel_1hz, measured like UAH): per second mean speeds around this second {vs} m/s; 3 s centred means "
          f"of the previous and next second {_f(h['v1s_prev'], 4)} and {_f(h['v1s_next'], 4)} m/s; "
          f"({_f(h['v1s_next'], 4)} - {_f(h['v1s_prev'], 4)}) / 2 s = **{_f(h['accel'], 3)} m/s2**",
          f"- gap: Space_Headway {h['space_headway_ft']} ft (front to front) minus leader {h['lead_id']} length "
          f"{h['lead_len_ft']} ft, x 0.3048 = **{_f(h['gap'], 2)} m**",
          f"- wave: smoothed Local_X {_f(h['lat_s'], 3)} m, lane centre (median of {h['lane_rows']} smoothed rows in the lane) "
          f"{_f(h['lane_centre'], 3)} m, |difference| = **{_f(h['wave'], 3)} m**", "",
          f"n_s = {_f(h['speed_kmh'], 2)}/150 = {_f(h['n'][0])}, squared {_f(h['phi'][0])}; "
          f"n_a = {_f(h['accel'], 3)}/5 = {_f(h['n'][1])}; n_p = 1 - {_f(h['gap'], 2)}/50 = {_f(h['n'][2])}, squared {_f(h['phi'][2])}; "
          f"n_w = {_f(h['wave'], 3)}/1.5 = {_f(h['n'][3])}",
          f"weighted: {_f(h['terms'][0])} + {_f(h['terms'][1])} + {_f(h['terms'][2])} + {_f(h['terms'][3])} = {_f(sum(h['terms']))}, "
          f"x 100 = **{_f(h['score'], 2)}**, label **{h['label'].lower()}**", "",
          f"pipeline: speed {_f(p['speed_kmh'], 2)} km/h, accel {_f(p['accel'], 3)}, gap {_f(p['gap'], 2)} m, wave {_f(p['wave'], 3)} m, "
          f"score **{_f(p['score'], 2)}**, label {p['label'].lower()}. difference {abs(h['score'] - p['score']):.2e}.", "",
          f"what drives this score: proximity {_f(100 * h['terms'][2], 1)} of {_f(h['score'], 1)} points, from a 15.8 m gap "
          f"at {_f(h['speed_kmh'], 0)} km/h (time headway {h['gap'] / h['speed_ms']:.1f} s, ordinary for dense traffic).", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


if __name__ == "__main__":
    for name, folder, t0 in UAH_WINDOWS:
        h, p = uah_window(folder, t0), uah_pipeline(folder, t0)
        print(name, "hand", round(h["score"], 4), h["label"], "pipeline", round(p["score"], 4), p["label"])
    h, p = ngsim_car(*NGSIM_CAR), ngsim_pipeline(*NGSIM_CAR)
    for k in ["speed_kmh", "accel", "gap", "wave", "score"]:
        print(f"NGSIM car {NGSIM_CAR[0]} {k:9s} hand {h[k]:.4f}  pipeline {p[k]:.4f}")
    print("labels", h["label"], p["label"])
    print("written", write_page(os.path.join(ROOT, "docs", "hand_checks", "hand_check_3_windows.md")))
