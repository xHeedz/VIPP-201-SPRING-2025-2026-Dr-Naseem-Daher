"""
German drone traffic as the reference for normal driving: score a window as its percentile among German windows
in the same context, and test the rule on labelled data.

    python scripts/context_reference.py          (needs data/german_windows.csv.gz from scripts/score_german.py)

Two context definitions:
  own_speed   own speed band (20 km/h) x car ahead within 50 m (metres proximity term > 0); highway reference
              (highD + exiD) only. Removes the speed signal: own speed is behaviour, not context.
  traffic     environment x traffic speed band (20 km/h, mean speed of the OTHER vehicles at the same second,
              window mean) x car ahead. reference: highway = highD + exiD, urban = inD + rounD + uniD
              (weather settings use the highway reference).
Bins with fewer than MIN_REF reference windows fall back to the band without the car ahead split, then to the
nearest band. Context score = 100 x share of reference windows in the bin with a lower score. Rule: aggressive if
context score >= 90 (top 10% of German drivers in the same context), against the fixed cut off 42.
Labelled tests: planted SUMO drivers, test seeds 7 to 9 (all settings), UAH motorway (own speed only: UAH has no
traffic state). Unlabelled: NGSIM US-101 (share flagged). Proximity: metres, headway, mix (PROX_MIX of the
environment).
Outputs data/context_reference.csv.
"""
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.sumo_log import windows_from_per_second  # noqa: E402
from datasets.uah import load_windows  # noqa: E402
from model.aggressiveness_model import PROX_MIX, THRESHOLDS, original_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_proximity_mix import paired_windows  # noqa: E402
from train_uah import auc  # noqa: E402

UAH_ROOT = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "UAH-DRIVESET-v1"))
NGSIM = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "ngsim_us-101_5min.csv"))
BAND_KMH, MIN_REF, PCT = 20.0, 500, 90.0
REF_ENV = {"highway": "highway", "weather": "highway", "urban": "urban"}


def scores(w, variant):
    """score of a window table with phi_prox_m (metres) and phi_prox_h (headway); mix uses each row's environment"""
    if variant == "metres":
        a = np.ones(len(w))
    elif variant == "headway":
        a = np.zeros(len(w))
    else:
        a = w["environment"].map(PROX_MIX).to_numpy(float)
    f = w[["phi_speed", "phi_accel", "phi_prox_m", "phi_wave"]].to_numpy().copy()
    f[:, 2] = a * w["phi_prox_m"].to_numpy() + (1 - a) * w["phi_prox_h"].to_numpy()
    return original_score(f)


def bins(w, mode):
    speed = w["speed_kmh"] if mode == "own_speed" else w["traffic_kmh"].fillna(w["speed_kmh"])
    band = np.floor(speed.to_numpy() / BAND_KMH).astype(int)
    lead = (w["phi_prox_m"].to_numpy() > 0).astype(int)
    env = w["environment"].map(REF_ENV).to_numpy() if mode == "traffic" else np.full(len(w), "highway")
    return env, band, lead


class Reference:
    def __init__(self, ref_scores, env, band, lead):
        self.full, self.band, self.env = {}, {}, {}
        for e in np.unique(env):
            me = env == e
            self.env[e] = sorted(np.unique(band[me]))
            for b in np.unique(band[me]):
                m = me & (band == b)
                self.band[(e, b)] = np.sort(ref_scores[m])
                for l in (0, 1):
                    if (m & (lead == l)).sum() >= MIN_REF:
                        self.full[(e, b, l)] = np.sort(ref_scores[m & (lead == l)])

    def percentile(self, s, env, band, lead):
        out = np.full(len(s), np.nan)
        for i, (v, e, b, l) in enumerate(zip(s, env, band, lead)):
            ref = self.full.get((e, b, l))
            if ref is None:
                ref = self.band.get((e, b))
            if ref is None or len(ref) < MIN_REF:
                nb = min(self.env[e], key=lambda k: (len(self.band[(e, k)]) < MIN_REF, abs(k - b)))
                ref = self.band[(e, nb)]
            out[i] = 100.0 * np.searchsorted(ref, v, side="left") / len(ref)
        return out


def traffic_speed(ps):
    g = ps.groupby("t")["speed_kmh"]
    n, tot = g.transform("size"), g.transform("sum")
    return np.where(n > 1, (tot - ps["speed_kmh"]) / (n - 1).clip(lower=1), np.nan)


def window_traffic(w, ps):
    """window mean of the per second traffic speed, matched on vehicle and window start"""
    ps = ps.assign(traffic_kmh=traffic_speed(ps))
    out = np.full(len(w), np.nan)
    by = {v: (g["t"].to_numpy(), g["traffic_kmh"].to_numpy()) for v, g in ps.groupby("vehicle_id", sort=False)}
    for i, (v, t0) in enumerate(zip(w["trip"], w["t0"])):
        t, tr = by[v]
        m = (t >= t0) & (t < t0 + 10)
        out[i] = np.nanmean(tr[m]) if m.any() else np.nan
    return out


def german():
    w = pd.read_csv(os.path.join(DATA_DIR, "german_windows.csv.gz"))
    return w.rename(columns={"phi_prox": "phi_prox_m", "phi_prox_headway": "phi_prox_h"})


def planted_test():
    from planted_drivers import ENVIRONMENT
    frames = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", "*_s*.per_second.csv.gz"))):
        m = re.match(r"([a-z0-9]+)_([a-z]+)_s(\d+)\.per_second", os.path.basename(p))
        if int(m.group(3)) < 7 or m.group(1) == "us101":
            continue
        ps = pd.read_csv(p)
        scen, env = f"{m.group(1)}_{m.group(2)}", ENVIRONMENT[m.group(1)]
        w = paired_windows(lambda f: windows_from_per_second(ps, scen, env, features=f))
        frames.append(w.assign(scenario=scen, traffic_kmh=window_traffic(w, ps)))
    return pd.concat(frames, ignore_index=True)


def ngsim_us101():
    from datasets.ngsim import per_second, read_raw, trajectories
    ps = per_second(trajectories(read_raw(NGSIM, "us-101"))).assign(label="unlabelled")
    w = paired_windows(lambda f: windows_from_per_second(ps, "us101", "highway", features=f))
    return w.assign(traffic_kmh=window_traffic(w, ps))


def evaluate(name, w, refs, mode, labelled=True):
    rows = []
    env, band, lead = bins(w, mode)
    for variant, ref in refs.items():
        s = scores(w, variant)
        pct = ref.percentile(s, env, band, lead)
        r = {"set": name, "context": mode, "proximity": variant, "windows": len(w),
             "flag_fixed42": float((s >= THRESHOLDS[1]).mean()), "flag_context90": float((pct >= PCT).mean())}
        if labelled:
            y = (w["behavior"] == "aggressive").to_numpy()
            nm = (w["behavior"] == "normal").to_numpy()
            k = y | nm
            r.update({"auc_fixed": auc(s[k], y[k].astype(int)), "auc_context": auc(pct[k], y[k].astype(int)),
                      "tpr_fixed42": float((s[y] >= THRESHOLDS[1]).mean()), "fpr_fixed42": float((s[nm] >= THRESHOLDS[1]).mean()),
                      "tpr_context90": float((pct[y] >= PCT).mean()), "fpr_context90": float((pct[nm] >= PCT).mean())})
        rows.append(r)
    return rows


def main():
    g = german()
    refs = {}
    for mode in ("own_speed", "traffic"):
        gm = g if mode == "traffic" else g[g["environment"] == "highway"]
        e, b, l = bins(gm, mode)
        refs[mode] = {v: Reference(scores(gm, v), e, b, l) for v in ("metres", "headway", "mix")}
        print(f"{mode}: {len(gm)} reference windows, {len(refs[mode]['metres'].full)} full bins", flush=True)
    pl = planted_test()
    us = ngsim_us101()
    uah = paired_windows(lambda f: load_windows(UAH_ROOT, features=f)[0])
    uah = uah[uah["environment"] == "highway"]
    rows = evaluate("UAH motorway", uah, refs["own_speed"], "own_speed")
    for mode in ("own_speed", "traffic"):
        hw = pl[pl["environment"] != "urban"] if mode == "own_speed" else pl
        rows += evaluate("planted SUMO, test seeds" + (" (highway settings)" if mode == "own_speed" else ""), hw, refs[mode], mode)
        for sc, g2 in hw.groupby("scenario"):
            rows += evaluate(f"planted {sc}", g2, refs[mode], mode)
        rows += evaluate("NGSIM US-101 (unlabelled)", us, refs[mode], mode, labelled=False)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA_DIR, "context_reference.csv"), index=False)
    with pd.option_context("display.width", 250):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
