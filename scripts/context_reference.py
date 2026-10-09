"""
German motorway traffic (highD + exiD) as the reference for normal driving: score a window as its percentile
among German windows in the same context, and test the rule on labelled motorway data.

    python scripts/context_reference.py          (needs data/german_windows.csv.gz from scripts/score_german.py)

Context bin: speed band (20 km/h) x car ahead within 50 m (metres proximity term > 0). Bins with fewer than
MIN_REF reference windows fall back to the speed band alone. Context score = 100 x share of reference windows in
the bin with a lower score. Rule: aggressive if context score >= 90 (top 10% of German drivers in the same
situation), against the fixed cut off 42.
Labelled tests (highway only, the reference is motorway traffic): UAH motorway trips (normal vs aggressive) and
the planted SUMO highway settings, test seeds 7 to 9 (highway low / medium, jam, merge). Unlabelled: NGSIM US-101.
Proximity variants: metres, headway, mix (PROX_MIX["highway"]).
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
from model.aggressiveness_model import PROX_MIX, THRESHOLDS, headway_features, index_features, original_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_proximity_mix import paired_windows  # noqa: E402
from train_uah import auc  # noqa: E402

UAH_ROOT = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "UAH-DRIVESET-v1"))
NGSIM = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "ngsim_us-101_5min.csv"))
BAND_KMH, MIN_REF, PCT = 20.0, 500, 90.0
ALPHA = PROX_MIX["highway"]


def scores(w, variant):
    """score of a window table with phi_prox_m (metres) and phi_prox_h (headway)"""
    a = {"metres": 1.0, "headway": 0.0, "mix": ALPHA}[variant]
    f = w[["phi_speed", "phi_accel", "phi_prox_m", "phi_wave"]].to_numpy().copy()
    f[:, 2] = a * w["phi_prox_m"].to_numpy() + (1 - a) * w["phi_prox_h"].to_numpy()
    return original_score(f)


def context(w):
    band = np.floor(w["speed_kmh"].to_numpy() / BAND_KMH).astype(int)
    lead = (w["phi_prox_m"].to_numpy() > 0).astype(int)
    return band, lead


class Reference:
    def __init__(self, ref_scores, band, lead):
        self.full, self.band = {}, {}
        for b in np.unique(band):
            m = band == b
            self.band[b] = np.sort(ref_scores[m])
            for l in (0, 1):
                mm = m & (lead == l)
                if mm.sum() >= MIN_REF:
                    self.full[(b, l)] = np.sort(ref_scores[mm])

    def percentile(self, s, band, lead):
        out = np.full(len(s), np.nan)
        for i, (v, b, l) in enumerate(zip(s, band, lead)):
            ref = self.full.get((b, l))
            if ref is None:
                ref = self.band.get(b)
            if ref is None:     # speed outside every reference band: nearest band
                ref = self.band[min(self.band, key=lambda k: abs(k - b))]
            out[i] = 100.0 * np.searchsorted(ref, v, side="left") / len(ref)
        return out


def german():
    w = pd.read_csv(os.path.join(DATA_DIR, "german_windows.csv.gz"))
    return w.rename(columns={"phi_prox": "phi_prox_m", "phi_prox_headway": "phi_prox_h"})


def planted_highway():
    from planted_drivers import ENVIRONMENT
    frames = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", "*_s*.per_second.csv.gz"))):
        m = re.match(r"([a-z0-9]+)_([a-z]+)_s(\d+)\.per_second", os.path.basename(p))
        if int(m.group(3)) < 7 or ENVIRONMENT.get(m.group(1)) != "highway" or m.group(1) == "us101":
            continue
        ps = pd.read_csv(p)
        scen = f"{m.group(1)}_{m.group(2)}"
        frames.append(paired_windows(lambda f: windows_from_per_second(ps, scen, "highway", features=f)).assign(scenario=scen))
    return pd.concat(frames, ignore_index=True)


def ngsim_us101():
    from datasets.ngsim import per_second, read_raw, trajectories
    ps = per_second(trajectories(read_raw(NGSIM, "us-101")))
    return paired_windows(lambda f: windows_from_per_second(ps.assign(label="unlabelled"), "us101", "highway", features=f))


def evaluate(name, w, refs, labelled=True):
    rows = []
    band, lead = context(w)
    for variant, ref in refs.items():
        s = scores(w, variant)
        pct = ref.percentile(s, band, lead)
        r = {"set": name, "proximity": variant, "windows": len(w),
             "flag_fixed42": float((s >= THRESHOLDS[1]).mean()), "flag_context90": float((pct >= PCT).mean()),
             "median_score": float(np.median(s)), "median_context": float(np.median(pct))}
        if labelled:
            y = (w["behavior"] == "aggressive").to_numpy()
            nm = (w["behavior"] == "normal").to_numpy()
            keep = y | nm
            r.update({"auc_fixed": auc(s[keep], y[keep].astype(int)), "auc_context": auc(pct[keep], y[keep].astype(int)),
                      "tpr_fixed42": float((s[y] >= THRESHOLDS[1]).mean()), "fpr_fixed42": float((s[nm] >= THRESHOLDS[1]).mean()),
                      "tpr_context90": float((pct[y] >= PCT).mean()), "fpr_context90": float((pct[nm] >= PCT).mean())})
        rows.append(r)
    return rows


def main():
    g = german()
    gb, gl = context(g)
    refs = {v: Reference(scores(g, v), gb, gl) for v in ("metres", "headway", "mix")}
    print(f"reference: {len(g)} German windows, {len(refs['metres'].full)} full bins")
    uah = paired_windows(lambda f: load_windows(UAH_ROOT, features=f)[0])
    uah = uah[uah["environment"] == "highway"]
    pl = planted_highway()
    rows = evaluate("UAH motorway", uah, refs)
    rows += evaluate("planted SUMO highway (test seeds)", pl, refs)
    for sc, g2 in pl.groupby("scenario"):
        rows += evaluate(f"planted {sc}", g2, refs)
    rows += evaluate("NGSIM US-101 (unlabelled)", ngsim_us101(), refs, labelled=False)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA_DIR, "context_reference.csv"), index=False)
    with pd.option_context("display.width", 250):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
