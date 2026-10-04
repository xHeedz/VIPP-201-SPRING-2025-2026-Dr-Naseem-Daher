"""
Sanity checks on the raw index inputs of every data source, plus feature histograms
side by side and the per feature contribution of the top 5% scores.

    python scripts/sanity_checks.py

Checks per source (rows = SUMO samples, UAH seconds, NGSIM 10 Hz rows):
    |accel| above 8 m/s2, wave above 2 m (NGSIM: outside +-3 s of a lane change),
    share of rows exactly at a clip limit (score 100, |accel| >= 5, wave >= 1.5), NaN rows.
Outputs data/sanity_checks.csv, results/figures/feature_check_*.png,
results/figures/top5_contributions.png and data/top5_contributions.csv.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR  # noqa: E402
from model.aggressiveness_model import WEIGHTS, index_features, original_score  # noqa: E402

EXT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "28:9:2026"))
FEATS = ["speed_kmh", "accel", "gap_m", "wave_m"]


def sources():
    """{name: DataFrame with speed_kmh, accel, gap_m, wave_m (+ near_lane_change)}"""
    out = {}
    for name, f in [("sumo urban", "sumo_npc_aggressiveness.csv"),
                    ("sumo highway", "sumo_highway_npc_aggressiveness.csv"),
                    ("sumo urban, before fix", "sumo_npc_aggressiveness_before_fix.csv"),
                    ("sumo highway, before fix", "sumo_highway_npc_aggressiveness_before_fix.csv")]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            out[name] = pd.DataFrame({"speed_kmh": d["speed_kmh"], "accel": d["accel_ms2"],
                                      "gap_m": d["prox_m"], "wave_m": d["wave_m"]})
    p = os.path.join(DATA_DIR, "driving_behaviors_dataset.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        out["legacy highway-env"] = pd.DataFrame({"speed_kmh": d["Speed_kmh"], "accel": d["Accel_ms2"],
                                                  "gap_m": d["Proximity_m"], "wave_m": d["Waviness_m"]})
    uah = os.path.join(EXT, "UAH-DRIVESET-v1")
    if os.path.isdir(uah):
        from datasets.uah import find_trips, load_trip
        out["uah"] = pd.concat([load_trip(t)[FEATS] for t in find_trips(uah)], ignore_index=True)
    ng = os.path.join(EXT, "ngsim_us-101_5min.csv")
    if os.path.exists(ng):
        from datasets.ngsim import read_raw, trajectories
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr = trajectories(read_raw(ng, "us-101"))
        # rows within 3 s of a lane change (Lane_ID differs from the row 3 s before or after)
        g = tr.groupby("vehicle_id")["lane"]
        lc = (g.shift(30) != tr["lane"]) & g.shift(30).notna() | (g.shift(-30) != tr["lane"]) & g.shift(-30).notna()
        out["ngsim us-101"] = pd.DataFrame({"speed_kmh": tr["speed"] * 3.6, "accel": tr["accel_1hz"],
                                            "gap_m": tr["gap"], "wave_m": tr["wave"], "near_lane_change": lc})
    return out


def checks(name, d):
    f = index_features(d["speed_kmh"], d["accel"], d["gap_m"], d["wave_m"])
    s = original_score(f)
    calm = ~d["near_lane_change"] if "near_lane_change" in d else pd.Series(True, index=d.index)
    return {"source": name, "rows": len(d),
            "nan_rows": int(d[FEATS].isna().any(axis=1).sum()),
            "accel_above_8": float((d["accel"].abs() > 8).mean()),
            "wave_above_2": float((d["wave_m"][calm] > 2).mean()),
            "at_score_100": float((s >= 100).mean()),
            "at_accel_5": float((d["accel"].abs() >= 5).mean()),
            "at_wave_1_5": float((d["wave_m"].abs() >= 1.5).mean()),
            "leader_share": float((d["gap_m"] > 0).mean()),
            "score_median": float(np.nanmedian(s)), "score_p95": float(np.nanpercentile(s, 95))}


def histograms(src):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in src if "before" not in n]
    spec = {"speed_kmh": (0, 160, "speed (km/h)"), "accel": (-10, 10, "acceleration (m/s2)"),
            "gap_m": (0.1, 100, "gap to leader (m), rows with a leader"), "wave_m": (0, 2.5, "offset from lane centre (m)"),
            "score": (0, 100, "aggressiveness index")}
    for feat, (lo, hi, xl) in spec.items():
        fig, axs = plt.subplots(1, len(names), figsize=(3.2 * len(names), 2.8), sharex=True)
        for ax, n in zip(np.atleast_1d(axs), names):
            d = src[n]
            x = original_score(index_features(*[d[c] for c in FEATS])) if feat == "score" else d[feat]
            x = np.asarray(x, dtype=float)
            if feat == "gap_m":
                x = x[x > 0]
            x = x[np.isfinite(x)]
            ax.hist(np.clip(x, lo, hi), bins=40, range=(lo, hi), color="#3a6ea5")
            ax.set_title(f"{n} (n={len(x):,})", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.supxlabel(xl, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"feature_check_{feat}.png"), dpi=130)
        plt.close(fig)


def top5(src):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for n, d in src.items():
        if "before" in n:
            continue
        f = index_features(*[d[c] for c in FEATS])
        s = original_score(f)
        c = 100 * f * np.array(WEIGHTS)
        ok = np.isfinite(s)
        top = ok & (s >= np.nanpercentile(s, 95))
        for part, m in [("all rows", ok), ("top 5%", top)]:
            rows.append({"source": n, "rows": part, "score": float(s[m].mean()),
                         **{k: float(c[m, i].mean()) for i, k in enumerate(["speed", "accel", "prox", "wave"])}})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(DATA_DIR, "top5_contributions.csv"), index=False)
    tt = t[t["rows"] == "top 5%"].set_index("source")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    left = np.zeros(len(tt))
    for k, col in zip(["speed", "accel", "prox", "wave"], ["#3a6ea5", "#e09f3e", "#9e2a2b", "#56876d"]):
        ax.barh(tt.index, tt[k], left=left, color=col, label=k)
        left += tt[k].to_numpy()
    ax.set_xlabel("mean points in the top 5% of scores, per term")
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "top5_contributions.png"), dpi=140)
    plt.close(fig)
    return t


if __name__ == "__main__":
    src = sources()
    res = pd.DataFrame([checks(n, d) for n, d in src.items()])
    res.to_csv(os.path.join(DATA_DIR, "sanity_checks.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
        histograms(src)
        print()
        print(top5(src).round(2).to_string(index=False))
