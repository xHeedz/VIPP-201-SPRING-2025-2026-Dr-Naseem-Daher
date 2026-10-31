"""
Label audit on UAH-DriveSet: where the formula and the trip label disagree, which one is right?

    python scripts/label_audit_uah.py [--per-group 8] [--controls 3] [--seed 0] [--out ../26-10-2026/label_audit]

1. every 10 s window (data/clean/windows_uah.csv.gz, normal and aggressive trips) gets the reference formula score
   (aggressive from the cut off 27.5) and DriveSafe's own estimate for the same 10 s (SEMANTIC_ONLINE column 14,
   "ratio aggressive" of the last minute, aggressive from 0.5): a third opinion that never saw the formula
2. four groups: label aggressive / formula calm, label normal / formula aggressive, and the two agreeing groups
3. how often DriveSafe sides with the formula against the label, per group and per driver
4. random sample per group (fixed seed): a figure per window (three video frames; speed with the OpenStreetMap
   speed limit, acceleration, gap and UAH's time to collision, lane offset, per second score; 60 s of context with
   the window shaded) and the 10 s video clip (video time = data time + data start - video start, as in the UAH
   reader tool)
5. review.csv for the manual verdict and index.html showing every sampled window with its clip
Outputs in --out; summary numbers also in data/label_audit_summary.csv.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.uah import find_trips  # noqa: E402
from datasets.windows import TERM_COLS, per_second_terms  # noqa: E402
from model.aggressiveness_model import REF_CUTOFFS, reference_score  # noqa: E402

ROOT = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "UAH-DRIVESET-v1"))
GROUPS = {"label aggressive, formula calm": ("aggressive", False), "label normal, formula aggressive": ("normal", True),
          "agree: aggressive": ("aggressive", True), "agree: normal": ("normal", False)}


def read(path):
    return pd.read_csv(path, sep=r"\s+", header=None) if os.path.exists(path) else None


def trip_info():
    out = {}
    for t in find_trips(ROOT):
        vids = glob.glob(os.path.join(t["path"], "*.mp4"))
        vstart = os.path.basename(vids[0])[:14] if vids else None
        delay = ((datetime.strptime(t["trip"][:14], "%Y%m%d%H%M%S") - datetime.strptime(vstart, "%Y%m%d%H%M%S")).total_seconds()
                 if vstart else None)
        sem, veh, osm = (read(os.path.join(t["path"], f)) for f in
                         ("SEMANTIC_ONLINE.txt", "PROC_VEHICLE_DETECTION.txt", "PROC_OPENSTREETMAP_DATA.txt"))
        out[t["trip"]] = {**t, "video": vids[0] if vids else None, "delay": delay, "sem": sem, "veh": veh, "osm": osm}
    return out


def drivesafe(win, info):
    r = np.full(len(win), np.nan)
    for trip, ix in win.groupby("vid").indices.items():
        sem = info[trip]["sem"]
        t, v = sem[0].to_numpy(), sem[13].to_numpy()
        for i in ix:
            m = (t >= win["t0"].iloc[i]) & (t < win["t0"].iloc[i] + 10)
            r[i] = v[m].mean() if m.any() else np.nan
    return r


def clip(video, start, duration, out):
    subprocess.run(["avconvert", "--source", video, "--output", out, "--preset", "PresetLowQuality", "--start",
                    f"{max(start, 0):.2f}", "--duration", f"{duration:.2f}", "--replace"], capture_output=True)
    return os.path.exists(out)


def frame(video, t, tmp):
    """one still at video second t (QuickLook thumbnail of a 1 s clip starting there)"""
    c = os.path.join(tmp, f"f_{t:.1f}.mp4")
    if not clip(video, t, 1.0, c):
        return None
    subprocess.run(["qlmanage", "-t", "-s", "480", "-o", tmp, c], capture_output=True)
    p = c + ".png"
    return p if os.path.exists(p) else None


def figure(row, ps, info, frames, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = row["t0"]
    lo, hi = t0 - 25, t0 + 35
    s = ps[(ps["t"] >= lo) & (ps["t"] <= hi)]
    fig = plt.figure(figsize=(12, 9), dpi=90)
    gs = fig.add_gridspec(6, 3, height_ratios=[1.6, 1, 1, 1, 1, 1], hspace=0.55)
    for k, fr in enumerate(frames):
        ax = fig.add_subplot(gs[0, k])
        ax.axis("off")
        if fr:
            ax.imshow(plt.imread(fr))
        ax.set_title(f"video at t = {t0 + 5 * k:.0f} s", fontsize=9)
    veh, osm = info["veh"], info["osm"]
    rows = [("speed (km/h)", s["t"], s["speed_kmh"], "#7A1F2B"), ("acceleration (m/s²)", s["t"], s["accel"], "#4A6FA5"),
            ("gap (m)", s["t"], s["gap_m"], "#D9A441"), ("lane offset (m)", s["t"], s["wave_m"], "#8E8385"),
            ("formula score per second", s["t"], s["score_s"], "#231A1C")]
    for k, (lab, x, y, c) in enumerate(rows):
        ax = fig.add_subplot(gs[k + 1, :])
        ax.plot(x, y, color=c, lw=1.8)
        ax.axvspan(t0, t0 + 10, color="#E8C9CE", alpha=0.6, lw=0)
        ax.set_xlim(lo, hi)
        ax.set_ylabel(lab, fontsize=9)
        ax.tick_params(labelsize=8)
        if k == 0 and osm is not None:
            m = (osm[0] >= lo) & (osm[0] <= hi) & (osm[1] > 0)
            ax.plot(osm.loc[m, 0], osm.loc[m, 1], color="#8E8385", ls="--", lw=1.2, label="speed limit (OpenStreetMap)")
            ax.legend(fontsize=8, loc="lower left")
        if k == 2 and veh is not None:
            m = (veh[0] >= lo) & (veh[0] <= hi) & (veh[2] > 0)
            ax2 = ax.twinx()
            ax2.scatter(veh.loc[m, 0], veh.loc[m, 2].clip(upper=20), s=6, color="#7A1F2B")
            ax2.set_ylabel("TTC (s)", fontsize=8, color="#7A1F2B")
            ax2.tick_params(labelsize=7)
        if k == 4:
            for cut, ls in zip(REF_CUTOFFS, (":", "--")):
                ax.axhline(cut, color="#8E8385", ls=ls, lw=1)
            ax.set_xlabel("trip time (s); shaded: the 10 s window", fontsize=9)
    fig.suptitle(f"{row['vid']}  t0 {t0:.0f} s | label {row['label']} | formula {row['score']:.1f} "
                 f"({'aggressive' if row['score'] >= REF_CUTOFFS[1] else 'not aggressive'}) | "
                 f"DriveSafe {row['drivesafe']:.2f}", fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(per_group, controls, seed, out):
    os.makedirs(os.path.join(out, "clips"), exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    info = trip_info()
    w = pd.read_csv(os.path.join(DATA_DIR, "clean", "windows_uah.csv.gz"))
    w = w[w["label"].isin(["normal", "aggressive"])].reset_index(drop=True)
    w["score"] = reference_score(w[TERM_COLS].to_numpy())
    w["formula_aggr"] = w["score"] >= REF_CUTOFFS[1]
    w["drivesafe"] = drivesafe(w, info)
    w["group"] = [next(g for g, (lab, fa) in GROUPS.items() if lab == l and fa == f) for l, f in zip(w["label"], w["formula_aggr"])]

    rows = []
    for g, d in w.groupby("group"):
        ds = d["drivesafe"] >= 0.5
        rows.append({"group": g, "windows": len(d), "share_of_label": len(d) / (w["label"] == d["label"].iloc[0]).sum(),
                     "drivesafe_says_aggressive": float(ds.mean()),
                     "drivesafe_sides_with_formula": float((ds == d["formula_aggr"]).mean())})
    summ = pd.DataFrame(rows)
    per_driver = w[w["label"] != w["formula_aggr"].map({True: "aggressive", False: "normal"})].groupby("label")
    summ.to_csv(os.path.join(DATA_DIR, "label_audit_summary.csv"), index=False)
    print(summ.round(3).to_string(index=False))
    dis = w[(w["label"] == "aggressive") != w["formula_aggr"]]
    print("disagreements per driver:", dis.groupby("group")["vid"].count().to_dict(),
          dis.assign(drv=dis["vid"].str.extract(r"-(D\d)-")[0]).groupby("drv").size().to_dict())

    rng = np.random.default_rng(seed)
    ps_all = per_second_terms(pd.read_csv(os.path.join(DATA_DIR, "clean", "uah.csv.gz")))
    ps_all["score_s"] = reference_score(ps_all[TERM_COLS].to_numpy())
    review, html = [], ["<!doctype html><meta charset='utf-8'><title>UAH label audit</title>",
                        "<style>body{font-family:Helvetica,Arial;max-width:1100px;margin:24px auto;color:#231a1c}"
                        "h2{color:#7a1f2b}img,video{max-width:100%}.w{border-bottom:1px solid #ddd;padding:12px 0}</style>",
                        "<h1>UAH label audit</h1><p>Each window: trip label, formula score (aggressive from "
                        f"{REF_CUTOFFS[1]}), DriveSafe ratio (aggressive from 0.5). Watch the clip, then fill review.csv.</p>"]
    n_id = 0
    for g in GROUPS:
        d = w[w["group"] == g]
        k = per_group if "agree" not in g else controls
        pick = d.iloc[rng.permutation(len(d))[:k]]
        html.append(f"<h2>{g} ({len(d)} windows, {k} sampled)</h2>")
        for _, r in pick.iterrows():
            n_id += 1
            sid = f"{n_id:02d}"
            tr = info[r["vid"]]
            ps = ps_all[ps_all["vid"] == r["vid"]]
            frames, clip_rel = [None, None, None], ""
            if tr["video"]:
                with tempfile.TemporaryDirectory() as tmp:
                    frames = [frame(tr["video"], r["t0"] + tr["delay"] + 5 * j, tmp) for j in range(3)]
                    figure(r, ps, tr, frames, os.path.join(out, "figures", f"{sid}.png"))
                clip_rel = f"clips/{sid}.mp4"
                clip(tr["video"], r["t0"] + tr["delay"], 10.0, os.path.join(out, clip_rel))
            else:
                figure(r, ps, tr, frames, os.path.join(out, "figures", f"{sid}.png"))
            review.append({"id": sid, "group": g, "trip": r["vid"], "t0": round(r["t0"], 1), "label": r["label"],
                           "formula_score": round(r["score"], 1), "formula_says": "aggressive" if r["formula_aggr"] else "not aggressive",
                           "drivesafe_ratio": round(r["drivesafe"], 2), "clip": clip_rel, "figure": f"figures/{sid}.png",
                           "your_verdict (normal / aggressive / unclear)": "", "what you saw": ""})
            html.append(f"<div class='w'><b>#{sid}</b> {r['vid']} t0 {r['t0']:.0f} s: label <b>{r['label']}</b>, formula "
                        f"<b>{r['score']:.1f}</b>, DriveSafe <b>{r['drivesafe']:.2f}</b><br>"
                        f"<video controls src='{clip_rel}' width='560'></video><br><img src='figures/{sid}.png'></div>")
            print(f"  #{sid} {g}: {r['vid']} t0 {r['t0']:.0f}", flush=True)
    pd.DataFrame(review).to_csv(os.path.join(out, "review.csv"), index=False)
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write("\n".join(html))
    print("written", out)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--per-group", type=int, default=8)
    p.add_argument("--controls", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.abspath(os.path.join(DATA_DIR, "..", "..", "26-10-2026", "label_audit")))
    a = p.parse_args()
    main(a.per_group, a.controls, a.seed, a.out)
