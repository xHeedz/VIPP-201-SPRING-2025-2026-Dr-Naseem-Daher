"""
Full DriveDNA release (465 drivers, 4,121 drives) through the same pipeline as every other source: human spans ->
per second table -> cleaning (datasets/cleaning.py) -> 10 s windows -> reference formula. Meant for the workstation.

    python scripts/drivedna_full.py --root /path/to/DriveDNA [--out results/drivedna_full] [--workers 16]

Needs pandas with pyarrow (parquet). Per drive: raw_signals/<driver>/<drive>.parquet sliced with
index/human_segments.parquet (rows i0 to i1 of each human span). One output file per drive
(<out>/windows/<driver>__<drive>.csv.gz), so a stopped run continues where it stopped. At the end:
<out>/drivedna_windows.csv.gz (all windows) and <out>/drivedna_drivers.csv (per driver: windows, drives, car models,
median score, shares per label, mean points per term, car ahead share, spread of the score across drives).
"""
import argparse
import glob
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.cleaning import clean  # noqa: E402
from datasets.drivedna import per_second_frame  # noqa: E402
from datasets.windows import TERM_COLS, windows  # noqa: E402
from model.aggressiveness_model import REF_CUTOFFS, REF_WEIGHTS  # noqa: E402

COLS = ["time_s", "vEgo", "leadOne_status", "leadOne_dRel", "laneLeft_y", "laneRight_y"]


def one_drive(args):
    root, out, driver, drive, spans, model = args
    path = os.path.join(out, "windows", f"{driver}__{drive}.csv.gz")
    if os.path.exists(path):
        return path
    try:
        sig = pd.read_parquet(os.path.join(root, "raw_signals", driver, f"{drive}.parquet"))
        cols = [c for c in COLS if c in sig.columns]
        parts = []
        for k, (i0, i1) in enumerate(spans):
            seg = sig.iloc[int(i0):int(i1)][cols]
            ps = per_second_frame(seg, human_only=False)
            if len(ps) >= 10:
                parts.append(ps.assign(vid=f"{driver}/{drive}/{k}"))
        if not parts:
            pd.DataFrame().to_csv(path, index=False)
            return path
        ps = pd.concat(parts, ignore_index=True)
        ps, _ = clean(ps, "drivedna")
        ps = ps.assign(source="drivedna", group=driver, label="unlabelled", environment="highway", scenario=model)
        w = windows(ps)
        w.assign(driver=driver, drive=drive, car_model=model).to_csv(path, index=False)
    except Exception as e:  # noqa: BLE001
        print(f"  {driver}/{drive}: {e}", flush=True)
        pd.DataFrame().to_csv(path, index=False)
    return path


def summarise(out):
    files = sorted(glob.glob(os.path.join(out, "windows", "*.csv.gz")))
    w = pd.concat([pd.read_csv(f) for f in files if os.path.getsize(f) > 30], ignore_index=True)
    T = w[TERM_COLS].to_numpy()
    k = ~np.isnan(T)
    wt = np.asarray(REF_WEIGHTS)
    den = (k * wt).sum(axis=1, keepdims=True)
    P = 100 * np.where(k, T, 0) * wt / np.where(den > 0, den, np.nan)
    for i, n in enumerate(["speed", "accel2", "prox2", "wave"]):
        w[f"pts_{n}"] = P[:, i]
    w["score"] = P.sum(axis=1)
    lo, hi = REF_CUTOFFS
    w.to_csv(os.path.join(out, "drivedna_windows.csv.gz"), index=False)
    per_drive = w.groupby(["driver", "drive"])["score"].median().rename("drive_median").reset_index()
    g = w.groupby("driver")
    d = pd.DataFrame({"windows": g.size(), "drives": g["drive"].nunique(), "car_models": g["car_model"].nunique(),
                      "median_score": g["score"].median(),
                      "share_conservative": g["score"].apply(lambda s: (s < lo).mean()),
                      "share_aggressive": g["score"].apply(lambda s: (s >= hi).mean()),
                      "leader_share": g["leader_share"].mean(),
                      **{f"pts_{n}": g[f"pts_{n}"].mean() for n in ["speed", "accel2", "prox2", "wave"]},
                      "spread_across_drives": per_drive.groupby("driver")["drive_median"].std()})
    d.to_csv(os.path.join(out, "drivedna_drivers.csv"))
    print(f"{len(w)} windows, {w['driver'].nunique()} drivers; median score {w['score'].median():.1f}, "
          f"aggressive {(w['score'] >= hi).mean():.1%}, conservative {(w['score'] < lo).mean():.1%}")


def main(root, out, workers, limit):
    os.makedirs(os.path.join(out, "windows"), exist_ok=True)
    seg = pd.read_parquet(os.path.join(root, "index", "human_segments.parquet"))
    drives = pd.read_parquet(os.path.join(root, "index", "drives.parquet"))
    model_col = next((c for c in ("car_model", "vehicle_model", "model") if c in drives.columns), None)
    models = drives.set_index(["driver", "drive"])[model_col].to_dict() if model_col else {}
    jobs = [(root, out, drv, dr, list(zip(g["i0"], g["i1"])), models.get((drv, dr), ""))
            for (drv, dr), g in seg.groupby(["driver", "drive"])]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} drives, {workers} workers", flush=True)
    t = time.time()
    with Pool(workers) as p:
        for i, _ in enumerate(p.imap_unordered(one_drive, jobs, chunksize=4), 1):
            if i % 100 == 0:
                print(f"  {i} / {len(jobs)} drives, {time.time() - t:.0f} s", flush=True)
    summarise(out)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="results/drivedna_full")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--limit", type=int, default=0, help="first N drives only (quick test)")
    a = p.parse_args()
    main(a.root, a.out, a.workers, a.limit)
