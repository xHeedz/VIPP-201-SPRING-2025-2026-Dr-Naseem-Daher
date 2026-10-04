"""
Score the planted SUMO drivers (scripts/planted_drivers.py) against their true labels.

    python scripts/planted_eval.py

Per run (scenario, density, seed), windows of 10 s built exactly like UAH windows:
  * AUC aggressive vs normal (window level) for the reference index (gap in metres), the
    time headway variant, the UAH trained agent (data/dynamic_weight_agent.json), and each
    single term
  * AUC conservative vs normal (does the index place calm drivers below normal ones)
  * vehicle level 3 x 3 confusion matrix with the reference cut offs 35 / 70
  * median score of normal drivers (density sweep: should not climb with density)
  * with sensor noise (model/noise.py gaussian): highway medium at 1x, weather at 2x
Mean and 95% CI over seeds (t distribution). Calibration: speed and time headway of the
simulated highway against NGSIM US-101, two sample KS test.

Outputs data/planted_runs.csv, data/planted_summary.csv, data/planted_confusion.csv,
data/planted_calibration.csv, results/figures/planted_*.png
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR  # noqa: E402
from datasets.sumo_log import windows_from_per_second  # noqa: E402
from model.aggressiveness_model import headway_features, label, original_score  # noqa: E402
from model.dynamic_weight_agent import DynamicWeightAgent  # noqa: E402
from model.noise import noise_suite, sensor_view  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from planted_drivers import ENVIRONMENT, OUT_DIR  # noqa: E402
from train_uah import auc  # noqa: E402

FEATS = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]
CATS = ["conservative", "normal", "aggressive"]
T95 = {n: t for n, t in zip(range(2, 31), [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
                                           2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
                                           2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045])}
NOISE = {("highway", "medium"): 1.0, ("weather", "medium"): 2.0}


def noisy(ps, level, seed):
    suite = noise_suite("gaussian", level, seed=seed)
    parts = []
    for vid, g in ps.groupby("vehicle_id", sort=False):
        v, a, gap, lat = sensor_view(g["t"], g["speed_kmh"] / 3.6, g["accel"], g["gap_m"], g["wave_m"], suite, key=vid)
        parts.append(g.assign(speed_kmh=v * 3.6, accel=a, gap_m=gap, wave_m=lat))
    return pd.concat(parts, ignore_index=True)


def metrics(w, agent, env, prefix=""):
    y_an = w["behavior"].map({"aggressive": 1, "normal": 0})
    an = y_an.notna()
    y_cn = w["behavior"].map({"conservative": 1, "normal": 0})
    cn = y_cn.notna()
    ref = original_score(w[FEATS].to_numpy())
    out = {prefix + "auc_reference": auc(ref[an], y_an[an]),
           prefix + "auc_agent": auc(agent.ai(w.loc[an, FEATS].to_numpy(), env), y_an[an]),
           prefix + "auc_cons_vs_normal": auc(-ref[cn], y_cn[cn])}
    if not prefix:
        for f in FEATS:
            out["auc_term_" + f[4:]] = auc(w.loc[an, f], y_an[an])
    return out, ref


def run_metrics(path, agent):
    tag = os.path.basename(path).replace(".per_second.csv.gz", "")
    scenario, density, seed = tag.rsplit("_", 2)
    seed = int(seed[1:])
    env = ENVIRONMENT[scenario]
    ps = pd.read_csv(path)
    w = windows_from_per_second(ps, scenario, env)
    out = {"scenario": scenario, "density": density, "seed": seed,
           "vehicles": ps["vehicle_id"].nunique(), "windows": len(w)}
    m, ref = metrics(w, agent, env)
    out.update(m)
    wh = windows_from_per_second(ps, scenario, env, features=headway_features)
    yh = wh["behavior"].map({"aggressive": 1, "normal": 0})
    k = yh.notna()
    out["auc_headway"] = auc(original_score(wh.loc[k, FEATS].to_numpy()), yh[k])
    w = w.assign(score=ref)
    for c in CATS:
        out[f"median_score_{c}"] = float(w.loc[w["behavior"] == c, "score"].median())
    out["share_aggressive_label_normal_drivers"] = float((w.loc[w["behavior"] == "normal", "score"] >= 70).mean())
    per_vehicle = w.groupby("trip").agg(score=("score", "mean"), truth=("behavior", "first"))
    per_vehicle["pred"] = per_vehicle["score"].apply(lambda s: label(s).lower())
    conf = pd.crosstab(per_vehicle["truth"], per_vehicle["pred"]).reindex(index=CATS, columns=CATS, fill_value=0)
    if (scenario, density) in NOISE:
        lvl = NOISE[(scenario, density)]
        wn = windows_from_per_second(noisy(ps, lvl, seed), scenario, env)
        m, _ = metrics(wn, agent, env, prefix=f"noise{lvl:g}x_")
        out.update(m)
    return out, conf.assign(scenario=scenario, density=density, seed=seed)


def summarise(runs):
    cols = [c for c in runs.columns if c.startswith(("auc", "median", "share", "noise"))]
    rows = []
    for (sc, d), g in runs.groupby(["scenario", "density"], sort=False):
        row = {"scenario": sc, "density": d, "seeds": len(g), "vehicles": g["vehicles"].mean(), "windows": g["windows"].mean()}
        for c in cols:
            x = g[c].dropna()
            if len(x):
                half = T95.get(len(x), 1.96) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
                row[c] = x.mean()
                row[c + "_ci"] = half
        rows.append(row)
    return pd.DataFrame(rows)


# ── calibration against NGSIM US-101 ────────────────────────────────────────
def ks(a, b):
    """Two sample Kolmogorov Smirnov statistic and asymptotic p value."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a) - np.searchsorted(b, x, side="right") / len(b))))
    n = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    p = 2 * sum((-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def calibration():
    ng = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "ngsim_us-101_5min.csv"))
    if not os.path.exists(ng):
        return None
    from datasets.ngsim import read_raw, trajectories
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = trajectories(read_raw(ng, "us-101"))
    tr = tr[np.isclose(tr["t"] % 1.0, 0.0, atol=0.05)]
    ref_speed = tr["speed"].to_numpy() * 3.6
    m = (tr["gap"] > 0) & (tr["speed"] > 1)
    ref_thw = (tr["gap"] / tr["speed"])[m].to_numpy()
    rows, sims = [], {}
    for d in ["low", "medium", "jam"]:
        sc = "jam" if d == "jam" else "highway"
        ps = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(OUT_DIR, f"{sc}_{d}_s*.per_second.csv.gz"))])
        sp = ps["speed_kmh"].to_numpy()
        mm = (ps["gap_m"] > 0) & (ps["speed_kmh"] > 3.6)
        thw = (ps["gap_m"] / (ps["speed_kmh"] / 3.6))[mm].to_numpy()
        sims[d] = (sp, thw)
        for name, s, r in [("speed_kmh", sp, ref_speed), ("time_headway_s", thw, ref_thw)]:
            dd, p = ks(s, r)
            rows.append({"density": d, "signal": name, "sim_median": float(np.median(s)), "ngsim_median": float(np.median(r)),
                         "ks_d": dd, "p": p, "n_sim": len(s), "n_ngsim": len(r)})
    return pd.DataFrame(rows), (ref_speed, ref_thw), sims


def plots(summary, runs, cal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [("highway", "low"), ("highway", "medium"), ("jam", "jam"), ("merge", "medium"), ("urban", "medium"),
             ("roundabout", "medium"), ("weather", "medium")]
    s = summary.set_index(["scenario", "density"]).reindex(order).dropna(how="all")
    names = [f"{a}\n{b}" if a == "highway" else a for a, b in s.index]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(s))
    for i, (c, col, lab) in enumerate([("auc_reference", "#3a6ea5", "reference (gap in m)"),
                                       ("auc_headway", "#9e2a2b", "time headway variant"),
                                       ("auc_agent", "#e09f3e", "UAH trained agent")]):
        ax.bar(x + (i - 1) * 0.27, s[c], 0.27, yerr=s[c + "_ci"], color=col, label=lab, capsize=2)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUC aggressive vs normal")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.set_title("planted SUMO drivers, mean and 95% CI over seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "planted_auc.png"), dpi=140)
    plt.close(fig)

    hw = runs[runs["scenario"].isin(["highway", "jam"])]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for c, col in zip(CATS, ["#56876d", "#3a6ea5", "#9e2a2b"]):
        g = hw.groupby("density", sort=False)[f"median_score_{c}"]
        dens = ["low", "medium", "jam"]
        m, sd = g.mean().reindex(dens), g.std().reindex(dens)
        ax.errorbar(range(3), m, yerr=T95[10] * sd / np.sqrt(10), marker="o", color=col, label=c, capsize=3)
    ax.set_xticks(range(3), ["low\n1200 veh/h", "medium\n2400 veh/h", "jam\n4500 veh/h"])
    ax.set_ylabel("median window score")
    ax.set_title("density sweep, same highway, true driver types", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "planted_density_sweep.png"), dpi=140)
    plt.close(fig)

    if cal is not None:
        table, (rs, rt), sims = cal
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, k, lo, hi, xl, ref in [(axs[0], 0, 0, 160, "speed (km/h)", rs), (axs[1], 1, 0, 6, "time headway (s)", rt)]:
            ax.hist(np.clip(ref, lo, hi), bins=40, range=(lo, hi), density=True, histtype="step", color="k", lw=1.5,
                    label="NGSIM US-101")
            for d, col in zip(["low", "medium", "jam"], ["#56876d", "#3a6ea5", "#9e2a2b"]):
                ax.hist(np.clip(sims[d][k], lo, hi), bins=40, range=(lo, hi), density=True, histtype="step", color=col,
                        label=f"sim {d}")
            ax.set_xlabel(xl)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "planted_calibration.png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    agent = DynamicWeightAgent.load(os.path.join(DATA_DIR, "dynamic_weight_agent.json"))
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.per_second.csv.gz")))
    res, confs = [], []
    for i, f in enumerate(files):
        r, c = run_metrics(f, agent)
        res.append(r)
        confs.append(c)
        print(f"  {i + 1}/{len(files)} {os.path.basename(f)}", flush=True)
    runs = pd.DataFrame(res)
    runs.to_csv(os.path.join(DATA_DIR, "planted_runs.csv"), index=False)
    pd.concat(confs).to_csv(os.path.join(DATA_DIR, "planted_confusion.csv"))
    summary = summarise(runs)
    summary.to_csv(os.path.join(DATA_DIR, "planted_summary.csv"), index=False)
    cal = calibration()
    if cal is not None:
        cal[0].to_csv(os.path.join(DATA_DIR, "planted_calibration.csv"), index=False)
    plots(summary, runs, cal)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(summary.round(3).to_string(index=False))
        conf = pd.concat(confs).groupby(level=0).sum(numeric_only=True).drop(columns="seed").reindex(CATS)
        print("\nvehicle level confusion, all runs (rows truth, columns label from 35 / 70):")
        print(conf.to_string())
        if cal is not None:
            print(cal[0].round(3).to_string(index=False))
