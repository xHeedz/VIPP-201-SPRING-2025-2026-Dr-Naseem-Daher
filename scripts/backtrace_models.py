"""
Backtrack what each model computes, from the raw per second values to its output, on real windows.

    python scripts/backtrace_models.py

1. formula: per chosen window (UAH aggressive and normal, planted conservative / normal / aggressive, highD,
   NGSIM US-101, pNEUMA): per second raw values, terms, mean over the window, weights, points, score, label.
2. supervised: learned weights in every leave one driver out fold (data/rerun_supervised.csv) and the AUC of each
   single term per held out driver, which shows what the learned weights can and cannot rely on.
3. inverse RL: one planted car: its decisions (chosen acceleration), the probability the fitted model gives each
   choice, the share of "keep speed" decisions, and its fitted preferred gap against its type's pooled value.
Writes docs/hand_checks/backtrace_formula.md, backtrace_supervised.md, backtrace_irl.md and data/backtrace_*.csv.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402
from datasets.windows import TERM_COLS, per_second_terms  # noqa: E402
from model.aggressiveness_model import PROX_MIX, REF_CUTOFFS, REF_WEIGHTS, reference_label  # noqa: E402
from rerun_formula_all import auc  # noqa: E402

CLEAN = os.path.join(DATA_DIR, "clean")
HC = os.path.join(os.path.dirname(DATA_DIR), "docs", "hand_checks")
NAMES = ["speed", "accel2", "prox2", "wave"]
W = np.asarray(REF_WEIGHTS)


def fmt(x, nd=2):
    return "unknown" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def window_points(row):
    t = np.array([row[c] for c in TERM_COLS], float)
    k = ~np.isnan(t)
    pts = np.where(k, 100 * W * np.where(k, t, 0) / (W[k].sum()), np.nan)
    return t, k, pts


def pick(w, label, q=0.5):
    g = w[w["label"] == label].copy()
    T = g[TERM_COLS].to_numpy()
    k = ~np.isnan(T)
    g["score"] = 100 * (np.where(k, T, 0) * W).sum(1) / (k * W).sum(1)
    g = g[g["seconds"] >= 10]
    return g.iloc[(g["score"] - g["score"].quantile(q)).abs().argsort().iloc[0]]


def formula_section(lines, source, label, title, q=0.5):
    w = pd.read_csv(os.path.join(CLEAN, f"windows_{source}.csv.gz"))
    row = pick(w, label, q)
    d = pd.read_csv(os.path.join(CLEAN, f"{source}.csv.gz"))
    s = d[(d["vid"].astype(str) == str(row["vid"])) & (d["t"] >= row["t0"]) & (d["t"] < row["t0"] + 10)]
    s = per_second_terms(s)
    env = row["environment"]
    lines += [f"## {title}", "", f"window: `{row['vid']}`, t0 {row['t0']:.0f} s, environment {env} "
              f"(gap mix: {PROX_MIX.get(env, 0.5):.1f} metres, {1 - PROX_MIX.get(env, 0.5):.1f} seconds), "
              f"true label {label}", "",
              "| second | speed km/h | accel m/s2 | gap m | car ahead | lane offset m | speed term | accel² term | gap term | lane term |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lead = {1.0: "yes", 0.0: "no"}.get(r["leader"], "unknown")
        lines.append(f"| {r['t']:.0f} | {r['speed_kmh']:.1f} | {r['accel']:.2f} | {fmt(r['gap_m'], 1)} | {lead} | "
                     f"{fmt(r['wave_m'])} | {r['t_speed']:.3f} | {r['t_accel2']:.4f} | {fmt(r['t_prox2'], 3)} | {fmt(r['t_wave'], 3)} |")
    t, k, pts = window_points(row)
    lines += ["", "| term | mean over the window | weight | points (weight x term / known weights x 100) |", "|---|---|---|---|"]
    for i, n in enumerate(NAMES):
        lines.append(f"| {n} | {fmt(t[i], 4)} | {W[i]:.2f} | {fmt(pts[i], 1)} |")
    score = float(np.nansum(pts))
    lines += ["", f"known weights add up to {W[k].sum():.2f}. score = {score:.1f}, label "
              f"**{reference_label(score)}** (cut offs {REF_CUTOFFS[0]:.1f} / {REF_CUTOFFS[1]:.1f}). "
              f"largest part: {NAMES[int(np.nanargmax(pts))]} ({np.nanmax(pts):.1f} of {score:.1f} points).", ""]
    return {"source": source, "label": label, "vid": row["vid"], "score": score, **{f"pts_{n}": pts[i] for i, n in enumerate(NAMES)}}


def supervised_section():
    r = pd.read_csv(os.path.join(DATA_DIR, "rerun_supervised.csv"))
    lodo = r[r["test"] == "uah lodo"]
    w = pd.read_csv(os.path.join(CLEAN, "windows_uah.csv.gz"))
    w = w[w["label"].isin(["normal", "aggressive"])]
    rows = []
    for drv, g in w.groupby("group"):
        y = g["label"] == "aggressive"
        rows.append({"driver": drv, **{n: auc(g.loc[y, c], g.loc[~y, c]) for n, c in zip(NAMES, TERM_COLS)}})
    single = pd.DataFrame(rows).set_index("driver")
    single.to_csv(os.path.join(DATA_DIR, "backtrace_single_term_auc.csv"))
    lines = ["# backtrace: supervised model on UAH", "",
             "the model: score = 100 x (w1 speed + w2 accel² + w3 gap + w4 lane), weights learned per environment "
             "(add up to 1), P(aggressive) = sigmoid((score - threshold) / tau), trained on 5 drivers, tested on the 6th.", "",
             "## learned weights per fold (speed / accel² / gap / lane) and result on the held out driver", "",
             "| held out | highway and urban weights | AUC supervised | AUC formula |", "|---|---|---|---|"]
    for _, x in lodo.iterrows():
        lines.append(f"| {x['held_out']} | {x['weights']} | {x['supervised_auc']:.3f} | {x['formula_auc']:.3f} |")
    lines += ["", "## how well each single term separates aggressive from normal, per driver (AUC)", "",
              "| driver | speed | accel² | gap | lane |", "|---|---|---|---|---|"]
    for drv, x in single.iterrows():
        lines.append(f"| {drv} | " + " | ".join(fmt(x[n], 3) for n in NAMES) + " |")
    lines += ["", f"mean over drivers: " + ", ".join(f"{n} {single[n].mean():.3f}" for n in NAMES) + "; spread (sd) "
              + ", ".join(f"{n} {single[n].std():.3f}" for n in NAMES), ""]
    with open(os.path.join(HC, "backtrace_supervised.md"), "w") as f:
        f.write("\n".join(lines))
    return single


def irl_section():
    from model.irl import A_GRID, decisions, fit
    v = pd.read_csv(os.path.join(DATA_DIR, "rerun_irl_vehicles.csv"))
    v = v[(v["variant"] == "all seconds") & (v["label"] == "aggressive") & (v["seed"].isin([7, 8, 9]))]
    car = v.iloc[(v["decisions"] - v["decisions"].median()).abs().argsort().iloc[0]]
    d = pd.read_csv(os.path.join(CLEAN, "planted.csv.gz"))
    g = d[d["vid"] == car["vid"]].sort_values("t")
    g = g[g["speed_kmh"] >= 5].assign(gap_m=lambda x: x["gap_m"].fillna(0.0))
    f, y = decisions(g)
    th = np.array([car[f"theta_{n}"] for n in ("progress", "progress_sq", "headway", "headway_sq", "discomfort")])
    logits = f @ th
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    chosen = A_GRID[y]
    lines = ["# backtrace: inverse RL on one planted aggressive car", "",
             f"car `{car['vid']}` ({car['scenario']}), {len(y)} decisions. each second the car picks one of 12 "
             "accelerations; the model gives each a probability from theta x features (speed reached, gap in seconds over "
             "the next 2 s, discomfort).", "",
             f"- decisions at 'keep speed' (a = 0): {np.mean(chosen == 0):.0%}; within 0.5 m/s2: {np.mean(np.abs(chosen) <= 0.5):.0%}",
             f"- mean probability the fitted model gives the choice actually made: {p[np.arange(len(y)), y].mean():.2f} "
             f"(12 equal choices would give 0.08)",
             f"- fitted preferred gap {car['desired_headway_s']:.2f} s; aggressive cars pooled: see data/rerun_irl_summary.csv", "",
             "| second | speed km/h | gap m | chosen accel | P(chosen) | P(keep speed) | most likely choice |", "|---|---|---|---|---|---|---|"]
    i0 = int(np.flatnonzero(A_GRID == 0)[0])
    for i in range(min(len(y), 25)):
        lines.append(f"| {i} | {g['speed_kmh'].iloc[i]:.1f} | {g['gap_m'].iloc[i]:.1f} | {chosen[i]:+.1f} | {p[i, y[i]]:.2f} | "
                     f"{p[i, i0]:.2f} | {A_GRID[int(p[i].argmax())]:+.1f} |")
    lines.append("")
    with open(os.path.join(HC, "backtrace_irl.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    lines = ["# backtrace: reference formula, from raw values to the label", "",
             f"formula: score = 100 x (sum of weight x term over the known terms) / (sum of the known weights). weights "
             f"speed {W[0]:.2f}, accel² {W[1]:.2f}, gap {W[2]:.2f}, lane {W[3]:.2f}. terms: speed = km/h / 150; accel² = "
             "(|a| / 5)²; gap = mix of (1 - metres / 50)² and (1 - headway / 3 s)²; lane = offset / 1.5 m. a window term "
             "is the mean of its per second terms.", ""]
    rows = []
    for src, lab, title in [("uah", "aggressive", "UAH, aggressive trip (median window)"),
                            ("uah", "normal", "UAH, normal trip (median window)"),
                            ("planted", "aggressive", "simulated aggressive car"), ("planted", "normal", "simulated normal car"),
                            ("planted", "conservative", "simulated conservative car"),
                            ("highd", "unlabelled", "highD, German motorway (median window)"),
                            ("ngsim_us_101", "unlabelled", "NGSIM US-101 (median window)"),
                            ("pneuma", "unlabelled", "pNEUMA, Athens (median window)")]:
        rows.append(formula_section(lines, src, lab, title))
        print("formula traced:", title, flush=True)
    with open(os.path.join(HC, "backtrace_formula.md"), "w") as f:
        f.write("\n".join(lines))
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "backtrace_formula.csv"), index=False)
    single = supervised_section()
    print(single.round(3).to_string(), flush=True)
    irl_section()
    print("written docs/hand_checks/backtrace_*.md")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
