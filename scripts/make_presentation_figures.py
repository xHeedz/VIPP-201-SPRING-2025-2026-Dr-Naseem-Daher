"""
Figures for the long presentation (week of 26 Oct 2026), one style, every number read from data/.

    python scripts/make_presentation_figures.py
writes results/figures/pres_*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR  # noqa: E402

MAROON, GOLD, BLUE, GREY, INK = "#7A1F2B", "#D9A441", "#4A6FA5", "#B9AEB0", "#231A1C"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8E8385", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.dpi": 150})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"pres_{name}.png"), transparent=False, facecolor="white")
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.0%}", dy=0.01):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=10)


def audit():
    s = pd.read_csv(os.path.join(DATA_DIR, "label_audit_summary.csv")).set_index("group")
    order = ["agree: normal", "label normal, formula aggressive", "agree: aggressive", "label aggressive, formula calm"]
    names = ["label normal\nformula normal", "label normal\nformula aggressive", "label aggressive\nformula aggressive",
             "label aggressive\nformula calm"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    b = a1.bar(names, [s.loc[g, "windows"] for g in order], color=[GREY, MAROON, GREY, MAROON])
    label_bars(a1, b, "{:.0f}", 15)
    a1.set_title("UAH 10 s windows")
    a1.tick_params(axis="x", labelsize=9)
    b = a2.bar(names, [s.loc[g, "drivesafe_says_aggressive"] for g in order], color=[GREY, MAROON, GREY, MAROON])
    label_bars(a2, b)
    a2.set_ylim(0, 1)
    a2.set_title("UAH's own software says aggressive")
    a2.tick_params(axis="x", labelsize=9)
    save(fig, "audit")


def single_terms():
    s = pd.read_csv(os.path.join(DATA_DIR, "extended_single_terms.csv")).set_index("test")
    cols = ["speed", "accel2", "gap2", "lane", "ttc", "rel_limit", "rel_traffic", "idm"]
    names = ["speed", "accel²", "gap²", "lane", "TTC", "speed vs\nlimit", "speed vs\ntraffic", "IDM\nresidual"]
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for k, (row, col, lab) in enumerate([("UAH A vs N", BLUE, "UAH (real)"), ("planted test A vs N", MAROON, "SUMO (simulated)")]):
        b = ax.bar(x + (k - 0.5) * 0.38, [s.loc[row, c] for c in cols], 0.38, color=col, label=lab)
        label_bars(ax, b, "{:.2f}", 0.005)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.text(len(cols) - 0.5, 0.505, "coin flip", color="#8E8385", fontsize=9, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("AUC, aggressive vs normal")
    ax.legend(frameon=False)
    save(fig, "single_terms")


def formulas():
    e = pd.read_csv(os.path.join(DATA_DIR, "extended_fit.csv"))
    k = pd.read_csv(os.path.join(DATA_DIR, "kan_results.csv"))
    l = e[e["test"] == "uah lodo"].set_index("held_out")
    kk = k[k["data"] == "uah lodo"].set_index("held_out")
    drv = list(l.index)
    x = np.arange(len(drv) + 1)
    vals = {"reference": list(l["auc_reference"]) + [l["auc_reference"].mean()],
            "extended": list(l["auc_extended"]) + [l["auc_extended"].mean()],
            "KAN": list(kk.loc[drv, "kan"]) + [kk.loc[drv, "kan"].mean()]}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for j, (n, c) in enumerate(zip(vals, [GREY, GOLD, MAROON])):
        b = ax.bar(x + (j - 1) * 0.27, vals[n], 0.27, color=c, label=n)
        label_bars(ax, b[-1:], "{:.2f}", 0.004)
    ax.set_xticks(x)
    ax.set_xticklabels(drv + ["mean"])
    ax.set_ylim(0.7, 0.97)
    ax.set_ylabel("AUC on the driver left out")
    ax.set_title("UAH, each driver tested after training on the other five")
    ax.legend(frameon=False, ncol=3)
    save(fig, "formulas")


def learning_curves():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for algo, col, lab in (("ppo_small6_wait5", MAROON, "PPO"), ("dqn_full", BLUE, "DQN")):
        f = os.path.join(DATA_DIR, f"learning_curve_{algo}.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            ax.plot(d["steps"] / 1e6, 100 * d["balanced_acc"], "o-", color=col, lw=2.2, label=lab, ms=4)
    ax.set_xlabel("training steps (millions)  ·  1 million steps ≈ 2.6 minutes")
    ax.set_ylabel("accuracy, SUMO test drivers (%)")
    ax.set_ylim(30, 85)
    ax.axvspan(0, 3, color=BLUE, alpha=0.06)
    ax.text(1.5, 33, "DQN still exploring at random", color=BLUE, fontsize=9, ha="center")
    ax.legend(frameon=False)
    save(fig, "learning_curves")


def agents():
    p = pd.read_csv(os.path.join(DATA_DIR, "ablation_state_reward.csv")).set_index("config")
    f = pd.read_csv(os.path.join(DATA_DIR, "fsm_vs_joint.csv"))
    f = f[f["environment"] == "all"].set_index("agent")
    names = ["previous best\n(8 inputs)", "+ new signals\nplain reward", "+ risk\nreward", "+ 3x longer\ntraining",
             "3 experts\n+ FSM"]
    vals = [p.loc["small6_wait5", "balanced_acc"], f.loc["joint_plain", "balanced_acc"], f.loc["joint_risk", "balanced_acc"],
            f.loc["joint_risk_long", "balanced_acc"], f.loc["fsm_experts", "balanced_acc"]]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    b = ax.bar(names, vals, color=[GREY, MAROON, MAROON, MAROON, GOLD])
    label_bars(ax, b, "{:.1%}", 0.004)
    ax.set_ylim(0.6, 0.85)
    ax.set_ylabel("accuracy, SUMO test drivers")
    ax.set_title("Only new information moves the ceiling")
    save(fig, "agents")


def fsm():
    f = pd.read_csv(os.path.join(DATA_DIR, "fsm_vs_joint.csv"))
    envs = ["highway", "urban", "weather"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(envs))
    for j, (ag, col, lab) in enumerate([("joint_risk", MAROON, "one joint agent"), ("fsm_experts", GOLD, "3 experts + FSM")]):
        g = f[f["agent"] == ag].set_index("environment")
        b = a1.bar(x + (j - 0.5) * 0.38, [g.loc[e, "balanced_acc"] for e in envs], 0.38, color=col, label=lab)
        label_bars(a1, b, "{:.0%}", 0.005)
    a1.set_xticks(x)
    a1.set_xticklabels(envs)
    a1.set_ylim(0.6, 0.95)
    a1.set_title("Accuracy per road type")
    a1.legend(frameon=False)
    t = pd.read_csv(os.path.join(DATA_DIR, "fsm_vs_joint_thresholds.csv"))
    t = t[t["agent"] == "fsm_experts"].set_index("environment")
    for c, col in (("conservative", BLUE), ("normal", GOLD), ("aggressive", MAROON)):
        a2.plot(envs, [t.loc[e, c] for e in envs], "o-", color=col, lw=2.2, ms=8, label=c)
    a2.set_ylabel("formula score when the expert gives this label")
    a2.set_title("Same rule, shifted per road type")
    a2.legend(frameon=False)
    save(fig, "fsm")


def german():
    g = pd.read_csv(os.path.join(DATA_DIR, "german_transfer.csv"))
    a = g[g["test"] == "autobahn test"].set_index("agent")
    names = ["A: other roads,\nold inputs", "B: other roads,\n+ relative speeds\n+ physics", "C: + German data\n+ country one-hot",
             "D: German\ndata only"]
    keys = ["A_naive", "B_relative", "C_de_onehot", "D_de_only"]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    b = ax.bar(names, [a.loc[k, "balanced_acc"] for k in keys], color=[GREY, MAROON, GOLD, BLUE])
    label_bars(ax, b, "{:.1%}", 0.004)
    ax.set_ylim(0.6, 0.85)
    ax.set_ylabel("accuracy on German autobahn test runs")
    ax.tick_params(axis="x", labelsize=10)
    save(fig, "german")


def highd_proxy():
    h = pd.read_csv(os.path.join(DATA_DIR, "german_transfer_highd_proxy.csv")).iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(2)
    for j, (n, col) in enumerate((("reference", GREY), ("extended", MAROON))):
        b = ax.bar(x + (j - 0.5) * 0.38, [h[f"auc_{n}_all_terms"], h[f"auc_{n}_without_gap_and_limit"]], 0.38, color=col, label=n)
        label_bars(ax, b, "{:.2f}", 0.005)
    ax.set_xticks(x)
    ax.set_xticklabels(["all terms\n(circular: rules use gap and limit)", "without gap and speed vs limit"])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUC vs rule-violation labels")
    ax.set_title(f"Real highD, {int(h['windows']):,} windows")
    ax.legend(frameon=False)
    save(fig, "highd_proxy")


def autobahn():
    d = pd.read_csv(os.path.join(DATA_DIR, "clean", "planted_de.csv.gz"), usecols=["vid", "label", "speed_kmh", "speed_limit_kmh"])
    d["section"] = np.where(d["speed_limit_kmh"].notna(), "120 km/h section", "no limit section")
    m = d.groupby(["section", "label"])["speed_kmh"].median().unstack()
    order = ["truck", "conservative", "normal", "aggressive"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(2)
    for j, (c, col) in enumerate(zip(order, [GREY, BLUE, GOLD, MAROON])):
        b = ax.bar(x + (j - 1.5) * 0.2, [m.loc[s, c] for s in m.index], 0.2, color=col, label=c)
        label_bars(ax, b, "{:.0f}", 1)
    ax.set_xticks(x)
    ax.set_xticklabels(m.index)
    ax.set_ylabel("median speed (km/h)")
    ax.set_ylim(0, 150)
    ax.legend(frameon=False, ncol=4, fontsize=9)
    ax.set_title("Simulated German autobahn")
    save(fig, "autobahn")


if __name__ == "__main__":
    for f in (audit, single_terms, formulas, learning_curves, agents, fsm, german, highd_proxy, autobahn):
        f()
        print("done", f.__name__, flush=True)
