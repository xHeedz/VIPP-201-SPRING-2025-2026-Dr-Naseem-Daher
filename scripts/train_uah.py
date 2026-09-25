"""
Train the DynamicWeightAgent on real drivers from UAH-DriveSet and test it on
drivers it has never seen.

    python scripts/train_uah.py --root /path/to/UAH-DRIVESET-v1

What happens:
  1. Every trip is cut into 10 s windows; each window gets the mean index
     features and the trip's label (normal, aggressive or drowsy).
     Motorway trips belong to the agent's highway environment, secondary-road
     trips to its urban environment. UAH has no weather trips.
  2. Leave-one-driver-out: for each of the 6 drivers, a fresh agent learns its
     weights and thresholds from the other 5 drivers' normal and aggressive
     windows, then is tested on the held-out driver.
  3. The same test is run for the original hand-set index, as the baseline.
  4. A final agent is trained on all drivers and saved to
     data/dynamic_weight_agent.json.

Outputs: data/uah_windows.csv, data/uah_lodo_results.csv, data/uah_summary.json,
data/dynamic_weight_agent.json, results/figures/uah_*.png
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR  # noqa: E402
from datasets.uah import load_windows  # noqa: E402
from model.aggressiveness_model import original_score  # noqa: E402
from model.dynamic_weight_agent import DynamicWeightAgent  # noqa: E402

FEATS = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]
ORIGINAL_THRESHOLD = 70.0


def auc(scores, labels):
    """Probability that a random aggressive window scores above a random normal one."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def scores_of(agent, w):
    X = w[FEATS].to_numpy()
    return np.array([agent.ai(x, e) for x, e in zip(X, w["environment"])])


def evaluate(w, agent):
    """Window- and trip-level metrics for normal vs aggressive windows in w."""
    y = (w["behavior"] == "aggressive").astype(int).to_numpy()
    if agent == "original":
        s = original_score(w[FEATS].to_numpy())
        pred = (s >= ORIGINAL_THRESHOLD).astype(int)
    else:
        s = scores_of(agent, w)
        pred = np.array([v >= agent.threshold(e) for v, e in zip(s, w["environment"])]).astype(int)
    trips = pd.DataFrame({"trip": w["trip"].to_numpy(), "y": y, "pred": pred}).groupby("trip").mean()
    return {"auc": auc(s, y), "window_acc": float((pred == y).mean()),
            "trip_acc": float(((trips["pred"] >= 0.5).astype(int) == trips["y"]).mean()), "n_windows": int(len(w))}


def train(w, epochs):
    agent = DynamicWeightAgent()
    agent.fit_labels(w[FEATS].to_numpy(), (w["behavior"] == "aggressive").astype(int), w["environment"], epochs=epochs)
    return agent


def main(root, epochs):
    win, report = load_windows(root)
    win.to_csv(os.path.join(DATA_DIR, "uah_windows.csv"), index=False)
    print(f"{len(report)} trips, {len(win)} windows")
    print(f"lane detection available {report['lane_coverage'].mean():.0%} of the time, "
          f"vehicle ahead detected {report['leader_coverage'].mean():.0%}")

    binary = win[win["behavior"].isin(["normal", "aggressive"])]
    rows = []
    for d in sorted(binary["driver"].unique()):
        agent = train(binary[binary["driver"] != d], epochs)
        test = binary[binary["driver"] == d]
        new, base = evaluate(test, agent), evaluate(test, "original")
        rows.append({"held_out_driver": d, **{f"agent_{k}": v for k, v in new.items()},
                     **{f"original_{k}": v for k, v in base.items() if k != "n_windows"}})
        print(f"  test on {d}: agent AUC {new['auc']:.2f} window acc {new['window_acc']:.0%} "
              f"trip acc {new['trip_acc']:.0%} | original AUC {base['auc']:.2f} window acc {base['window_acc']:.0%}")
    lodo = pd.DataFrame(rows)
    lodo.to_csv(os.path.join(DATA_DIR, "uah_lodo_results.csv"), index=False)

    final = train(binary, epochs)
    y = (binary["behavior"] == "aggressive").astype(int)
    win["score_agent"] = scores_of(final, win)
    win["score_original"] = original_score(win[FEATS].to_numpy())
    summary = {
        "trips": int(len(report)), "windows": int(len(win)),
        "lodo_mean": lodo.drop(columns="held_out_driver").mean().round(3).to_dict(),
        "single_feature_auc": {f: round(auc(binary[f], y), 3) for f in FEATS},
        "mean_score_by_behavior": win.groupby("behavior")[["score_agent", "score_original"]].mean().round(1).to_dict(),
        "final_agent": {e: v for e, v in final.to_dict().items()},
    }
    with open(os.path.join(DATA_DIR, "uah_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    final.save(os.path.join(DATA_DIR, "dynamic_weight_agent.json"),
               {"trained_on": "UAH-DriveSet normal and aggressive windows, all 6 drivers; "
                              "weather weights are untrained (no weather trips in UAH)",
                "lodo_mean": summary["lodo_mean"]})
    plots(win, lodo, final)
    print(json.dumps({k: summary[k] for k in ("lodo_mean", "single_feature_auc")}, indent=2))
    d = final.to_dict()
    print("learned weights (speed^2, accel, prox^2, wave):", {e: d["weights"][e] for e in ("highway", "urban")},
          "thresholds:", {e: d["thresholds"][e] for e in ("highway", "urban")})


def plots(win, lodo, agent):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"normal": "#5b8def", "aggressive": "#d1495b", "drowsy": "#2a9d8f"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, key, title in zip(axs, ["score_original", "score_agent"],
                              ["original hand-set index", "dynamic weight agent trained on UAH"]):
        for b, g in win.groupby("behavior"):
            ax.hist(g[key], bins=40, range=(0, 100), alpha=0.55, color=col.get(b, "gray"), label=b, density=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("score (10 s window mean)")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "uah_score_distributions.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(lodo))
    ax.bar(x - 0.2, lodo["original_auc"], 0.4, label="original index", color="#bbbbbb")
    ax.bar(x + 0.2, lodo["agent_auc"], 0.4, label="dynamic weight agent", color="#840032")
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xticks(x, lodo["held_out_driver"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC on the unseen driver")
    ax.set_title("normal vs aggressive, leave-one-driver-out", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "uah_lodo_auc.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(4)
    orig = np.array([0.5, 0.2, 0.8, 0.4]) / 1.9
    ax.bar(x - 0.27, orig, 0.27, label="original (normalized)", color="#bbbbbb")
    for k, (e, shade) in enumerate(zip(["highway", "urban"], ["#840032", "#e59500"])):
        ax.bar(x + k * 0.27, agent.weights_np(e), 0.27, label=f"learned, {e}", color=shade)
    ax.set_xticks(x + 0.13, ["speed^2", "accel", "prox^2", "wave"])
    ax.set_ylabel("weight share")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "uah_learned_weights.png"), dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing D1 ... D6")
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()
    main(a.root, a.epochs)
