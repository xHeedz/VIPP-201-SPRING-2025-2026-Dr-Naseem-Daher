"""
Train (calibrate) the Aggressiveness Index on UAH-DriveSet labels and test it
on drivers it has never seen.

    python scripts/train_uah.py --root /path/to/UAH-DRIVESET-v1

What happens:
  1. Every trip is cut into 10 s windows; each window gets the mean index
     features and the trip's label (normal, aggressive or drowsy).
  2. Leave-one-driver-out: for each of the 6 drivers, fit the weights and
     thresholds on the other 5 drivers' normal and aggressive windows, then
     test on the held-out driver.
  3. The same test is run for the original hand-set index, as the baseline.
  4. A final model is fitted on all drivers and saved to data/calibrated_index.json.

Outputs: data/uah_windows.csv, data/uah_lodo_results.csv, data/uah_summary.json,
data/calibrated_index.json, results/figures/uah_*.png
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
from model.calibrated_index import CONTEXTS, ORIGINAL_THRESHOLD, CalibratedIndex, original_ai  # noqa: E402

FEATS = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]


def auc(scores, labels):
    """Probability that a random aggressive window scores above a random normal one."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(w, model):
    """Window- and trip-level metrics for normal vs aggressive windows in w."""
    out = {}
    X = w[FEATS].to_numpy()
    y = (w["behavior"] == "aggressive").astype(int).to_numpy()
    if model == "original":
        ai = original_ai(X)
        pred = (ai >= ORIGINAL_THRESHOLD).astype(int)
    else:
        ai = np.array([model.ai(x, c) for x, c in zip(X, w["road"])])
        pred = np.array([model.ai(x, c) >= model.thresholds[c] for x, c in zip(X, w["road"])]).astype(int)
    out["auc"] = auc(ai, y)
    out["window_acc"] = float((pred == y).mean())
    trips = pd.DataFrame({"trip": w["trip"].to_numpy(), "y": y, "pred": pred}).groupby("trip").mean()
    out["trip_acc"] = float(((trips["pred"] >= 0.5).astype(int) == trips["y"]).mean())
    out["n_windows"] = int(len(w))
    return out, ai


def main(root, epochs):
    win, report = load_windows(root)
    win.to_csv(os.path.join(DATA_DIR, "uah_windows.csv"), index=False)
    print(f"{len(report)} trips, {len(win)} windows")
    print(f"lane detection available {report['lane_coverage'].mean():.0%} of the time, "
          f"vehicle ahead detected {report['leader_coverage'].mean():.0%}")

    binary = win[win["behavior"].isin(["normal", "aggressive"])]
    rows = []
    for d in sorted(binary["driver"].unique()):
        train, test = binary[binary["driver"] != d], binary[binary["driver"] == d]
        m = CalibratedIndex()
        m.fit(train[FEATS].to_numpy(), (train["behavior"] == "aggressive").astype(int), train["road"], epochs=epochs)
        cal, _ = evaluate(test, m)
        base, _ = evaluate(test, "original")
        rows.append({"held_out_driver": d, **{f"calibrated_{k}": v for k, v in cal.items()},
                     **{f"original_{k}": v for k, v in base.items() if k != "n_windows"}})
        print(f"  test on {d}: calibrated AUC {cal['auc']:.2f} window acc {cal['window_acc']:.0%} "
              f"trip acc {cal['trip_acc']:.0%} | original AUC {base['auc']:.2f} window acc {base['window_acc']:.0%}")
    lodo = pd.DataFrame(rows)
    lodo.to_csv(os.path.join(DATA_DIR, "uah_lodo_results.csv"), index=False)

    final = CalibratedIndex()
    final.fit(binary[FEATS].to_numpy(), (binary["behavior"] == "aggressive").astype(int), binary["road"], epochs=epochs)
    single = {f: auc(binary[f], (binary["behavior"] == "aggressive").astype(int)) for f in FEATS}
    X = win[FEATS].to_numpy()
    win["ai_calibrated"] = [final.ai(x, c) for x, c in zip(X, win["road"])]
    win["ai_original"] = original_ai(X)
    by_behavior = win.groupby("behavior")[["ai_calibrated", "ai_original"]].mean().round(1).to_dict()
    summary = {
        "trips": int(len(report)), "windows": int(len(win)),
        "lodo_mean": lodo.drop(columns="held_out_driver").mean().round(3).to_dict(),
        "single_feature_auc": {k: round(v, 3) for k, v in single.items()},
        "mean_ai_by_behavior": by_behavior,
        "final_model": final.to_dict(),
    }
    with open(os.path.join(DATA_DIR, "uah_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    final.save(os.path.join(DATA_DIR, "calibrated_index.json"),
               {"trained_on": "UAH-DriveSet, normal vs aggressive windows, all 6 drivers",
                "lodo_mean": summary["lodo_mean"]})
    plots(win, lodo, final)
    print(json.dumps({k: summary[k] for k in ("lodo_mean", "single_feature_auc")}, indent=2))
    print("final weights (speed^2, accel, prox^2, wave):",
          {c: final.to_dict()["weights"][c] for c in CONTEXTS}, "thresholds:", final.to_dict()["thresholds"])


def plots(win, lodo, final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"normal": "#5b8def", "aggressive": "#d1495b", "drowsy": "#2a9d8f"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, key, title in zip(axs, ["ai_original", "ai_calibrated"], ["original hand-set index", "calibrated index"]):
        for b, g in win.groupby("behavior"):
            ax.hist(g[key], bins=40, range=(0, 100), alpha=0.55, color=col.get(b, "gray"), label=b, density=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Aggressiveness Index (10 s window mean)")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "uah_ai_distributions.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(lodo))
    ax.bar(x - 0.2, lodo["original_auc"], 0.4, label="original", color="#bbbbbb")
    ax.bar(x + 0.2, lodo["calibrated_auc"], 0.4, label="calibrated", color="#840032")
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xticks(x, lodo["held_out_driver"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC on the unseen driver")
    ax.set_title("normal vs aggressive, leave-one-driver-out", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "uah_lodo_auc.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = ["speed^2", "accel", "prox^2", "wave"]
    orig = np.array([0.5, 0.2, 0.8, 0.4]) / 1.9
    x = np.arange(4)
    ax.bar(x - 0.27, orig, 0.27, label="original (normalized)", color="#bbbbbb")
    for k, (c, shade) in enumerate(zip(CONTEXTS, ["#840032", "#e59500"])):
        ax.bar(x + k * 0.27, final.weights[c], 0.27, label=f"learned, {c}", color=shade)
    ax.set_xticks(x + 0.13, names)
    ax.set_ylabel("weight share")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "uah_learned_weights.png"), dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing D1 ... D6")
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    main(a.root, a.epochs)
