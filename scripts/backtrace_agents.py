"""
Where the labeling agents (PPO and DQN) make their mistakes, on the planted test vehicles.

    python scripts/backtrace_agents.py [--runs ppo:small6_wait5,dqn:full]

Per run (episodes from scripts/state_reward_ablation.py): confusion matrix, accuracy per scenario, and accuracy by
how much the car followed another car during the seconds the agent watched (share of seconds with a car within
3 s headway), which is when the type specific following gap can show. Also the formula after 60 s on the same split.
Writes docs/hand_checks/backtrace_agents.md and data/backtrace_agents.csv.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402
from env.labeling_ref import sequences_from_clean  # noqa: E402
from state_reward_ablation import CLASSES, TEST_SEEDS  # noqa: E402

HC = os.path.join(os.path.dirname(DATA_DIR), "docs", "hand_checks")


def following_share(seq, n):
    v = seq["speed_kmh"][:n] / 3.6
    g = seq["gap_m"][:n]
    thw = np.where(np.nan_to_num(g) > 0, np.nan_to_num(g) / np.maximum(v, 0.5), np.inf)
    return float(np.mean(thw < 3.0))


def main(runs):
    d = pd.read_csv(os.path.join(DATA_DIR, "clean", "planted.csv.gz"))
    test = [s for s in sequences_from_clean(d[d["group"].isin(TEST_SEEDS)]) if len(s["terms"]) >= 60]
    lines = ["# backtrace: where the labeling agents go wrong", "",
             "planted test vehicles (seeds 7 to 9, at least 60 s). 'following' = share of the watched seconds with a car "
             "ahead within 3 s; only then can the type specific following gap (tau 1.26 / 0.84 / 0.42 s) show.", ""]
    rows = []
    for run in runs:
        algo, cfg = run.split(":")
        e = pd.read_csv(os.path.join(DATA_DIR, f"ablation_episodes_{algo}_{cfg}.csv.gz"))
        e["following"] = [following_share(s, int(n)) for s, n in zip(test, e["seconds"])]
        e["ok"] = e["true"] == e["pred"]
        e["fol_bin"] = pd.cut(e["following"], [-0.01, 0.0, 0.25, 0.75, 1.0], labels=["never", "under 25%", "25 to 75%", "over 75%"])
        cm = pd.crosstab(e["true"], e["pred"], normalize="index").reindex(index=CLASSES, columns=CLASSES).fillna(0)
        lines += [f"## {algo.upper()}, state {cfg}", "", "confusion (rows: true type, share of that type)", "",
                  "| true \\ predicted | " + " | ".join(CLASSES) + " |", "|---|---|---|---|"]
        for c in CLASSES:
            lines.append(f"| {c} | " + " | ".join(f"{cm.loc[c, p]:.2f}" for p in CLASSES) + " |")
        nc = (((e["true"] == "normal") & (e["pred"] == "conservative")) | ((e["true"] == "conservative") & (e["pred"] == "normal")))
        lines += ["", f"share of all mistakes that are normal and conservative mixed up: {nc[~e['ok']].mean():.0%}", "",
                  "accuracy by how much the car followed another car while watched", "",
                  "| following | episodes | accuracy | normal correct | conservative correct | aggressive correct |", "|---|---|---|---|---|---|"]
        for b, g in e.groupby("fol_bin", observed=True):
            acc = g.groupby("true")["ok"].mean()
            lines.append(f"| {b} | {len(g)} | {g['ok'].mean():.2f} | {acc.get('normal', np.nan):.2f} | "
                         f"{acc.get('conservative', np.nan):.2f} | {acc.get('aggressive', np.nan):.2f} |")
            rows.append({"run": run, "following": str(b), "episodes": len(g), "accuracy": g["ok"].mean(),
                         **{f"acc_{c}": acc.get(c, np.nan) for c in CLASSES}})
        lines += ["", "accuracy per scenario (balanced over the three types)", "", "| scenario | balanced accuracy | following share |", "|---|---|---|"]
        for sc, g in e.groupby("scenario"):
            lines.append(f"| {sc} | {g.groupby('true')['ok'].mean().mean():.2f} | {g['following'].mean():.2f} |")
        lines.append("")
        print(run, "done", flush=True)
    with open(os.path.join(HC, "backtrace_agents.md"), "w") as f:
        f.write("\n".join(lines))
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "backtrace_agents.csv"), index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="ppo:small6_wait5,dqn:full")
    main(p.parse_args().runs.split(","))
