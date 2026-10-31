"""
Scenario experts switched by a finite state machine against one joint agent, on the planted SUMO drivers, with the
extended state and the risk weighted reward (env/labeling_ref.py). Dr. Daher, Oct 2026: train each scenario on its own,
test each on its own, and see what the experts share to learn how to fuse the three scenarios.

    python scripts/fsm_vs_joint.py [--steps 1000000]

Agents (PPO, gamma 1, wait bias 4, wait 1 s and wait 5 s actions; seeds 0 to 6 train, 7 to 9 test):
  joint_plain        one agent for all road types (road type one hot in the state), plain +1 / -1 reward
  joint_risk         same, risk weighted reward
  joint_risk_long    same, 3x the steps (as many as the three experts together)
  fsm_experts        one agent per road type (highway, urban, weather), trained and tested only on its own road type
                     (no road type input); the FSM routes every test vehicle to the expert of its road type
Metrics per agent and road type: balanced accuracy, per type, seconds; on the aggressive test drivers whose highest
danger (TTC under 4 s or headway under 1 s) in the first 60 s reaches 0.5: share caught and seconds to decision.
Commonality: per agent and road type, the score so far at which its labels switch (median score of the vehicles it
labels conservative / normal / aggressive).
Outputs data/fsm_vs_joint.csv, data/fsm_vs_joint_episodes.csv.gz, data/fsm_vs_joint_thresholds.csv.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402
from env.labeling_ref import EXT_FEATURES, LabelingRefEnv, run_policy, sequences_from_clean  # noqa: E402
from state_reward_ablation import CLASSES, STATES, TEST_SEEDS, train_agent  # noqa: E402
from model.aggressiveness_model import REF_WEIGHTS  # noqa: E402

ENVS = ("highway", "urban", "weather")
JOINT = STATES["small6"] + EXT_FEATURES
EXPERT = [f for f in JOINT if not f.startswith("env_")]


def score_after(s, n):
    w = np.asarray(REF_WEIGHTS)
    m = np.nanmean(s["terms"][:n], axis=0)
    k = ~np.isnan(m)
    return 100 * (w[k] * m[k]).sum() / w[k].sum()


def episodes(agent, seqs, cfg):
    preds, secs = run_policy(agent, LabelingRefEnv(seqs, CLASSES, **cfg), len(seqs))
    return pd.DataFrame({"true": [s["label"] for s in seqs], "pred": preds, "seconds": secs,
                         "environment": [s["environment"] for s in seqs], "scenario": [s["scenario"] for s in seqs],
                         "max_danger": [float(np.max(s["ext"][:60, 3])) for s in seqs],
                         "score_at_decision": [score_after(s, int(n)) for s, n in zip(seqs, secs)]})


def metrics(e):
    ok = e["true"] == e["pred"]
    acc = ok.groupby(e["true"]).mean()
    dang = e[(e["true"] == "aggressive") & (e["max_danger"] >= 0.5)]
    return {"balanced_acc": float(acc.mean()), **{f"acc_{c}": float(acc.get(c, np.nan)) for c in CLASSES},
            "seconds": float(e["seconds"].mean()), "dangerous_aggressive": len(dang),
            "dangerous_caught": float((dang["pred"] == "aggressive").mean()) if len(dang) else np.nan,
            "dangerous_seconds": float(dang["seconds"].mean()) if len(dang) else np.nan}


def main(steps):
    d = pd.read_csv(os.path.join(DATA_DIR, "clean", "planted.csv.gz"))
    seqs = sequences_from_clean(d, extended=True)
    train = [s for s in seqs if s["group"] not in TEST_SEEDS]
    test = [s for s in seqs if s["group"] in TEST_SEEDS and len(s["terms"]) >= 60]
    print(f"train {len(train)}, test {len(test)}", flush=True)
    all_ep, rows = [], []
    runs = [("joint_plain", dict(features=JOINT, wait5=True), steps, None),
            ("joint_risk", dict(features=JOINT, wait5=True, risk=True), steps, None),
            ("joint_risk_long", dict(features=JOINT, wait5=True, risk=True), 3 * steps, None)]
    runs += [(f"expert_{e}", dict(features=EXPERT, wait5=True, risk=True), steps, e) for e in ENVS]
    for name, cfg, n_steps, env_only in runs:
        t = time.time()
        tr = [s for s in train if env_only is None or s["environment"] == env_only]
        te = [s for s in test if env_only is None or s["environment"] == env_only]
        agent = train_agent("ppo", LabelingRefEnv(tr, CLASSES, **cfg), n_steps, 0)
        agent.save(os.path.join(DATA_DIR, f"fsm_{name}.zip"))
        e = episodes(agent, te, cfg).assign(agent="fsm_experts" if env_only else name)
        all_ep.append(e)
        print(f"{name}: {len(tr)} train, {len(te)} test, {time.time() - t:.0f} s, balanced {metrics(e)['balanced_acc']:.3f}",
              flush=True)
    ep = pd.concat(all_ep, ignore_index=True)
    ep.to_csv(os.path.join(DATA_DIR, "fsm_vs_joint_episodes.csv.gz"), index=False)
    for agent, g in ep.groupby("agent"):
        rows.append({"agent": agent, "environment": "all", **metrics(g)})
        for env, h in g.groupby("environment"):
            rows.append({"agent": agent, "environment": env, **metrics(h)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DATA_DIR, "fsm_vs_joint.csv"), index=False)
    thr = ep.groupby(["agent", "environment", "pred"])["score_at_decision"].median().unstack("pred")
    thr.to_csv(os.path.join(DATA_DIR, "fsm_vs_joint_thresholds.csv"))
    with pd.option_context("display.width", 200):
        print(res.round(3).to_string(index=False))
        print(thr.round(1).to_string())


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    main(p.parse_args().steps)
