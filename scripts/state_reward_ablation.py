"""
State, reward and action space of the labeling agent, one change at a time (Dr. Daher, Oct 2026), on the
planted SUMO drivers from the cleaned tables with the reference formula.

    python scripts/state_reward_ablation.py [--steps 1000000] [--configs full,weighted,...] [--algo ppo|dqn]

Split as before: seeds 0 to 6 train, 7 to 9 test (vehicles with at least 60 s). PPO settings as the best earlier
run: gamma 1, wait bias 4 on the 1 s wait action, ent_coef 0.01, n_steps 4096, batch 512.
Baseline on the same test vehicles: reference score of the mean terms after T = 30 and 60 s, two cut offs
refitted per environment on the training vehicles (grid, balanced accuracy).
Outputs data/ablation_state_reward.csv (one row per config) and data/ablation_episodes_<config>.csv.gz.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from env.labeling_ref import (ENVIRONMENTS, STATE_FEATURES, LabelingRefEnv, evaluate, run_policy,  # noqa: E402
                              sequences_from_clean)
from model.aggressiveness_model import REF_WEIGHTS  # noqa: E402

CLASSES = ("conservative", "normal", "aggressive")
TEST_SEEDS = (7, 8, 9)
ENV1H = [f"env_{e}" for e in ENVIRONMENTS]
STATES = {
    "full": STATE_FEATURES,
    "weighted": [f"pts_{t}_{p}" for p in ("run", "rec") for t in ("speed", "accel2", "prox2", "wave")] + ["time"] + ENV1H,
    "small6": ["score_run", "score_rec", "prox_follow", "leader_share", "time"] + ENV1H,
    "small4": ["score_run", "score_rec", "leader_share", "time"],
    "tiny": ["score_run", "time"],
}
CONFIGS = {name: dict(features=f) for name, f in STATES.items()}
CONFIGS.update({
    "small6_growing_wait": dict(features=STATES["small6"], wait_growth=0.0001),
    "small6_ordinal_cost": dict(features=STATES["small6"], cost="ordinal"),
    "small6_wait5": dict(features=STATES["small6"], wait5=True),
})


def score_after(seqs, T):
    w = np.asarray(REF_WEIGHTS)
    out = []
    for s in seqs:
        m = np.nanmean(s["terms"][:T], axis=0)
        k = ~np.isnan(m)
        out.append(100 * (w[k] * m[k]).sum() / w[k].sum())
    return np.array(out)


def fit_cuts(scores, labels):
    best = (-1, 0, 0)
    for lo in np.arange(5, 60, 1.0):
        for hi in np.arange(lo + 1, 80, 1.0):
            p = np.where(scores < lo, CLASSES[0], np.where(scores < hi, CLASSES[1], CLASSES[2]))
            b = np.mean([(p[labels == c] == c).mean() for c in CLASSES])
            if b > best[0]:
                best = (b, lo, hi)
    return best[1:]


def baseline(train, test, T):
    pred = np.empty(len(test), dtype=object)
    s_tr, s_te = score_after(train, T), score_after(test, T)
    y_tr = np.array([s["label"] for s in train])
    e_tr, e_te = np.array([s["environment"] for s in train]), np.array([s["environment"] for s in test])
    for e in np.unique(e_te):
        lo, hi = fit_cuts(s_tr[e_tr == e], y_tr[e_tr == e])
        x = s_te[e_te == e]
        pred[e_te == e] = np.where(x < lo, CLASSES[0], np.where(x < hi, CLASSES[1], CLASSES[2]))
    return evaluate(test, list(pred), [T] * len(test))


def train_agent(algo, env, steps, seed, wait_bias=4.0):
    if algo == "dqn":
        from stable_baselines3 import DQN
        m = DQN("MlpPolicy", env, seed=seed, verbose=0, gamma=1.0, learning_rate=1e-4, buffer_size=200_000,
                learning_starts=10_000, batch_size=256, target_update_interval=2_000, train_freq=4,
                exploration_fraction=0.3, exploration_final_eps=0.05)
    else:
        from stable_baselines3 import PPO
        m = PPO("MlpPolicy", env, seed=seed, verbose=0, ent_coef=0.01, n_steps=4096, batch_size=512, gamma=1.0)
        with torch.no_grad():
            m.policy.action_net.bias[0] += wait_bias
    m.learn(total_timesteps=steps)
    return m


def main(steps, configs, algo, seed):
    d = pd.read_csv(os.path.join(DATA_DIR, "clean", "planted.csv.gz"))
    seqs = sequences_from_clean(d)
    train = [s for s in seqs if s["group"] not in TEST_SEEDS]
    test = [s for s in seqs if s["group"] in TEST_SEEDS and len(s["terms"]) >= 60]
    print(f"train {len(train)}, test {len(test)} vehicles; weights {REF_WEIGHTS}", flush=True)
    out_csv = os.path.join(DATA_DIR, f"ablation_state_reward{'' if algo == 'ppo' else '_' + algo}.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    if not any(r["config"].startswith("formula") for r in rows):
        for T in (30, 60):
            _, m = baseline(train, test, T)
            rows.append({"config": f"formula_T{T}", "algo": "formula", "n_state": 0, **m})
            print(f"formula after {T} s: {m['balanced_acc']:.3f}", flush=True)
    for name in configs:
        cfg = CONFIGS[name]
        t = time.time()
        env = LabelingRefEnv(train, CLASSES, **cfg)
        model = train_agent(algo, env, steps, seed)
        test_env = LabelingRefEnv(test, CLASSES, **cfg)
        ep, m = evaluate(test, *run_policy(model, test_env, len(test)))
        ep.to_csv(os.path.join(DATA_DIR, f"ablation_episodes_{algo}_{name}.csv.gz"), index=False)
        model.save(os.path.join(DATA_DIR, f"ablation_{algo}_{name}.zip"))
        rows = [r for r in rows if not (r["config"] == name and r["algo"] == algo)]
        rows.append({"config": name, "algo": algo, "n_state": len(cfg["features"]), **m,
                     "train_s": round(time.time() - t)})
        print(f"{algo} {name} ({len(cfg['features'])} inputs): balanced {m['balanced_acc']:.3f}, "
              f"cons {m.get('acc_conservative', 0):.2f} norm {m.get('acc_normal', 0):.2f} agg {m.get('acc_aggressive', 0):.2f}, "
              f"{m['seconds']:.1f} s, at 1 s {m['decided_at_1s']:.2f}, {time.time() - t:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--configs", default="full,weighted,small6,small4,tiny")
    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.steps, [c.strip() for c in a.configs.split(",")], a.algo, a.seed)
