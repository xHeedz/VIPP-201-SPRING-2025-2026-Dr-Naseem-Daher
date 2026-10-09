"""
Sequential labeling with RL on the planted SUMO drivers: three classes, per vehicle labels.

    python scripts/train_labeling_rl_planted.py [--steps 1000000] [--wait-cost 0.002] [--horizon 60]
                                                [--proximity metres|headway|mixed] [--seed 0]

Episodes: one planted vehicle (data/sumo_planted/*_s<seed>.per_second.csv.gz, 7 settings x 10 seeds), seconds
below 5 km/h dropped. Split by seed: seeds 0 to 6 train, 7 to 9 test (other drivers, other traffic runs).
Test episodes: every test vehicle with at least `horizon` seconds, from its first second.
Baselines on the same episodes, label from the reference AI of the mean features after a fixed T (10, 30, 60 s):
  reference_T*   cut offs 29 / 42 (model/aggressiveness_model.py THRESHOLDS)
  refit_T*       two cut offs refitted on the training seeds for that T (grid, balanced accuracy)
  refit_env_T*   same, one pair of cut offs per environment (highway, urban, weather; PPO sees the environment too)
Proximity: metres (index_features), headway (headway_features) or mixed (mixed_features with the alpha of
each environment from data/proximity_mix.json, scripts/fit_proximity_mix.py); baselines use the same features.
PPO discount gamma 1 by default: a label after n seconds would otherwise be worth gamma^n (0.99^60 = 0.55),
much more than the wait cost, and the policy commits after one second.
Wait bias: at the start every action has probability 1/4 and three of them end the episode, so early episodes
last about 1.3 s and the policy learns to commit at once; --wait-bias b makes the untrained policy wait.
Metrics: balanced accuracy over the three classes, per class accuracy, seconds to decision, balanced reward.
Outputs data/labeling_rl_planted_{summary,by_scenario}[_<proximity>]_seed<k>.csv, ..._episodes..csv.gz and the
policy data/labeling_rl_planted[_<proximity>]_ppo_seed<k>.zip (no suffix for metres).
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from env.labeling_env import LabelingEnv, planted_sequences, reference_ai  # noqa: E402
from model.aggressiveness_model import THRESHOLDS, headway_features, index_features, mixed_features  # noqa: E402

CLASSES = ("conservative", "normal", "aggressive")
FIXED_T = (10, 30, 60)
TEST_SEEDS = (7, 8, 9)


def to_rows(seqs, preds, seconds):
    return pd.DataFrame({"true": [s["label"] for s in seqs], "pred": preds, "seconds": seconds,
                         "scenario": [s["scenario"] for s in seqs]})


def metrics(d, wait_cost):
    d = d.assign(correct=d["true"] == d["pred"])
    d["reward"] = np.where(d["correct"], 1.0, -1.0) - wait_cost * (d["seconds"] - 1)
    acc = d.groupby("true")["correct"].mean()
    return {"balanced_acc": float(acc.mean()), **{f"acc_{c}": float(acc.get(c, np.nan)) for c in CLASSES},
            "seconds": float(d["seconds"].mean()), "balanced_reward": float(d.groupby("true")["reward"].mean().mean()),
            "episodes": int(len(d))}


def label_by(ai100, lo, hi):
    return np.where(ai100 < lo, CLASSES[0], np.where(ai100 < hi, CLASSES[1], CLASSES[2]))


def mean_ai(seqs, T):
    return np.array([100 * reference_ai(s["features"][:T].mean(axis=0)) for s in seqs])


def refit(seqs, T):
    ai, y = mean_ai(seqs, T), np.array([s["label"] for s in seqs])
    best = None
    for lo in np.arange(10, 60, 1.0):
        for hi in np.arange(lo + 1, 80, 1.0):
            p = label_by(ai, lo, hi)
            b = np.mean([(p[y == c] == c).mean() for c in CLASSES])
            if best is None or b > best[0]:
                best = (b, lo, hi)
    return best[1], best[2]


def run_policy(model, env, n):
    preds, secs = [], []
    for i in range(n):
        obs, _ = env.reset(options={"seq": i, "t0": 0})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        preds.append(info["predicted"])
        secs.append(info["seconds"])
    return preds, secs


def feature_fn(proximity):
    if proximity == "metres":
        return index_features
    if proximity == "headway":
        return headway_features
    with open(os.path.join(DATA_DIR, "proximity_mix.json")) as f:
        alpha = json.load(f)["alpha_metres"]
    print("alpha (share of metres):", alpha)
    return {e: (lambda s, a, g, w, al=al: mixed_features(s, a, g, w, al)) for e, al in alpha.items()}


def main(steps, wait_cost, horizon, seed, proximity="metres", gamma=1.0, wait_bias=0.0):
    tag = (("" if proximity == "metres" else f"_{proximity}") + ("" if gamma == 1 else f"_g{gamma:g}")
           + ("" if wait_bias == 0 else f"_wb{wait_bias:g}") + f"_seed{seed}")
    from stable_baselines3 import PPO
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", "*_s*.per_second.csv.gz")))
    paths = [p for p in paths if "us101" not in os.path.basename(p)]
    seqs = planted_sequences(paths, features=feature_fn(proximity))
    train = [s for s in seqs if s["seed"] not in TEST_SEEDS]
    test = [s for s in seqs if s["seed"] in TEST_SEEDS and len(s["features"]) >= horizon]
    print(f"{len(seqs)} vehicles from {len(paths)} runs; train {len(train)}, test {len(test)} (>= {horizon} s)")
    print("test classes:", pd.Series([s["label"] for s in test]).value_counts().to_dict())
    t = time.time()
    env = LabelingEnv(train, CLASSES, horizon=horizon, wait_cost=wait_cost)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, ent_coef=0.01, n_steps=4096, batch_size=512, gamma=gamma)
    if wait_bias:      # untrained policy waits with probability e^b / (e^b + 3) per second instead of 1/4
        with torch.no_grad():
            model.policy.action_net.bias[0] += wait_bias
    model.learn(total_timesteps=steps)
    print(f"PPO {steps} steps in {time.time() - t:.0f} s")
    model.save(os.path.join(DATA_DIR, f"labeling_rl_planted{tag.replace('_seed', '_ppo_seed')}.zip"))
    results = {"rl_ppo": to_rows(test, *run_policy(model, LabelingEnv(test, CLASSES, horizon=horizon, wait_cost=wait_cost), len(test)))}
    for T in FIXED_T:
        ai = mean_ai(test, T)
        results[f"reference_T{T}"] = to_rows(test, label_by(ai, *THRESHOLDS), [T] * len(test))
        lo, hi = refit(train, T)
        results[f"refit_T{T}"] = to_rows(test, label_by(ai, lo, hi), [T] * len(test))
        envs = np.array([s["environment"] for s in test])
        pred = np.empty(len(test), dtype=object)
        cuts = {}
        for e in np.unique(envs):
            cuts[e] = refit([s for s in train if s["environment"] == e], T)
            pred[envs == e] = label_by(ai[envs == e], *cuts[e])
        results[f"refit_env_T{T}"] = to_rows(test, list(pred), [T] * len(test))
        print(f"  refit cut offs at T = {T} s: {lo:.0f} / {hi:.0f}; per environment "
              + ", ".join(f"{e} {a:.0f} / {b:.0f}" for e, (a, b) in cuts.items()))
    summary = pd.DataFrame({m: metrics(d, wait_cost) for m, d in results.items()}).T
    summary = summary.sort_values("balanced_reward", ascending=False)
    summary.to_csv(os.path.join(DATA_DIR, f"labeling_rl_planted_summary{tag}.csv"))
    by_scen = pd.DataFrame([{"method": m, "scenario": sc, **metrics(g, wait_cost)}
                            for m, d in results.items() for sc, g in d.groupby("scenario")])
    by_scen.to_csv(os.path.join(DATA_DIR, f"labeling_rl_planted_by_scenario{tag}.csv"), index=False)
    pd.concat([d.assign(method=m) for m, d in results.items()]).to_csv(
        os.path.join(DATA_DIR, f"labeling_rl_planted_episodes{tag}.csv.gz"), index=False)
    print(summary.astype(float).round(3).to_string())
    piv = by_scen.pivot(index="scenario", columns="method", values="balanced_acc")
    print(piv[["rl_ppo", "refit_T60", "refit_env_T10", "refit_env_T30", "refit_env_T60"]].round(3).to_string())


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--wait-cost", type=float, default=0.002)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=1.0, help="PPO discount; episodes are short, 1 = no discount")
    p.add_argument("--proximity", choices=["metres", "headway", "mixed"], default="metres")
    p.add_argument("--wait-bias", type=float, default=0.0, help="added to the wait logit of the untrained policy")
    a = p.parse_args()
    main(a.steps, a.wait_cost, a.horizon, a.seed, a.proximity, a.gamma, a.wait_bias)
