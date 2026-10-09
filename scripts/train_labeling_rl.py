"""
Sequential labeling with RL on UAH-DriveSet (option a of docs/rl_pivot.md), leave one driver out.

    python scripts/train_labeling_rl.py [--steps 300000] [--wait-cost 0.01] [--horizon 120]
                                        [--miss-aggressive-cost 2] [--agent-obs]

Per held-out driver (same 6 folds as scripts/train_uah.py):
  - PPO (stable-baselines3) on env/labeling_env.py with the other 5 drivers' normal and aggressive trips
  - test episodes: every trip of the held-out driver, start every 30 s, only starts with a full horizon left
  - compared on the same test episodes against
      reference index after a fixed T (10, 30, 60, 120 s): aggressive if AI of the mean features >= 42
      DynamicWeightAgent after a fixed T (the fold's agent from data/uah_lodo_agents/, its own threshold)
      two-threshold stopping rule (not RL): aggressive once the running AI >= hi, normal once <= lo, else wait;
          at the horizon label by (lo + hi) / 2. lo and hi fitted on the 5 training drivers by grid search
          on the same reward as the RL agent
Metrics: balanced accuracy (mean of per-class accuracy), mean seconds to decision, mean reward
(+1 / -1, minus wait_cost per second waited). The cut off 42 was fitted on all UAH drivers, so the
reference index baseline has a small advantage on the held-out driver.

Outputs data/labeling_rl_{lodo,summary}_wait<cost>.csv and data/labeling_rl_episodes_wait<cost>.csv.gz.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from env.labeling_env import LabelingEnv, reference_ai, uah_sequences  # noqa: E402
from model.aggressiveness_model import THRESHOLDS  # noqa: E402
from model.dynamic_weight_agent import DynamicWeightAgent  # noqa: E402

UAH_ROOT = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026", "UAH-DRIVESET-v1"))
FIXED_T = (10, 30, 60, 120)
START_EVERY_S = 30
CLASSES = ("normal", "aggressive")


def start_points(seqs, horizon):
    return [(i, t0) for i, s in enumerate(seqs) for t0 in range(0, len(s["features"]) - horizon + 1, START_EVERY_S)]


def score(rows, wait_cost, miss_cost=1.0):
    """rows: list of (true, predicted or None, seconds) -> balanced accuracy, seconds, reward; a wrong label
    on an aggressive driver costs miss_cost, on a normal one 1 (same as the env). balanced_reward: mean of
    the per class mean reward."""
    d = pd.DataFrame(rows, columns=["true", "pred", "seconds"])
    d["correct"] = d["true"] == d["pred"]
    wrong = np.where(d["true"] == "aggressive", -miss_cost, -1.0)
    d["reward"] = np.where(d["correct"], 1.0, wrong) - wait_cost * (d["seconds"] - 1)
    return {"balanced_acc": float(d.groupby("true")["correct"].mean().mean()),
            "balanced_reward": float(d.groupby("true")["reward"].mean().mean()),
            "acc_normal": float(d.loc[d["true"] == "normal", "correct"].mean()),
            "acc_aggressive": float(d.loc[d["true"] == "aggressive", "correct"].mean()),
            "seconds": float(d["seconds"].mean()), "reward": float(d["reward"].mean()), "episodes": len(d)}


def run_fixed(seqs, starts, T, decide):
    rows = []
    for i, t0 in starts:
        s = seqs[i]
        rows.append((s["label"], decide(s["features"][t0:t0 + T].mean(axis=0), s["environment"]), T))
    return rows


def ai_curves(seqs, starts, horizon):
    """running AI of every start point, (n_starts, horizon), and the true labels"""
    curves = np.empty((len(starts), horizon))
    for k, (i, t0) in enumerate(starts):
        f = seqs[i]["features"][t0:t0 + horizon]
        curves[k] = reference_ai(np.cumsum(f, axis=0) / np.arange(1, horizon + 1)[:, None])
    return curves, [seqs[i]["label"] for i, _ in starts]


def run_rule(curves, labels, lo, hi):
    horizon = curves.shape[1]
    hit = (curves >= hi) | (curves <= lo)
    any_hit = hit.any(axis=1)
    n = np.where(any_hit, hit.argmax(axis=1) + 1, horizon)
    a = curves[np.arange(len(curves)), n - 1]
    aggr = np.where(any_hit, a >= hi, a >= (lo + hi) / 2)
    return [(y, CLASSES[int(g)], int(k)) for y, g, k in zip(labels, aggr, n)]


def fit_rule(curves, labels, wait_cost, miss_cost):
    """lo, hi maximising the balanced reward on the training drivers"""
    best = None
    grid = np.arange(0.10, 0.80, 0.02)
    for lo in grid:
        for hi in grid[grid > lo]:
            key = score(run_rule(curves, labels, lo, hi), wait_cost, miss_cost)["balanced_reward"]
            if best is None or key > best[0]:
                best = (key, lo, hi)
    return best[1], best[2]


def run_policy(model, env, starts):
    rows = []
    for i, t0 in starts:
        obs, _ = env.reset(options={"seq": i, "t0": t0})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        rows.append((info["true"], info["predicted"], info["seconds"]))
    return rows


def attach_agent(seqs, agent):
    """copy of the sequences carrying the fold's DynamicWeightAgent weights and threshold per environment"""
    return [{**s, "agent": {"weights": agent.weights_np(s["environment"]), "threshold": agent.threshold(s["environment"])}}
            for s in seqs]


def main(steps, wait_cost, horizon, seed, miss_cost, agent_obs, gamma=1.0):
    tag = f"_wait{wait_cost:g}" + (f"_miss{miss_cost:g}" if miss_cost != 1 else "") + ("_agentobs" if agent_obs else "") + ("" if gamma == 1 else f"_g{gamma:g}")
    from stable_baselines3 import PPO
    seqs = uah_sequences(UAH_ROOT)
    drivers = sorted({s["driver"] for s in seqs})
    print(f"{len(seqs)} trips ({sum(s['label'] == 'aggressive' for s in seqs)} aggressive), "
          f"{sum(len(s['features']) for s in seqs)} seconds, drivers {drivers}")
    results, episodes = [], []
    for d in drivers:
        agent = DynamicWeightAgent.load(os.path.join(DATA_DIR, "uah_lodo_agents", f"agent_without_{d}.json"))
        train = attach_agent([s for s in seqs if s["driver"] != d], agent)
        test = attach_agent([s for s in seqs if s["driver"] == d], agent)
        test_starts, train_starts = start_points(test, horizon), start_points(train, horizon)
        t = time.time()
        costs = {"aggressive": miss_cost}
        env = LabelingEnv(train, CLASSES, horizon=horizon, wait_cost=wait_cost, wrong_cost=costs, agent_obs=agent_obs)
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, ent_coef=0.01, n_steps=2048, batch_size=256, gamma=gamma)
        model.learn(total_timesteps=steps)
        test_env = LabelingEnv(test, CLASSES, horizon=horizon, wait_cost=wait_cost, wrong_cost=costs, agent_obs=agent_obs)
        methods = {"rl_ppo": run_policy(model, test_env, test_starts)}
        lo, hi = fit_rule(*ai_curves(train, train_starts, horizon), wait_cost, miss_cost)
        methods["stopping_rule"] = run_rule(*ai_curves(test, test_starts, horizon), lo, hi)
        for T in FIXED_T:
            methods[f"reference_T{T}"] = run_fixed(test, test_starts, T, lambda m, e: CLASSES[int(100 * reference_ai(m) >= THRESHOLDS[1])])
            methods[f"weight_agent_T{T}"] = run_fixed(test, test_starts, T, lambda m, e, a=agent: CLASSES[int(a.ai(m, e) >= a.threshold(e))])
        for name, rows in methods.items():
            results.append({"held_out_driver": d, "method": name, **score(rows, wait_cost, miss_cost)})
            episodes += [{"held_out_driver": d, "method": name, "true": a, "pred": b, "seconds": c} for a, b, c in rows]
        r = {m["method"]: m for m in results if m["held_out_driver"] == d}
        print(f"  {d}: {len(test_starts)} test episodes, {time.time() - t:.0f} s | "
              f"RL acc {r['rl_ppo']['balanced_acc']:.2f} in {r['rl_ppo']['seconds']:.0f} s | "
              f"rule ({lo:.2f}/{hi:.2f}) {r['stopping_rule']['balanced_acc']:.2f} in {r['stopping_rule']['seconds']:.0f} s | "
              f"ref T60 {r['reference_T60']['balanced_acc']:.2f}")
    lodo = pd.DataFrame(results)
    lodo.to_csv(os.path.join(DATA_DIR, f"labeling_rl_lodo{tag}.csv"), index=False)
    pd.DataFrame(episodes).to_csv(os.path.join(DATA_DIR, f"labeling_rl_episodes{tag}.csv.gz"), index=False)
    summary = lodo.groupby("method")[["balanced_acc", "acc_normal", "acc_aggressive", "seconds", "reward", "balanced_reward"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.sort_values("balanced_reward_mean", ascending=False)
    summary.to_csv(os.path.join(DATA_DIR, f"labeling_rl_summary{tag}.csv"))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--wait-cost", type=float, default=0.01)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=1.0, help="PPO discount; episodes are short, 1 = no discount")
    p.add_argument("--miss-aggressive-cost", type=float, default=1.0, help="cost of a wrong label on an aggressive driver")
    p.add_argument("--agent-obs", action="store_true", help="add the DynamicWeightAgent margin to the observation")
    a = p.parse_args()
    main(a.steps, a.wait_cost, a.horizon, a.seed, a.miss_aggressive_cost, a.agent_obs, a.gamma)
