"""
Sequential labeling with DQN instead of PPO, on the planted SUMO drivers. Work in progress (step 1 of the DQN plan):
same environment, reward, split and evaluation as scripts/train_labeling_rl_planted.py, only the learner changes.

    python scripts/train_labeling_dqn_planted.py [--steps 200000] [--proximity headway] [--seed 0]

Why DQN: the action set is small and discrete (wait, conservative, normal, aggressive). DQN learns one number per
action, Q(s, a) = expected total reward of taking a in state s and acting well afterwards, and picks the largest.
PPO also handles discrete actions; DQN is a comparison, not a fix (same inputs, so a similar ceiling is expected).

Settings: gamma 1 (as PPO: no discount inside a 60 s episode), epsilon-greedy exploration from 1.0 to 0.05 over the
first 30% of steps. Known issue for the next steps: a random action ends the episode 3 times out of 4, so early
experience is mostly 1 s episodes (the same problem the wait bias solved for PPO).
Outputs data/labeling_dqn_planted_{summary,by_scenario}_<proximity>_seed<k>.csv (prefixed "trial_" below 1M steps).
"""
import argparse
import glob
import os
import sys
import time
import warnings

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402
from env.labeling_env import LabelingEnv, planted_sequences  # noqa: E402
from train_labeling_rl_planted import CLASSES, TEST_SEEDS, feature_fn, metrics, run_policy, to_rows  # noqa: E402


def main(steps, wait_cost, horizon, seed, proximity):
    from stable_baselines3 import DQN
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", "*_s*.per_second.csv.gz")))
    paths = [p for p in paths if "us101" not in os.path.basename(p)]
    seqs = planted_sequences(paths, features=feature_fn(proximity))
    train = [s for s in seqs if s["seed"] not in TEST_SEEDS]
    test = [s for s in seqs if s["seed"] in TEST_SEEDS and len(s["features"]) >= horizon]
    print(f"train {len(train)} vehicles, test {len(test)} (>= {horizon} s)")
    t = time.time()
    env = LabelingEnv(train, CLASSES, horizon=horizon, wait_cost=wait_cost)
    model = DQN("MlpPolicy", env, seed=seed, verbose=0, gamma=1.0, learning_rate=1e-4, buffer_size=200_000,
                learning_starts=10_000, batch_size=256, target_update_interval=2_000, train_freq=4,
                exploration_fraction=0.3, exploration_final_eps=0.05)
    model.learn(total_timesteps=steps)
    print(f"DQN {steps} steps in {time.time() - t:.0f} s")
    d = to_rows(test, *run_policy(model, LabelingEnv(test, CLASSES, horizon=horizon, wait_cost=wait_cost), len(test)))
    tag = ("trial_" if steps < 1_000_000 else "") + f"{proximity}_seed{seed}"
    summary = pd.DataFrame({"rl_dqn": metrics(d, wait_cost)}).T
    summary.to_csv(os.path.join(DATA_DIR, f"labeling_dqn_planted_summary_{tag}.csv"))
    by_scen = pd.DataFrame([{"scenario": sc, **metrics(g, wait_cost)} for sc, g in d.groupby("scenario")])
    by_scen.to_csv(os.path.join(DATA_DIR, f"labeling_dqn_planted_by_scenario_{tag}.csv"), index=False)
    print(summary.astype(float).round(3).to_string())
    print("decisions after 1 s: %.0f%%" % (100 * (d["seconds"] == 1).mean()))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--wait-cost", type=float, default=0.002)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--proximity", choices=["metres", "headway", "mixed"], default="headway")
    a = p.parse_args()
    main(a.steps, a.wait_cost, a.horizon, a.seed, a.proximity)
