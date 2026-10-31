"""
How long does the labeling agent need to train? Learning curve of PPO (or DQN) on the planted SUMO drivers.

    python scripts/learning_curve.py [--algo ppo] [--config small6_wait5] [--steps 10000000] [--every 500000]

Same environment, state and split as scripts/state_reward_ablation.py. Every --every steps the current policy is
tested on a fixed stratified sample of test vehicles (seeds 7 to 9, 500 per type) and the wall clock time is logged.
Output data/learning_curve_<algo>_<config>.csv (steps, minutes, balanced accuracy, per type, seconds to decision).
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
from env.labeling_ref import LabelingRefEnv, evaluate, run_policy, sequences_from_clean  # noqa: E402
from state_reward_ablation import CLASSES, CONFIGS, TEST_SEEDS, train_agent  # noqa: E402


def main(algo, config, steps, every, per_type, seed):
    from stable_baselines3.common.callbacks import BaseCallback
    d = pd.read_csv(os.path.join(DATA_DIR, "clean", "planted.csv.gz"))
    seqs = sequences_from_clean(d)
    train = [s for s in seqs if s["group"] not in TEST_SEEDS]
    test = [s for s in seqs if s["group"] in TEST_SEEDS and len(s["terms"]) >= 60]
    rng = np.random.default_rng(0)
    sample = []
    for c in CLASSES:
        pool = [s for s in test if s["label"] == c]
        sample += [pool[i] for i in rng.permutation(len(pool))[:per_type]]
    cfg = CONFIGS[config]
    rows, t_start = [], time.time()
    out = os.path.join(DATA_DIR, f"learning_curve_{algo}_{config}.csv")

    class Eval(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next = every

        def _on_step(self):
            if self.num_timesteps >= self.next:
                self.next += every
                _, m = evaluate(sample, *run_policy(self.model, LabelingRefEnv(sample, CLASSES, **cfg), len(sample)))
                rows.append({"steps": self.num_timesteps, "minutes": (time.time() - t_start) / 60, **m})
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{self.num_timesteps:>10,} steps, {rows[-1]['minutes']:.1f} min: balanced {m['balanced_acc']:.3f}, "
                      f"{m['seconds']:.1f} s", flush=True)
            return True

    env = LabelingRefEnv(train, CLASSES, **cfg)
    model = train_agent(algo, env, 0, seed)                 # built, not trained yet
    model.learn(total_timesteps=steps, callback=Eval())
    model.save(os.path.join(DATA_DIR, f"learning_curve_{algo}_{config}.zip"))
    print("written", out)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    p.add_argument("--config", default="small6_wait5")
    p.add_argument("--steps", type=int, default=10_000_000)
    p.add_argument("--every", type=int, default=500_000)
    p.add_argument("--per-type", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.algo, a.config, a.steps, a.every, a.per_type, a.seed)
