"""
Sequential labeling: estimating aggressiveness as a decision problem (option a of docs/rl_pivot.md).

One episode = one driver observed second by second from a random start point. At every second the agent
either waits (sees one more second, small cost) or commits to a label, which ends the episode:

    actions   0 wait, 1 .. K label = classes[action - 1]
    reward    +1 correct label, -cost[true class] wrong label (default 1), -wait_cost per wait;
              waiting at the horizon ends the episode as a wrong label
    obs       running mean of the 4 index features since the start (n_s^2, n_a, n_p^2, n_w),
              mean of the last 10 s, reference AI of both / 100, elapsed / horizon, environment one hot;
              with agent_obs: also (learned score - learned threshold) / 100 of both means, from the
              DynamicWeightAgent weights and threshold the sequence carries (seq["agent"])

The agent does not drive and does not change the driver: waiting only reveals more of the recorded drive.
Real RL because the action decides what is observed next and the reward for waiting is delayed.

A sequence is a dict: features (T, 4) per second index features, label (class name), environment, driver,
trip. Build them with uah_sequences (or any loader that gives per second features).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from model.aggressiveness_model import WEIGHTS, index_features

ENVIRONMENTS = ("highway", "urban", "weather")
RECENT_S = 10


def reference_ai(features):
    """Reference AI (/ 100) of mean features; (..., 4) -> (...)."""
    return np.minimum((np.asarray(features, float) * np.array(WEIGHTS)).sum(axis=-1), 1.0)


def observation(features, t0, n, horizon, environment, agent=None):
    """Observation after n seconds seen from second t0 of a (T, 4) feature array. agent: {"weights": (4,),
    "threshold": score} of a DynamicWeightAgent for this environment; adds its margin for both means."""
    seen = features[t0:t0 + n]
    run, rec = seen.mean(axis=0), seen[-RECENT_S:].mean(axis=0)
    env = np.zeros(len(ENVIRONMENTS))
    env[ENVIRONMENTS.index(environment)] = 1.0
    parts = [run, rec, [reference_ai(run), reference_ai(rec), n / horizon], env]
    if agent is not None:
        w, thr = np.asarray(agent["weights"], float), float(agent["threshold"])
        parts.append(np.clip([(100 * (run * w).sum() - thr) / 100, (100 * (rec * w).sum() - thr) / 100], -1, 1))
    return np.concatenate(parts).astype(np.float32)


class LabelingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sequences, classes=("normal", "aggressive"), horizon=120, wait_cost=0.01,
                 wrong_cost=None, balance=True, agent_obs=False):
        self.sequences = [s for s in sequences if len(s["features"]) >= 1]
        self.classes = tuple(classes)
        self.horizon = int(horizon)
        self.wait_cost = float(wait_cost)
        self.wrong_cost = {c: 1.0 for c in self.classes}
        self.wrong_cost.update(wrong_cost or {})
        self.balance = balance
        self.by_class = {c: [i for i, s in enumerate(self.sequences) if s["label"] == c] for c in self.classes}
        self.action_space = spaces.Discrete(1 + len(self.classes))
        self.agent_obs = agent_obs
        n_obs = 4 + 4 + 3 + len(ENVIRONMENTS) + (2 if agent_obs else 0)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n_obs,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Random sequence (class first when balance) and random start, or options={"seq": i, "t0": s}."""
        super().reset(seed=seed)
        if options and "seq" in options:
            self.i, self.t0 = int(options["seq"]), int(options.get("t0", 0))
        else:
            if self.balance:
                cls = [c for c in self.classes if self.by_class[c]]
                pool = self.by_class[cls[self.np_random.integers(len(cls))]]
                self.i = pool[self.np_random.integers(len(pool))]
            else:
                self.i = int(self.np_random.integers(len(self.sequences)))
            T = len(self.sequences[self.i]["features"])
            self.t0 = int(self.np_random.integers(max(T - self.horizon, 0) + 1))
        s = self.sequences[self.i]
        self.n = 1
        self.max_n = min(self.horizon, len(s["features"]) - self.t0)
        return self._obs(), {}

    def _obs(self):
        s = self.sequences[self.i]
        return observation(s["features"], self.t0, self.n, self.horizon, s["environment"],
                           s["agent"] if self.agent_obs else None)

    def step(self, action):
        s = self.sequences[self.i]
        info = {"seconds": self.n, "true": s["label"]}
        if action == 0:
            if self.n >= self.max_n:           # out of time (horizon or end of the drive): counts as wrong
                info["predicted"] = None
                return self._obs(), -self.wrong_cost[s["label"]], True, False, info
            self.n += 1
            return self._obs(), -self.wait_cost, False, False, info
        pred = self.classes[action - 1]
        info["predicted"] = pred
        reward = 1.0 if pred == s["label"] else -self.wrong_cost[s["label"]]
        return self._obs(), reward, True, False, info


def uah_sequences(root, min_speed_kmh=5.0, behaviors=("normal", "aggressive"), features=index_features):
    """Per second index features of every UAH trip with one of the given behaviours (stopped seconds dropped)."""
    from datasets.uah import find_trips, load_trip
    out = []
    for trip in find_trips(root):
        if trip["behavior"] not in behaviors:
            continue
        ps = load_trip(trip)
        ps = ps[ps["speed_kmh"] >= min_speed_kmh]
        f = features(ps["speed_kmh"], ps["accel"], ps["gap_m"], ps["wave_m"])
        out.append({"features": np.asarray(f, float), "label": trip["behavior"], "environment": trip["environment"],
                    "driver": trip["driver"], "trip": trip["trip"]})
    return out


def planted_sequences(paths, min_speed_kmh=5.0, min_seconds=10, features=index_features):
    """One sequence per planted SUMO vehicle (data/sumo_planted/<scenario>_s<seed>.per_second.csv.gz):
    true type as label, environment from scripts/planted_drivers.py ENVIRONMENT, seed and scenario kept."""
    import os
    import re
    import pandas as pd
    env_of = {"highway": "highway", "jam": "highway", "merge": "highway", "weather": "weather", "us101": "highway",
              "urban": "urban", "roundabout": "urban"}      # same mapping as scripts/planted_drivers.py
    out = []
    for path in paths:
        m = re.match(r"(?P<scenario>[a-z0-9]+)_(?P<density>[a-z]+)_s(?P<seed>\d+)\.per_second", os.path.basename(path))
        ps = pd.read_csv(path)
        ps = ps[ps["speed_kmh"] >= min_speed_kmh]
        for vid, g in ps.groupby("vehicle_id", sort=False):
            if len(g) < min_seconds:
                continue
            f = features(g["speed_kmh"], g["accel"], g["gap_m"], g["wave_m"])
            out.append({"features": np.asarray(f, float), "label": g["label"].iloc[0],
                        "environment": env_of[m.group("scenario")], "driver": vid,
                        "trip": f"{m.group('scenario')}_{m.group('density')}_s{m.group('seed')}/{vid}",
                        "scenario": f"{m.group('scenario')}_{m.group('density')}", "seed": int(m.group("seed"))})
    return out
