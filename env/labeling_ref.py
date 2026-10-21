"""
Sequential labeling on cleaned data with the reference formula (Oct 2026 revision): configurable state, reward
and action space, so each can be compared on its own. env/labeling_env.py (the earlier environment) is unchanged.

Sequences come from data/clean/*.csv.gz (sequences_from_clean): per second reference terms (NaN = unknown),
car ahead flag (1 / 0 / NaN), label, environment, group (split unit), scenario.

State: any subset of STATE_FEATURES (all between about -1 and 1; unknown terms count 0 points):
  pts_<term>_run   weight x mean term since the start (points / 100 of the score), term = speed, accel2, prox2, wave
  pts_<term>_rec   same over the last 10 s
  score_run        reference score of the means since the start / 100 (unknown terms left out)
  score_rec        same, last 10 s
  prox_follow      weight x mean proximity term over the seconds with a car ahead only (the following clue)
  leader_share     share of seconds with a car seen ahead
  time             seconds watched / horizon
  env_highway, env_urban, env_weather   environment one hot
Actions: wait (1 s), optionally wait 5 s ("wait5"), then one label per class.
Reward: +1 correct; wrong = -cost[true][predicted] (default 1, "ordinal": 1 for neighbouring classes, 2 for
conservative vs aggressive); each waited second costs wait_cost + wait_growth x seconds already watched;
no label at the horizon = -1.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from model.aggressiveness_model import REF_WEIGHTS, reference_terms

ENVIRONMENTS = ("highway", "urban", "weather")
TERMS = ("speed", "accel2", "prox2", "wave")
RECENT_S = 10
STATE_FEATURES = ([f"pts_{t}_run" for t in TERMS] + [f"pts_{t}_rec" for t in TERMS]
                  + ["score_run", "score_rec", "prox_follow", "leader_share", "time"]
                  + [f"env_{e}" for e in ENVIRONMENTS])
ORDER = {"conservative": 0, "normal": 1, "aggressive": 2}


def sequences_from_clean(d, min_speed_kmh=5.0, min_seconds=10):
    """cleaned per second table -> list of sequences (dict: terms (T, 4), leader (T,), label, environment, ...)"""
    d = d[d["speed_kmh"] >= min_speed_kmh].sort_values(["vid", "t"])
    T = reference_terms(d["speed_kmh"], d["accel"], d["gap_m"], d["wave_m"], d["environment"].to_numpy())
    lead = d["leader"].to_numpy(float)
    out = []
    starts = np.flatnonzero(np.r_[True, d["vid"].to_numpy()[1:] != d["vid"].to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(d)]
    meta = d[["vid", "label", "environment", "group", "scenario"]].to_numpy()
    raw = d[["speed_kmh", "gap_m"]].to_numpy(float)
    for a, b in zip(starts, ends):
        if b - a < min_seconds:
            continue
        vid, label, env, group, scen = meta[a]
        out.append({"terms": T[a:b], "leader": lead[a:b], "label": label, "environment": env, "group": group,
                    "scenario": scen, "trip": str(vid), "speed_kmh": raw[a:b, 0], "gap_m": raw[a:b, 1]})
    return out


def _wmean(x):
    """mean over known values per column, NaN when none known"""
    k = ~np.isnan(x)
    n = k.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(k, x, 0.0).sum(axis=0) / np.maximum(n, 1), np.nan)


def state_vector(terms, leader, n, horizon, environment, features, weights=REF_WEIGHTS):
    seen, lead = terms[:n], leader[:n]
    w = np.asarray(weights, float)
    run, rec = _wmean(seen), _wmean(seen[-RECENT_S:])
    vals = {}
    for i, t in enumerate(TERMS):
        vals[f"pts_{t}_run"] = 0.0 if np.isnan(run[i]) else w[i] * run[i]
        vals[f"pts_{t}_rec"] = 0.0 if np.isnan(rec[i]) else w[i] * rec[i]
    for name, m in (("score_run", run), ("score_rec", rec)):
        k = ~np.isnan(m)
        vals[name] = float((w[k] * m[k]).sum() / w[k].sum()) if k.any() else 0.0
    fol = seen[lead == 1, 2]
    vals["prox_follow"] = float(w[2] * fol.mean()) if len(fol) and not np.all(np.isnan(fol)) else 0.0
    vals["leader_share"] = float(np.nansum(lead == 1) / n)
    vals["time"] = n / horizon
    for e in ENVIRONMENTS:
        vals[f"env_{e}"] = 1.0 if environment == e else 0.0
    return np.array([vals[f] for f in features], dtype=np.float32)


class LabelingRefEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sequences, classes=("conservative", "normal", "aggressive"), features=STATE_FEATURES,
                 horizon=60, wait_cost=0.002, wait_growth=0.0, cost="flat", wait5=False, balance=True):
        self.sequences = list(sequences)
        self.classes = tuple(classes)
        self.features = list(features)
        self.horizon, self.wait_cost, self.wait_growth = int(horizon), float(wait_cost), float(wait_growth)
        self.cost, self.wait5, self.balance = cost, bool(wait5), balance
        self.waits = (1, 5) if self.wait5 else (1,)
        self.by_class = {c: [i for i, s in enumerate(self.sequences) if s["label"] == c] for c in self.classes}
        self.action_space = spaces.Discrete(len(self.waits) + len(self.classes))
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(len(self.features),), dtype=np.float32)

    def wrong_cost(self, true, pred):
        if self.cost == "ordinal":
            return 2.0 if abs(ORDER[true] - ORDER[pred]) == 2 else 1.0
        return 1.0

    def reset(self, seed=None, options=None):
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
            T = len(self.sequences[self.i]["terms"])
            self.t0 = int(self.np_random.integers(max(T - self.horizon, 0) + 1))
        s = self.sequences[self.i]
        self.n = 1
        self.max_n = min(self.horizon, len(s["terms"]) - self.t0)
        return self._obs(), {}

    def _obs(self):
        s = self.sequences[self.i]
        return state_vector(s["terms"][self.t0:], s["leader"][self.t0:], self.n, self.horizon, s["environment"],
                            self.features)

    def step(self, action):
        s = self.sequences[self.i]
        info = {"seconds": self.n, "true": s["label"]}
        if action < len(self.waits):
            k = self.waits[action]
            if self.n >= self.max_n:
                info["predicted"] = None
                return self._obs(), -1.0, True, False, info
            steps = min(k, self.max_n - self.n)
            r = -sum(self.wait_cost + self.wait_growth * (self.n + j) for j in range(steps))
            self.n += steps
            return self._obs(), r, False, False, info
        pred = self.classes[action - len(self.waits)]
        info["predicted"] = pred
        r = 1.0 if pred == s["label"] else -self.wrong_cost(s["label"], pred)
        return self._obs(), r, True, False, info


def run_policy(model, env, n):
    preds, secs = [], []
    for i in range(n):
        obs, _ = env.reset(options={"seq": i, "t0": 0})
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(a))
        preds.append(info["predicted"])
        secs.append(info["seconds"])
    return preds, secs


def evaluate(seqs, preds, secs):
    d = pd.DataFrame({"true": [s["label"] for s in seqs], "pred": preds, "seconds": secs,
                      "scenario": [s["scenario"] for s in seqs], "group": [s["group"] for s in seqs]})
    acc = d.assign(ok=d["true"] == d["pred"]).groupby("true")["ok"].mean()
    return d, {"balanced_acc": float(acc.mean()), **{f"acc_{c}": float(v) for c, v in acc.items()},
               "seconds": float(d["seconds"].mean()), "decided_at_1s": float((d["seconds"] == 1).mean())}
