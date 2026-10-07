import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from fixtures import make_uah  # noqa: E402

from env.labeling_env import LabelingEnv, observation, reference_ai, uah_sequences  # noqa: E402


def seq(value, label, T=30, environment="highway"):
    return {"features": np.full((T, 4), value, float), "label": label, "environment": environment,
            "driver": "D1", "trip": label}


def test_reference_ai_matches_hand_value():
    # 0.5*0.2 + 0.2*0.4 + 0.8*0.3 + 0.4*0.1 = 0.1 + 0.08 + 0.24 + 0.04 = 0.46
    assert abs(reference_ai([0.2, 0.4, 0.3, 0.1]) - 0.46) < 1e-12
    assert reference_ai([1, 1, 1, 1]) == 1.0           # capped at 100


def test_observation_running_and_recent_means():
    f = np.zeros((40, 4))
    f[:20, 0] = 0.1
    f[20:, 0] = 0.5
    o = observation(f, 0, 30, 120, "urban")
    assert abs(o[0] - (20 * 0.1 + 10 * 0.5) / 30) < 1e-6       # running mean, seconds 0..29
    assert abs(o[4] - 0.5) < 1e-6                               # last 10 s: seconds 20..29
    assert abs(o[8] - 0.5 * o[0]) < 1e-6 and abs(o[9] - 0.25) < 1e-6
    assert abs(o[10] - 30 / 120) < 1e-6
    assert list(o[11:]) == [0.0, 1.0]


def test_rewards_wait_correct_wrong_and_timeout():
    env = LabelingEnv([seq(0.1, "normal"), seq(0.6, "aggressive")], horizon=5, wait_cost=0.01,
                      wrong_cost={"aggressive": 2.0})
    env.reset(options={"seq": 1, "t0": 0})
    _, r, done, _, info = env.step(0)
    assert (r, done, info["seconds"]) == (-0.01, False, 1)
    _, r, done, _, info = env.step(2)                           # aggressive: correct
    assert (r, done, info["predicted"]) == (1.0, True, "aggressive")
    env.reset(options={"seq": 1, "t0": 0})
    _, r, done, _, _ = env.step(1)                              # missed aggressive driver costs 2
    assert (r, done) == (-2.0, True)
    env.reset(options={"seq": 0, "t0": 0})
    total, done, steps = 0.0, False, 0
    while not done:
        _, r, done, _, info = env.step(0)
        total, steps = total + r, steps + 1
    assert steps == 5 and info["predicted"] is None             # 4 waits reveal seconds 2..5, the 5th times out
    assert abs(total - (-0.04 - 1.0)) < 1e-12


def test_episode_cannot_run_past_the_end_of_the_drive():
    env = LabelingEnv([seq(0.1, "normal", T=8)], horizon=120)
    env.reset(options={"seq": 0, "t0": 5})
    assert env.max_n == 3


def test_balanced_reset_and_spaces():
    seqs = [seq(0.1, "normal") for _ in range(9)] + [seq(0.6, "aggressive")]
    env = LabelingEnv(seqs, horizon=10)
    env.reset(seed=0)
    labels = []
    for _ in range(400):
        env.reset()
        labels.append(env.sequences[env.i]["label"])
    assert 0.4 < np.mean(np.array(labels) == "aggressive") < 0.6
    o, _ = env.reset()
    assert env.observation_space.contains(o) and env.action_space.n == 3


def test_uah_sequences_from_fixture(tmp_path):
    root = make_uah(str(tmp_path), seconds=60)
    seqs = uah_sequences(root)
    assert {s["label"] for s in seqs} == {"normal", "aggressive"}
    assert len(seqs) == 6 * 5                                   # per driver: motorway 2 + secondary 3, drowsy left out
    assert all(s["features"].shape[1] == 4 for s in seqs)


def test_agent_margin_in_observation():
    # weights 0.25 each, features (0.2, 0.4, 0.3, 0.1): score 100 * 0.25 * 1.0 = 25; threshold 20 -> (25 - 20) / 100
    f = np.tile([0.2, 0.4, 0.3, 0.1], (15, 1))
    o = observation(f, 0, 15, 120, "highway", {"weights": [0.25] * 4, "threshold": 20.0})
    assert len(o) == 15 and abs(o[13] - 0.05) < 1e-6 and abs(o[14] - 0.05) < 1e-6
    s = {**seq(0.1, "normal"), "agent": {"weights": [0.25] * 4, "threshold": 50.0}}
    env = LabelingEnv([s], agent_obs=True)
    o, _ = env.reset(options={"seq": 0, "t0": 0})
    assert env.observation_space.contains(o) and abs(o[13] - (10 - 50) / 100) < 1e-6
