import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.irl import A_GRID, candidate_features, desired_speed_kmh, fit, log_likelihood  # noqa: E402


def test_candidate_features_hand_values():
    f = candidate_features([20.0], [30.0], [20.0], [True])[0]
    k0, km4 = list(A_GRID).index(0.0), list(A_GRID).index(-4.0)
    # a = 0: progress 20 / 36.11, headway stays 1.5 s -> (1 - 0.5)^2, discomfort 0
    p = 20 / (130 / 3.6)
    assert np.allclose(f[k0], [p, p ** 2, 0.25, 0.0])
    # a = -4: smallest headway at 0.5 s: gap 30 + 10 - 9.5 = 30.5 m at 18 m/s = 1.6944 s
    assert np.isclose(f[km4, 2], (1 - 30.5 / 18 / 3) ** 2) and np.isclose(f[km4, 3], 16 / 9)
    # no leader: no risk
    assert (candidate_features([20.0], [0.0], [20.0], [False])[0][:, 2] == 0).all()


def test_fit_recovers_known_weights():
    rng = np.random.default_rng(0)
    n = 4000
    f = candidate_features(rng.uniform(10, 35, n), rng.uniform(5, 60, n), rng.uniform(10, 35, n), rng.random(n) < 0.8)
    theta = np.array([12.0, -8.0, -8.0, -3.0])      # desired speed 12 / 16 x 130 = 97.5 km/h
    logits = f @ theta
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(len(A_GRID), p=row) for row in p])
    est = fit(f, y)
    assert np.allclose(est, theta, atol=1.5), est
    assert abs(desired_speed_kmh(est) - 97.5) < 5
    assert log_likelihood(f, y, est) >= log_likelihood(f, y, theta)
