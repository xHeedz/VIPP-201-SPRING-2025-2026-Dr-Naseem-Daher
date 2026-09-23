# Written by Claude (Anthropic), September 2026, as part of what_was_missing/. See ../README.md.
from agent.social_latency import LagrangianSocialLatency, SocialLatencyReward


def test_no_neighbors_means_no_penalty():
    r = SocialLatencyReward()
    reward, harm = r(40.0, (0, 0), [])
    assert harm == 0.0 and abs(reward - 0.6) < 1e-9


def test_rising_neighbors_are_penalized_and_falling_are_not():
    r = SocialLatencyReward(lam=2.0, baseline_tau_steps=1e9)   # frozen baseline
    r(20.0, (0, 0), [(10, 0, 30.0)])
    reward, harm = r(20.0, (0, 0), [(10, 0, 40.0)])
    assert abs(harm - 10.0) < 1e-6 and abs(reward - (0.8 - 0.2)) < 1e-6
    _, harm = r(20.0, (0, 0), [(10, 0, 10.0)])
    assert harm == 0.0


def test_radius_excludes_far_vehicles():
    r = SocialLatencyReward(radius_m=50.0)
    assert r.neighbor_mean_ai((0, 0), [(100, 0, 90.0), (10, 0, 20.0)]) == 20.0


def test_lagrangian_moves_lambda_toward_constraint():
    r = LagrangianSocialLatency(tau=2.0, eta=0.5, lam_init=1.0)
    r.episode_harm = [6.0, 6.0]
    lam, _ = r.end_episode()
    assert lam == 3.0                      # harm above budget: price goes up
    r.episode_harm = [0.0]
    lam, _ = r.end_episode()
    assert lam == 2.0                      # harm below budget: price goes down
