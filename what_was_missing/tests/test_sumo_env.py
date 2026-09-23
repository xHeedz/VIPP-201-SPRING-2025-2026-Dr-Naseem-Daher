# Written by Claude (Anthropic), September 2026, as part of what_was_missing/. See ../README.md.
import pytest

sumo_env = pytest.importorskip("env.sumo_env")


@pytest.mark.parametrize("scenario", ["highway", "intersection", "low_friction"])
def test_scenario_runs(scenario):
    env = sumo_env.SumoEnv(scenario, max_steps=30, seed=3)
    obs, info = env.reset()
    assert len(obs) == 9
    for _ in range(30):
        obs, reward, term, trunc, info = env.step(1)
        for t in info["telemetry"].values():
            assert 0.0 <= t["ai"] <= 100.0
            assert 0.0 <= t["slip"] <= 1.0
        if term or trunc:
            break
    env.close()
    assert trunc or term


def test_low_friction_context_and_edges():
    env = sumo_env.SumoEnv("low_friction", max_steps=5)
    obs, _ = env.reset()
    assert obs[4:6] == [1.0, 0.0] and abs(obs[7] - 0.4) < 1e-9
    assert 'friction="0.4"' in open(env.cfg.replace("scenario.sumocfg", "net.edg.xml")).read()
    env.close()
