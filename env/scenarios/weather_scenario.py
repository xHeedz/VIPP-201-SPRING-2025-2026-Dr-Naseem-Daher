"""
Weather / Friction RL scenario.

Simulates wet or icy road conditions by modifying IDM vehicle behaviour
(longer time gaps, gentler accelerations) and increasing sensor noise in
the AgentAssessor (poor visibility).

Surrounding vehicles: IDMVehicle with weather-tuned instance parameters.
Aggressiveness:       AgentAssessor with elevated noise (noise_scale > 1).
Ground truth:         GroundTruthAssessor for accuracy reporting only.
Reward:               unified_reward() with weather_factor < 1.0.

Weather presets
---------------
'clear'     weather_factor=1.00  noise_scale=1.0
'rain'      weather_factor=0.70  noise_scale=1.5
'heavy_fog' weather_factor=0.55  noise_scale=2.0
'ice'       weather_factor=0.45  noise_scale=1.8
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from paths import FIG_DIR
import time

import gymnasium as gym
import highway_env          # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.aggressiveness_core import (
    AgentAssessor,
    GroundTruthAssessor,
    match_obs_to_vehicles,
)
from agent.reward_function import unified_reward


# ── WEATHER PRESETS ──────────────────────────────────────────────────────────
_PRESETS = {
    "clear":     {"weather_factor": 1.00, "noise_scale": 1.0,
                  "time_wanted": 1.5, "comfort_acc_max": 3.0, "comfort_acc_min": -5.0,
                  "distance_wanted": 5.0},
    "rain":      {"weather_factor": 0.70, "noise_scale": 1.5,
                  "time_wanted": 2.2, "comfort_acc_max": 2.0, "comfort_acc_min": -3.0,
                  "distance_wanted": 8.0},
    "heavy_fog": {"weather_factor": 0.55, "noise_scale": 2.0,
                  "time_wanted": 2.8, "comfort_acc_max": 1.5, "comfort_acc_min": -2.5,
                  "distance_wanted": 10.0},
    "ice":       {"weather_factor": 0.45, "noise_scale": 1.8,
                  "time_wanted": 3.0, "comfort_acc_max": 1.0, "comfort_acc_min": -1.5,
                  "distance_wanted": 12.0},
}


# ── 1. POLICY NETWORK ────────────────────────────────────────────────────────
class WeatherPolicyNet(nn.Module):
    """
    State (5 features):
        ego_vx / 20        speed (normalised for reduced target in weather)
        ego_vy / 5         lateral velocity
        front_dist / 50    proximity
        near_aggr          max nearby aggressiveness [0, 1]
        weather_factor     road condition signal [0.4, 1.0]

    Actions: 0=LaneLeft  1=Idle  2=LaneRight  3=Faster  4=Slower
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


# ── 2. ENVIRONMENT ───────────────────────────────────────────────────────────
def _make_env(render: bool = False):
    env = gym.make("highway-v0", render_mode="human" if render else None)
    env.unwrapped.configure({
        "vehicles_count": 15,
        "duration": 40,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,
            "normalize": False,
        },
    })
    return env


def _apply_weather_to_vehicles(env, ego_vehicle, preset: dict):
    """
    Modify IDM parameters on each non-ego vehicle instance to simulate
    cautious driving on wet / icy / foggy roads.
    IDMVehicle reads these via self.X, so instance-level attributes override
    the class defaults correctly.
    """
    for v in env.unwrapped.road.vehicles:
        if v is ego_vehicle:
            continue
        v.TIME_WANTED       = preset["time_wanted"]
        v.COMFORT_ACC_MAX   = preset["comfort_acc_max"]
        v.COMFORT_ACC_MIN   = preset["comfort_acc_min"]
        v.DISTANCE_WANTED   = preset["distance_wanted"]


# ── 3. TRAINING ──────────────────────────────────────────────────────────────
def train(epochs: int = 100, condition: str = "rain", render: bool = False):
    if condition not in _PRESETS:
        raise ValueError(f"Unknown condition '{condition}'. Choose from: {list(_PRESETS)}")

    preset         = _PRESETS[condition]
    wf             = preset["weather_factor"]
    noise_scale    = preset["noise_scale"]

    print(f"[Weather] Condition: {condition!r}  "
          f"weather_factor={wf:.2f}  sensor_noise_scale={noise_scale:.1f}")

    env    = _make_env(render)
    policy = WeatherPolicyNet()
    opt    = optim.Adam(policy.parameters(), lr=0.01)

    gt_assessor    = GroundTruthAssessor()
    agent_assessor = AgentAssessor(noise_scale=noise_scale)

    history_rewards: list[float] = []
    all_gt:          list[str]   = []
    all_ag:          list[str]   = []

    entropy_beta  = 0.01
    initial_noise = 3.0
    min_noise     = 0.01
    decay_rate    = 0.02

    for epoch in range(1, epochs + 1):
        obs, _ = env.reset()
        gt_assessor.reset()
        agent_assessor.reset()

        ego = env.unwrapped.vehicle
        _apply_weather_to_vehicles(env, ego, preset)

        done = truncated = False
        log_probs:   list = []
        rewards:     list = []
        action_probs_history: list = []

        exploration_noise = min_noise + (initial_noise - min_noise) * np.exp(-decay_rate * epoch)

        while not (done or truncated):
            if render:
                time.sleep(0.04)

            # ── Agent aggressiveness perception (noisy obs only) ───────────
            ag_results = agent_assessor.assess_all(obs)
            near_aggr  = max((s for s, _ in ag_results.values()), default=0.0) / 100.0

            # ── Ground truth (for report only — agent never sees this) ─────
            slot_to_veh = match_obs_to_vehicles(obs, env.unwrapped.road.vehicles, ego)
            gt_step = {slot: gt_assessor.compute(veh, ego) for slot, veh in slot_to_veh.items()}

            for slot in set(gt_step) & set(ag_results):
                all_gt.append(gt_step[slot][1])
                all_ag.append(ag_results[slot][1])

            # ── State features ─────────────────────────────────────────────
            ego_vx     = float(obs[0][3])
            ego_vy     = float(obs[0][4])
            ego_x      = float(obs[0][1])
            front_dist = 50.0
            for i in range(1, len(obs)):
                if obs[i][0] < 0.5:
                    continue
                rel_x = float(obs[i][1]) - ego_x
                if 0 < rel_x < front_dist:
                    front_dist = rel_x

            state = torch.tensor([
                ego_vx / 20.0,          # lower normalisation (reduced target speed)
                ego_vy / 5.0,
                front_dist / 50.0,
                near_aggr,
                wf,                     # weather condition is observable to the agent
            ], dtype=torch.float32)

            # ── Action selection ───────────────────────────────────────────
            probs  = policy(state)
            action_probs_history.append(probs)
            noisy_p = (probs + exploration_noise) / (1.0 + exploration_noise * 5)
            m      = torch.distributions.Categorical(probs=noisy_p)
            action = m.sample()

            obs, _, done, truncated, info = env.step(action.item())

            # ── Reward ─────────────────────────────────────────────────────
            r = unified_reward(
                scenario       = "weather",
                ego_speed      = ego_vx,
                action_idx     = action.item(),
                crashed        = info.get("crashed", False),
                context        = {"front_dist": front_dist, "nearby_aggressiveness": near_aggr},
                weather_factor = wf,
            )
            log_probs.append(m.log_prob(action))
            rewards.append(r)

        # ── REINFORCE + entropy ────────────────────────────────────────────
        total_r = sum(rewards)
        history_rewards.append(total_r)

        gamma, R, returns = 0.99, 0.0, []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = torch.stack([-lp * ret for lp, ret in zip(log_probs, returns)]).sum()
        entropy     = -torch.stack(
            [torch.sum(p * torch.log(p + 1e-9)) for p in action_probs_history]
        ).mean()
        total_loss  = policy_loss - entropy_beta * entropy

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(
                f"[Weather/{condition}] Epoch {epoch:3d}/{epochs}  "
                f"AvgR(10): {np.mean(history_rewards[-10:]):6.1f}  "
                f"Noise: {exploration_noise:.3f}"
            )

    env.close()

    n    = len(all_gt)
    hits = sum(g == a for g, a in zip(all_gt, all_ag))
    acc  = (hits / n * 100) if n else 0.0
    print(
        f"\n[Weather/{condition}] Aggressiveness accuracy: {acc:.1f}%  "
        f"({hits}/{n} observations)"
    )
    print(f"  Expected: 90-95% clear / 85-92% rain / 80-90% heavy fog+ice "
          f"(higher noise → more boundary misclassification)")

    _save_plots(history_rewards, all_gt, all_ag, f"weather_{condition}")
    return policy


def _save_plots(rewards, gt_labels, ag_labels, tag):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    label = tag.replace("weather_", "").replace("_", " ").title()
    fig.suptitle(f"Weather Scenario — {label}", fontweight="bold")

    smoothed = pd.Series(rewards).rolling(10, min_periods=1).mean()
    axs[0].plot(rewards, color="#bdc3c7", alpha=0.4, label="Raw reward")
    axs[0].plot(smoothed, color="#8e44ad", lw=2, label="10-epoch avg")
    axs[0].set(title="Reward over Training", xlabel="Epoch", ylabel="Reward")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    cats   = ["Conservative", "Normal", "Aggressive"]
    c2i    = {c: i for i, c in enumerate(cats)}
    matrix = np.zeros((3, 3), dtype=int)
    for g, a in zip(gt_labels, ag_labels):
        if g in c2i and a in c2i:
            matrix[c2i[g], c2i[a]] += 1
    im = axs[1].imshow(matrix, cmap="Purples")
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(cats, rotation=20, ha="right")
    axs[1].set_yticks(range(3)); axs[1].set_yticklabels(cats)
    axs[1].set(title="Agent vs Ground-Truth Category", xlabel="Agent", ylabel="Ground Truth")
    for i in range(3):
        for j in range(3):
            axs[1].text(j, i, matrix[i, j], ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=axs[1])

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{tag}_results.png"), dpi=150)
    print(f"[{tag}] Saved {tag}_results.png")
    plt.show()


if __name__ == "__main__":
    # Try: 'clear', 'rain', 'heavy_fog', 'ice'
    train(epochs=50, condition="rain", render=False)
