"""
Highway RL scenario.

Surrounding vehicles: IDMVehicle (IDM longitudinal + MOBIL lane changes).
Aggressiveness:       AgentAssessor  — noisy observations only (no simulation access).
Ground truth:         GroundTruthAssessor — simulation internals, used only for
                      accuracy reporting at the end, NEVER fed to the agent.
Reward:               unified_reward() shared with urban and weather scenarios.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from paths import FIG_DIR
import time

import gymnasium as gym
import highway_env          # noqa: F401 — registers highway-v0
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
    compare_and_report,
    match_obs_to_vehicles,
)
from agent.reward_function import unified_reward


# ── 1. POLICY NETWORK ────────────────────────────────────────────────────────
class HighwayPolicyNet(nn.Module):
    """
    State (4 features):
        ego_vx / 30        normalised forward speed
        ego_vy / 5         lateral velocity
        front_dist / 50    proximity to nearest forward vehicle
        near_aggr          max perceived aggressiveness of neighbours [0, 1]

    Actions: 0=LaneLeft  1=Idle  2=LaneRight  3=Faster  4=Slower
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
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
        # IDM longitudinal control + MOBIL lane-change decisions
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,   # absolute world coordinates
            "normalize": False, # raw metres and m/s
        },
    })
    return env


# ── 3. TRAINING ──────────────────────────────────────────────────────────────
def train(epochs: int = 100, render: bool = False):
    env    = _make_env(render)
    policy = HighwayPolicyNet()
    opt    = optim.Adam(policy.parameters(), lr=0.01)

    gt_assessor    = GroundTruthAssessor()
    agent_assessor = AgentAssessor(noise_scale=1.0)

    history_rewards: list[float] = []
    all_gt:          list[str]   = []    # accumulated ground-truth labels
    all_ag:          list[str]   = []    # accumulated agent-predicted labels

    # Entropy-based exploration
    entropy_beta  = 0.01
    initial_noise = 3.0
    min_noise     = 0.01
    decay_rate    = 0.02

    for epoch in range(1, epochs + 1):
        obs, _ = env.reset()
        gt_assessor.reset()
        agent_assessor.reset()
        done = truncated = False

        log_probs:    list = []
        rewards:      list = []
        action_probs_history: list = []

        ego = env.unwrapped.vehicle
        exploration_noise = min_noise + (initial_noise - min_noise) * np.exp(-decay_rate * epoch)

        while not (done or truncated):
            if render:
                time.sleep(0.04)

            # ── Aggressiveness assessment (agent perception only) ──────────
            ag_results = agent_assessor.assess_all(obs)
            near_aggr  = max((s for s, _ in ag_results.values()), default=0.0) / 100.0

            # ── Ground truth (for accuracy report — NOT used by agent) ─────
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

            state = torch.tensor(
                [ego_vx / 30.0, ego_vy / 5.0, front_dist / 50.0, near_aggr],
                dtype=torch.float32,
            )

            # ── Action selection ───────────────────────────────────────────
            probs = policy(state)
            action_probs_history.append(probs)
            noisy_probs = (probs + exploration_noise) / (1.0 + exploration_noise * 5)
            m      = torch.distributions.Categorical(probs=noisy_probs)
            action = m.sample()

            obs, _, done, truncated, info = env.step(action.item())

            # ── Reward ─────────────────────────────────────────────────────
            r = unified_reward(
                scenario   = "highway",
                ego_speed  = ego_vx,
                action_idx = action.item(),
                crashed    = info.get("crashed", False),
                context    = {"front_dist": front_dist, "nearby_aggressiveness": near_aggr},
            )
            log_probs.append(m.log_prob(action))
            rewards.append(r)

        # ── REINFORCE + entropy regularisation ────────────────────────────
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
                f"[Highway] Epoch {epoch:3d}/{epochs}  "
                f"AvgR(10): {np.mean(history_rewards[-10:]):6.1f}  "
                f"Noise: {exploration_noise:.3f}"
            )

    env.close()

    # ── Final accuracy report ───────────────────────────────────────────────
    n    = len(all_gt)
    hits = sum(g == a for g, a in zip(all_gt, all_ag))
    acc  = (hits / n * 100) if n else 0.0
    print(
        f"\n[Highway] Aggressiveness category accuracy: {acc:.1f}%  "
        f"({hits}/{n} observations matched)"
    )
    print("  Note: ~90-95% is expected — boundary cases diverge due to sensor noise.")

    _save_plots(history_rewards, all_gt, all_ag, "highway")
    return policy


def _save_plots(rewards, gt_labels, ag_labels, tag):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Highway Scenario", fontweight="bold")

    smoothed = pd.Series(rewards).rolling(10, min_periods=1).mean()
    axs[0].plot(rewards, color="#bdc3c7", alpha=0.4, label="Raw reward")
    axs[0].plot(smoothed, color="#2ecc71", lw=2, label="10-epoch avg")
    axs[0].set(title="Reward over Training", xlabel="Epoch", ylabel="Reward")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    cats   = ["Conservative", "Normal", "Aggressive"]
    c2i    = {c: i for i, c in enumerate(cats)}
    matrix = np.zeros((3, 3), dtype=int)
    for g, a in zip(gt_labels, ag_labels):
        if g in c2i and a in c2i:
            matrix[c2i[g], c2i[a]] += 1
    im = axs[1].imshow(matrix, cmap="Blues")
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
    # Set render=True to watch the agent drive
    train(epochs=50, render=False)
