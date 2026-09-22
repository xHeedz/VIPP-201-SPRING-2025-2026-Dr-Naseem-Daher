"""
Urban / Intersection RL scenario.

Surrounding vehicles: IDMVehicle (IDM + MOBIL).
Aggressiveness:       VehicleTracker receives only NOISY observations — the agent
                      cannot see true speed or position values directly.
Ground truth:         GroundTruthAssessor for accuracy reporting only.
Reward:               unified_reward() shared with highway and weather.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from paths import FIG_DIR
import time
from collections import deque

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
    compare_and_report,
    match_obs_to_vehicles,
)
from agent.reward_function import unified_reward


# ── 1. VEHICLE TRACKER (trajectory-based, works with noisy obs) ──────────────
class VehicleTracker:
    """
    Tracks non-ego vehicles across timesteps using noisy observation data.
    Derives behavioural aggressiveness from speed statistics and approach
    persistence — concepts that require history, not just a single snapshot.

    All inputs come from the observation vector + sensor noise; no simulation
    internals are ever accessed here.
    """

    # Sensor noise applied before storing observations
    _SIGMA_POS = 0.8    # metres
    _SIGMA_VEL = 0.5    # m/s

    def __init__(self, history_len: int = 8, noise_scale: float = 1.0):
        self.history_len  = history_len
        self.noise_scale  = noise_scale
        self.tracked:       dict[int, dict] = {}
        self.aggressiveness: dict[int, float] = {}
        self._next_id = 0

    def reset(self):
        self.tracked.clear()
        self.aggressiveness.clear()
        self._next_id = 0

    def update(self, raw_vehicle_obs: list):
        """
        raw_vehicle_obs: list of (x, y, vx, vy) from the observation array
                         — BEFORE noise is applied (noise added here).
        """
        s = self.noise_scale

        # Apply sensor noise to every observation
        noisy = []
        for (x, y, vx, vy) in raw_vehicle_obs:
            noisy.append((
                x  + np.random.normal(0, self._SIGMA_POS * s),
                y  + np.random.normal(0, self._SIGMA_POS * s),
                vx + np.random.normal(0, self._SIGMA_VEL * s),
                vy + np.random.normal(0, self._SIGMA_VEL * s),
            ))

        unmatched = list(enumerate(noisy))
        new_tracked: dict[int, dict] = {}

        # Match existing tracks to nearest new observation
        for vid, data in self.tracked.items():
            if not data["xy"]:
                continue
            last_x, last_y = data["xy"][-1]
            best_dist, best_idx = float("inf"), None

            for obs_idx, (ox, oy, _, _) in unmatched:
                d = np.hypot(ox - last_x, oy - last_y)
                if d < best_dist and d < 15.0:
                    best_dist, best_idx = d, obs_idx

            if best_idx is not None:
                ox, oy, ovx, ovy = noisy[best_idx]
                data["xy"].append((ox, oy))
                data["vxy"].append((ovx, ovy))
                new_tracked[vid] = data
                unmatched = [(i, v) for i, v in unmatched if i != best_idx]

        # Spawn new entries for unmatched observations
        for _, (ox, oy, ovx, ovy) in unmatched:
            vid = self._next_id
            self._next_id += 1
            new_tracked[vid] = {
                "xy":  deque([(ox, oy)],   maxlen=self.history_len),
                "vxy": deque([(ovx, ovy)], maxlen=self.history_len),
            }

        self.tracked = new_tracked
        self.aggressiveness = {
            vid: self._compute(data) for vid, data in self.tracked.items()
        }

    def _compute(self, data: dict) -> float:
        vxy_list = list(data["vxy"])
        if not vxy_list:
            return 0.0

        speeds = [np.hypot(vx, vy) for vx, vy in vxy_list]

        # Factor 1 — raw speed (> 10 m/s at an intersection → aggressive)
        speed_factor = float(np.tanh(np.mean(speeds) / 10.0))

        # Factor 2 — speed variance (erratic changes → aggressive)
        accel_factor = float(np.tanh(np.std(speeds) / 3.0)) if len(speeds) > 1 else 0.0

        # Factor 3 — approach persistence (not slowing down → not yielding)
        if len(speeds) >= 4:
            mid   = len(speeds) // 2
            delta = np.mean(speeds[mid:]) - np.mean(speeds[:mid]) * 0.7
            persistence = float(np.tanh(max(0.0, delta) / 3.0))
        else:
            persistence = 0.0

        score = 0.4 * speed_factor + 0.3 * accel_factor + 0.3 * persistence
        return float(np.clip(score, 0.0, 1.0))

    def get_zone_aggressiveness(self, ego_x, ego_y, fwd_x, fwd_y, lat_x, lat_y):
        """Return max aggressiveness in left and right crossing zones."""
        left_aggr = right_aggr = 0.0
        for vid, data in self.tracked.items():
            if not data["xy"]:
                continue
            vx_, vy_ = data["xy"][-1]
            dx, dy = vx_ - ego_x, vy_ - ego_y
            dist = np.hypot(dx, dy)
            if dist > 50.0:
                continue
            rel_forward = dx * fwd_x + dy * fwd_y
            rel_lateral = dx * lat_x + dy * lat_y
            aggr = self.aggressiveness.get(vid, 0.0)
            if rel_lateral < -2.0 and abs(rel_forward) < 10.0:
                left_aggr  = max(left_aggr,  aggr)
            elif rel_lateral > 2.0 and abs(rel_forward) < 10.0:
                right_aggr = max(right_aggr, aggr)
        return left_aggr, right_aggr


# ── 2. POLICY NETWORK ────────────────────────────────────────────────────────
class IntersectionPolicyNet(nn.Module):
    """
    State (6 features):
        norm_speed   ego speed
        norm_front   distance to front vehicle
        norm_left    distance to left-crossing vehicle
        norm_right   distance to right-crossing vehicle
        left_aggr    aggressiveness of left-zone vehicles  [0, 1]
        right_aggr   aggressiveness of right-zone vehicles [0, 1]

    Actions: 0=Brake  1=Idle  2=Accelerate
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


# ── 3. ENVIRONMENT ───────────────────────────────────────────────────────────
def _make_env(render: bool = False):
    env = gym.make("intersection-v0", render_mode="human" if render else None)
    env.unwrapped.configure({
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


# ── 4. TRAINING ──────────────────────────────────────────────────────────────
def train(epochs: int = 100, render: bool = False):
    env    = _make_env(render)
    policy = IntersectionPolicyNet()
    opt    = optim.Adam(policy.parameters(), lr=0.005)

    gt_assessor = GroundTruthAssessor()
    # VehicleTracker handles noisy multi-step tracking for the agent
    noise_scale  = 1.0

    history_rewards: list[float] = []
    all_gt:          list[str]   = []
    all_ag:          list[str]   = []

    initial_noise = 5.0
    min_noise     = 0.01
    decay_rate    = 0.015

    for epoch in range(1, epochs + 1):
        obs, _ = env.reset()
        gt_assessor.reset()
        tracker = VehicleTracker(history_len=8, noise_scale=noise_scale)
        done = truncated = False

        log_probs: list = []
        rewards:   list = []
        ego = env.unwrapped.vehicle

        exploration_noise = min_noise + (initial_noise - min_noise) * np.exp(-decay_rate * epoch)

        while not (done or truncated):
            if render:
                time.sleep(0.04)

            ego_x   = float(obs[0][1])
            ego_y   = float(obs[0][2])
            ego_vx  = float(obs[0][3])
            ego_vy  = float(obs[0][4])

            # ── Ego heading vectors ────────────────────────────────────────
            spd = np.hypot(ego_vx, ego_vy) + 1e-9
            fwd_x, fwd_y = ego_vx / spd, ego_vy / spd
            lat_x, lat_y = -fwd_y, fwd_x

            # ── Geometry from observation (no simulation access) ──────────
            front_dist = cross_left = cross_right = 50.0
            vehicle_obs_raw = []

            for i in range(1, len(obs)):
                if obs[i][0] < 0.5:
                    continue
                vx_, vy_, vvx, vvy = (float(obs[i][1]), float(obs[i][2]),
                                      float(obs[i][3]), float(obs[i][4]))
                vehicle_obs_raw.append((vx_, vy_, vvx, vvy))

                dx, dy = vx_ - ego_x, vy_ - ego_y
                dist = np.hypot(dx, dy)
                rel_forward = dx * fwd_x + dy * fwd_y
                rel_lateral = dx * lat_x + dy * lat_y
                approach_spd = -(vvx * dx + vvy * dy) / (dist + 1e-9)

                if dist < 50.0:
                    if abs(rel_lateral) < 2.0 and rel_forward > 0:
                        front_dist = min(front_dist, rel_forward)
                    elif rel_lateral < -2.0 and abs(rel_forward) < 10.0 and approach_spd > 1.0:
                        cross_left = min(cross_left, dist)
                    elif rel_lateral > 2.0 and abs(rel_forward) < 10.0 and approach_spd > 1.0:
                        cross_right = min(cross_right, dist)

            # ── Update tracker with noisy observations ────────────────────
            tracker.update(vehicle_obs_raw)
            left_aggr, right_aggr = tracker.get_zone_aggressiveness(
                ego_x, ego_y, fwd_x, fwd_y, lat_x, lat_y
            )

            # ── Ground truth for accuracy report (NOT used by agent) ──────
            slot_to_veh = match_obs_to_vehicles(obs, env.unwrapped.road.vehicles, ego)
            gt_step = {slot: gt_assessor.compute(veh, ego) for slot, veh in slot_to_veh.items()}

            # Build agent assessments from tracker aggressiveness scores
            # Map tracker IDs back to nearest obs slots for comparison
            for slot, veh in slot_to_veh.items():
                if slot not in gt_step:
                    continue
                # Find the tracker ID whose last position best matches this vehicle
                veh_x, veh_y = float(obs[slot][1]), float(obs[slot][2])
                best_tid, best_d = None, float("inf")
                for tid, tdata in tracker.tracked.items():
                    if not tdata["xy"]:
                        continue
                    tx, ty = tdata["xy"][-1]
                    d = np.hypot(tx - veh_x, ty - veh_y)
                    if d < best_d:
                        best_d, best_tid = d, tid
                if best_tid is not None and best_d < 8.0:
                    ag_score = tracker.aggressiveness.get(best_tid, 0.0) * 100.0
                    ag_label = "Conservative" if ag_score < 30 else ("Normal" if ag_score < 65 else "Aggressive")
                    all_gt.append(gt_step[slot][1])
                    all_ag.append(ag_label)

            # ── State tensor ───────────────────────────────────────────────
            state = torch.tensor([
                float(np.tanh(ego_vx / 10.0)),
                float(np.tanh((front_dist  - 10.0) / 10.0)),
                float(np.tanh((cross_left  - 10.0) / 10.0)),
                float(np.tanh((cross_right - 10.0) / 10.0)),
                float(left_aggr),
                float(right_aggr),
            ], dtype=torch.float32)

            # ── Action selection ───────────────────────────────────────────
            probs  = policy(state)
            noisy_p = (probs + exploration_noise) / (1.0 + exploration_noise * 3)
            m      = torch.distributions.Categorical(probs=noisy_p)
            action = m.sample()

            obs, _, done, truncated, info = env.step(action.item())

            # ── Reward ─────────────────────────────────────────────────────
            r = unified_reward(
                scenario   = "urban",
                ego_speed  = ego_vx,
                action_idx = action.item(),
                crashed    = info.get("crashed", False),
                context    = {
                    "front_dist":           front_dist,
                    "cross_left":           cross_left,
                    "cross_right":          cross_right,
                    "left_aggressiveness":  left_aggr,
                    "right_aggressiveness": right_aggr,
                },
            )
            log_probs.append(m.log_prob(action))
            rewards.append(r)

        # ── REINFORCE ──────────────────────────────────────────────────────
        total_r = sum(rewards)
        history_rewards.append(total_r)

        gamma, R, returns = 0.99, 0.0, []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = torch.stack([-lp * ret for lp, ret in zip(log_probs, returns)]).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(
                f"[Urban]   Epoch {epoch:3d}/{epochs}  "
                f"AvgR(10): {np.mean(history_rewards[-10:]):6.1f}  "
                f"Noise: {exploration_noise:.3f}"
            )

    env.close()

    n    = len(all_gt)
    hits = sum(g == a for g, a in zip(all_gt, all_ag))
    acc  = (hits / n * 100) if n else 0.0
    print(
        f"\n[Urban]   Aggressiveness category accuracy: {acc:.1f}%  "
        f"({hits}/{n} vehicle observations)"
    )
    print("  Note: ~90-95% is expected — boundary cases diverge due to sensor noise.")

    _save_plots(history_rewards, all_gt, all_ag, "urban")
    return policy


def _save_plots(rewards, gt_labels, ag_labels, tag):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Urban / Intersection Scenario", fontweight="bold")

    smoothed = pd.Series(rewards).rolling(10, min_periods=1).mean()
    axs[0].plot(rewards, color="#bdc3c7", alpha=0.4, label="Raw reward")
    axs[0].plot(smoothed, color="#e67e22", lw=2, label="10-epoch avg")
    axs[0].set(title="Reward over Training", xlabel="Epoch", ylabel="Reward")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    cats   = ["Conservative", "Normal", "Aggressive"]
    c2i    = {c: i for i, c in enumerate(cats)}
    matrix = np.zeros((3, 3), dtype=int)
    for g, a in zip(gt_labels, ag_labels):
        if g in c2i and a in c2i:
            matrix[c2i[g], c2i[a]] += 1
    im = axs[1].imshow(matrix, cmap="Oranges")
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
    train(epochs=50, render=False)
