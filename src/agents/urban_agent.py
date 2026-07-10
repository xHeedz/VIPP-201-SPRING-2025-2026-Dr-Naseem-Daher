import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from collections import deque


# --- 1. VEHICLE TRACKER: Profiles surrounding drivers across timesteps ---
class VehicleTracker:
    def __init__(self, history_len=8):
        self.history_len = history_len
        self.tracked = {}        # id -> {'xy': deque, 'vxy': deque}
        self.aggressiveness = {} # id -> float [0, 1]
        self.next_id = 0

    def update(self, vehicle_observations):
        """
        vehicle_observations: list of (x, y, vx, vy) for each visible non-ego vehicle.
        Matches new observations to tracked vehicles by nearest position, then
        creates new entries for any unmatched vehicle.
        """
        unmatched = list(enumerate(vehicle_observations))
        new_tracked = {}

        # Match existing tracked vehicles to nearest new observation
        for vid, data in self.tracked.items():
            if not data['xy']:
                continue
            last_x, last_y = data['xy'][-1]
            best_dist, best_idx = float('inf'), None

            for obs_idx, (ox, oy, _, _) in unmatched:
                d = np.sqrt((ox - last_x) ** 2 + (oy - last_y) ** 2)
                if d < best_dist and d < 15.0:
                    best_dist, best_idx = d, obs_idx

            if best_idx is not None:
                ox, oy, ovx, ovy = vehicle_observations[best_idx]
                data['xy'].append((ox, oy))
                data['vxy'].append((ovx, ovy))
                new_tracked[vid] = data
                unmatched = [(i, v) for i, v in unmatched if i != best_idx]

        # Spawn new entries for unmatched observations
        for _, (ox, oy, ovx, ovy) in unmatched:
            vid = self.next_id
            self.next_id += 1
            new_tracked[vid] = {
                'xy':  deque([(ox, oy)],   maxlen=self.history_len),
                'vxy': deque([(ovx, ovy)], maxlen=self.history_len),
            }

        self.tracked = new_tracked

        # Recompute aggressiveness for every live vehicle
        self.aggressiveness = {
            vid: self._compute_aggressiveness(data)
            for vid, data in self.tracked.items()
        }

    def _compute_aggressiveness(self, data):
        vxy_list = list(data['vxy'])
        if not vxy_list:
            return 0.0

        speeds = [np.sqrt(vx ** 2 + vy ** 2) for vx, vy in vxy_list]

        # Factor 1 — raw speed (>10 m/s at an intersection is aggressive)
        speed_factor = float(np.tanh(np.mean(speeds) / 10.0))

        # Factor 2 — acceleration variance (erratic speed changes = aggressive)
        accel_factor = float(np.tanh(np.std(speeds) / 3.0)) if len(speeds) > 1 else 0.0

        # Factor 3 — persistence (not slowing down as they get close = not yielding)
        if len(speeds) >= 4:
            mid = len(speeds) // 2
            delta = np.mean(speeds[mid:]) - np.mean(speeds[:mid]) * 0.7
            persistence = float(np.tanh(max(0.0, delta) / 3.0))
        else:
            persistence = 0.0

        score = 0.4 * speed_factor + 0.3 * accel_factor + 0.3 * persistence
        return float(np.clip(score, 0.0, 1.0))

    def get_zone_aggressiveness(self, ego_x, ego_y, fwd_x, fwd_y, lat_x, lat_y):
        """Return the max aggressiveness of vehicles in the left-cross and right-cross zones."""
        left_aggr = right_aggr = 0.0

        for vid, data in self.tracked.items():
            if not data['xy']:
                continue
            vx, vy = data['xy'][-1]
            dx, dy = vx - ego_x, vy - ego_y
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > 50.0:
                continue

            rel_forward = dx * fwd_x + dy * fwd_y
            rel_lateral = dx * lat_x + dy * lat_y
            aggr = self.aggressiveness.get(vid, 0.0)

            if rel_lateral < -2.0 and abs(rel_forward) < 10.0:
                left_aggr = max(left_aggr, aggr)
            elif rel_lateral > 2.0 and abs(rel_forward) < 10.0:
                right_aggr = max(right_aggr, aggr)

        return left_aggr, right_aggr


# --- 2. THE URBAN AI BRAIN (now takes 6 inputs) ---
class IntersectionAI(nn.Module):
    def __init__(self, input_size=6, num_actions=3):
        super(IntersectionAI, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        # Action Space: 0: Brake, 1: Idle, 2: Accelerate
        self.fc3 = nn.Linear(32, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


# --- 3. THE TRAINING LOOP ---
def train_urban_agent():
    print("--- Booting OOD Intersection Environment (with Aggressiveness Tracking) ---")

    env = gym.make("intersection-v0", render_mode="human")
    env.unwrapped.configure({
        "duration": 40,
        "simulation_frequency": 15,
        "policy_frequency": 5,
    })

    agent = IntersectionAI()
    optimizer = optim.Adam(agent.parameters(), lr=0.005)

    epochs = 5
    history_rewards = []

    # Exploration parameters
    initial_noise = 5.0
    min_noise = 0.01
    decay_rate = 0.015

    for epoch in range(1, epochs + 1):
        obs, info = env.reset()
        done = truncated = False

        log_probs = []
        rewards = []

        tracker = VehicleTracker(history_len=8)  # fresh tracker per episode

        exploration_noise = min_noise + (initial_noise - min_noise) * np.exp(-decay_rate * epoch)

        while not (done or truncated):
            time.sleep(0.04)

            ego_x, ego_y, ego_vx, ego_vy = obs[0][1], obs[0][2], obs[0][3], obs[0][4]

            # --- EGO-CENTRIC HEADING VECTORS ---
            speed = np.sqrt(ego_vx ** 2 + ego_vy ** 2) + 1e-9
            fwd_x, fwd_y = ego_vx / speed, ego_vy / speed
            lat_x, lat_y = -fwd_y, fwd_x

            # --- RADAR: geometric distances + collect observations for tracker ---
            front_dist = cross_left = cross_right = 50.0
            vehicle_obs = []

            for i in range(1, len(obs)):
                presence = obs[i][0]
                if presence < 0.5:
                    continue
                v_x, v_y, v_vx, v_vy = obs[i][1], obs[i][2], obs[i][3], obs[i][4]
                vehicle_obs.append((v_x, v_y, v_vx, v_vy))

                dx, dy = v_x - ego_x, v_y - ego_y
                rel_forward = dx * fwd_x + dy * fwd_y
                rel_lateral = dx * lat_x + dy * lat_y
                dist = np.sqrt(dx ** 2 + dy ** 2)

                # Only count a crossing vehicle as a threat if it's actively
                # closing in on the ego's path (filters out vehicles already past)
                approach_speed = -(v_vx * dx + v_vy * dy) / (dist + 1e-9)
                is_approaching = approach_speed > 1.0

                if dist < 50.0:
                    if abs(rel_lateral) < 2.0 and rel_forward > 0:
                        front_dist = min(front_dist, rel_forward)
                    elif rel_lateral < -2.0 and abs(rel_forward) < 10.0 and is_approaching:
                        cross_left = min(cross_left, dist)
                    elif rel_lateral > 2.0 and abs(rel_forward) < 10.0 and is_approaching:
                        cross_right = min(cross_right, dist)

            # --- UPDATE TRACKER and get per-zone aggressiveness ---
            tracker.update(vehicle_obs)
            left_aggr, right_aggr = tracker.get_zone_aggressiveness(
                ego_x, ego_y, fwd_x, fwd_y, lat_x, lat_y
            )

            # --- STATE TENSOR (6 features) ---
            norm_speed  = np.tanh(ego_vx / 10.0)
            norm_front  = np.tanh((front_dist - 10.0) / 10.0)
            norm_left   = np.tanh((cross_left  - 10.0) / 10.0)
            norm_right  = np.tanh((cross_right - 10.0) / 10.0)
            # aggressiveness is already in [0, 1], no extra scaling needed

            state_tensor = torch.tensor(
                [norm_speed, norm_front, norm_left, norm_right, left_aggr, right_aggr],
                dtype=torch.float32
            )

            action_probs = agent(state_tensor)
            m = torch.distributions.Categorical(
                probs=(action_probs + exploration_noise) / (1.0 + exploration_noise * 3)
            )
            action = m.sample()
            log_prob = m.log_prob(action)

            obs, env_reward, done, truncated, info = env.step(action.item())

            # --- REWARD SHAPING (aggressiveness-aware) ---
            custom_reward = 0.0

            if info.get('crashed', False):
                custom_reward -= 50.0

            else:
                # 1. LOITERING TAX — penalise parking on a clear road
                if ego_vx < 3.0 and cross_left > 20.0 and cross_right > 20.0 and front_dist > 15.0:
                    custom_reward -= 3.0

                # Aggressiveness-scaled threshold: calm driver = 8m, very aggressive = 15m
                left_threshold  = 8.0 + 7.0 * left_aggr
                right_threshold = 8.0 + 7.0 * right_aggr
                cross_threat = cross_left < left_threshold or cross_right < right_threshold

                if cross_threat:
                    threat_level = max(
                        left_aggr  if cross_left  < left_threshold  else 0.0,
                        right_aggr if cross_right < right_threshold else 0.0,
                    )
                    yield_bonus  = 3.0 + 2.0 * threat_level
                    run_penalty  = 5.0 + 5.0 * threat_level

                    if action.item() == 0:
                        if ego_vx > 1.5:            # reward yielding only if still moving
                            custom_reward += yield_bonus
                        else:
                            custom_reward -= 1.5    # penalise staying frozen at intersection
                    elif action.item() == 2:
                        custom_reward -= run_penalty

                else:
                    # 3. PROGRESS — clear intersection, reward speed
                    if front_dist > 20.0:
                        if action.item() == 2:
                            custom_reward += 1.5
                        custom_reward += ego_vx / 10.0

            log_probs.append(log_prob)
            rewards.append(custom_reward)

        total_reward = sum(rewards)
        history_rewards.append(total_reward)

        # --- REINFORCE BACKPROPAGATION ---
        gamma = 0.99
        returns, R = [], 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean()

        policy_loss = torch.stack([-lp * R for lp, R in zip(log_probs, returns)]).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            avg = np.mean(history_rewards[-10:])
            print(f"Epoch {epoch:3d}/{epochs} | Avg Reward: {avg:6.2f} | Noise: {exploration_noise:.3f}")

    env.close()
    print("\n[SYSTEM] Urban intersection run complete.")


if __name__ == "__main__":
    train_urban_agent()
