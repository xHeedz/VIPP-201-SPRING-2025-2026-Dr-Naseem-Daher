"""
SUMO Intersection RL Agent with Aggressiveness Tracking
Ports scripts/highway_env/urban_agent.py from highway-env to SUMO TraCI.

Usage:
    python scripts/sumo_urban_agent.py           # GUI mode with visualization (default)
    python scripts/sumo_urban_agent.py --no-gui  # headless / fast batch training

Run build_networks.py first to generate intersection.net.xml.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR, SUMO_DIR, sumo_binary_name
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# --- Link back to model.py (AggressivenessModel is the shared scoring engine) ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AggressivenessModel

# --- SUMO / TraCI setup ---
SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    r"C:\Program Files (x86)\Eclipse\Sumo",
)
if not os.path.isdir(SUMO_HOME):
    print(f"ERROR: SUMO_HOME not found at '{SUMO_HOME}'.")
    print("Set the SUMO_HOME environment variable to your SUMO installation folder.")
    sys.exit(1)

sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci

BASE     = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(SUMO_DIR, "intersection", "intersection.sumocfg")
EGO_ID   = "ego"

POLICY_FREQ = 3     # 0.1 s × 3 = 0.3 s between decisions (~3.3 Hz)


# ============================================================
# NPC AGGRESSIVENESS DATA COLLECTOR  (linked back to model.py)
# ============================================================
class NPCDataCollector:
    """
    Collects real-time telemetry from every NPC in the SUMO scene and computes
    the AI Aggressiveness Index using the exact same AggressivenessModel from
    model.py — the same weights, normalisation, and scoring formula used across
    all other files in this project.

    Ground-truth linkage:
        SUMO sensor data  →  raw metrics  →  model.py normalise()
        →  weighted squared/linear terms  →  get_ai_score()
        →  Conservative / Normal / Aggressive label

    Call update() every policy step; call save_csv() after training.
    Pass print_breakdown=True to see every intermediate calculation printed.
    """

    def __init__(self):
        self.model   = AggressivenessModel()          # same as model.py
        self.history = {}    # vid -> {'prev_speed_ms': float, 'ys': deque}
        self.records = []    # accumulates rows for the output CSV

    # ----------------------------------------------------------
    def update(self, epoch, step, ego_x, ego_y, print_breakdown=False):
        for vid in traci.vehicle.getIDList():
            if vid == EGO_ID:
                continue

            speed_ms  = traci.vehicle.getSpeed(vid)
            speed_kmh = speed_ms * 3.6                       # SUMO → km/h
            vx, vy    = traci.vehicle.getPosition(vid)
            prox_m    = float(np.sqrt((vx - ego_x) ** 2 + (vy - ego_y) ** 2))

            if vid not in self.history:
                self.history[vid] = {
                    'prev_speed_ms': speed_ms,
                    'ys': deque([vy], maxlen=30),
                }
                accel_ms2 = 0.0
            else:
                h = self.history[vid]
                accel_ms2          = (speed_ms - h['prev_speed_ms']) / 0.1   # Δv / Δt (0.1 s step)
                h['prev_speed_ms'] = speed_ms
                h['ys'].append(vy)

            ys     = list(self.history[vid]['ys'])
            wave_m = float(np.std(ys)) if len(ys) > 3 else 0.0

            ai_score, label = self.model.get_ai_score(speed_kmh, accel_ms2, prox_m, wave_m)

            self.records.append({
                'epoch':      epoch,
                'step':       step,
                'vehicle_id': vid,
                'speed_kmh':  round(speed_kmh, 2),
                'accel_ms2':  round(accel_ms2,  3),
                'prox_m':     round(prox_m,     2),
                'wave_m':     round(wave_m,     3),
                'ai_score':   round(ai_score,   2),
                'category':   label,
            })

            if print_breakdown:
                self._print_breakdown(vid, speed_kmh, accel_ms2, prox_m, wave_m,
                                      ai_score, label)

    # ----------------------------------------------------------
    def _print_breakdown(self, vid, speed_kmh, accel_ms2, prox_m, wave_m,
                         ai_score, label):
        """
        Full hand-calculation trace — mirrors model.py line-by-line so you
        can verify exactly how the final score is produced.
        """
        m = self.model

        # Step 1 – Normalise (same as model.py AggressivenessModel.normalize)
        n_speed = min(speed_kmh / 150.0, 1.0)
        n_accel = min(abs(accel_ms2) / 5.0, 1.0)
        n_prox  = (1.0 - prox_m / 50.0) if 0 < prox_m <= 50 else 0.0
        n_wave  = min(abs(wave_m) / 1.5, 1.0)

        # Step 2 – Weighted contributions (same as get_ai_score)
        c_speed = n_speed ** 2 * m.w_speed
        c_accel = n_accel      * m.w_accel
        c_prox  = n_prox  ** 2 * m.w_prox
        c_wave  = n_wave       * m.w_wave
        raw_sum = c_speed + c_accel + c_prox + c_wave

        print(f"\n  ╔══ NPC Aggressiveness: {vid} ══╗")
        print(f"  │  Raw inputs  : speed={speed_kmh:.1f} km/h  "
              f"accel={accel_ms2:+.3f} m/s²  prox={prox_m:.1f} m  wave={wave_m:.3f} m")
        print(f"  │  Normalise")
        print(f"  │    n_speed = min({speed_kmh:.1f}/150, 1)              = {n_speed:.4f}")
        print(f"  │    n_accel = min(|{accel_ms2:.3f}|/5, 1)             = {n_accel:.4f}")
        if 0 < prox_m <= 50:
            print(f"  │    n_prox  = 1 - ({prox_m:.1f}/50)                  = {n_prox:.4f}  [closer = higher risk]")
        else:
            print(f"  │    n_prox  = 0.0  [vehicle outside 50 m window]")
        print(f"  │    n_wave  = min({wave_m:.3f}/1.5, 1)               = {n_wave:.4f}")
        print(f"  │  Weighted score (model.py get_ai_score formula)")
        print(f"  │    speed²×w_speed : {n_speed:.4f}² × {m.w_speed} = {c_speed:.5f}")
        print(f"  │    accel ×w_accel : {n_accel:.4f}  × {m.w_accel} = {c_accel:.5f}")
        print(f"  │    prox² ×w_prox  : {n_prox:.4f}² × {m.w_prox} = {c_prox:.5f}")
        print(f"  │    wave  ×w_wave  : {n_wave:.4f}  × {m.w_wave} = {c_wave:.5f}")
        print(f"  │  Sum = {c_speed:.5f}+{c_accel:.5f}+{c_prox:.5f}+{c_wave:.5f} = {raw_sum:.5f}")
        print(f"  │  AI Score = min({raw_sum:.5f}×100, 100) = {ai_score:.2f}")
        print(f"  ╚══ Label: [{label}]  (< 35 Conservative | 35-70 Normal | ≥ 70 Aggressive) ══╝")

    # ----------------------------------------------------------
    def epoch_summary(self, epoch):
        rows = [r for r in self.records if r['epoch'] == epoch]
        if not rows:
            return
        scores = [r['ai_score'] for r in rows]
        cats   = [r['category'] for r in rows]
        print(f"  [NPC Data] epoch={epoch:3d}  obs={len(rows):4d}  "
              f"avg_AI={np.mean(scores):5.1f}  "
              f"C:{cats.count('Conservative'):3d}  "
              f"N:{cats.count('Normal'):3d}  "
              f"A:{cats.count('Aggressive'):3d}")

    # ----------------------------------------------------------
    def save_csv(self, filename=os.path.join(DATA_DIR, "sumo_npc_aggressiveness.csv")):
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        df.to_csv(filename, index=False)
        print(f"\n[DATA] {len(self.records):,} NPC observations saved → {filename}")
        print(f"       Columns: {list(df.columns)}")
        print(f"       Score range: {df['ai_score'].min():.1f} – {df['ai_score'].max():.1f}")
        label_counts = df['category'].value_counts()
        for cat, cnt in label_counts.items():
            print(f"       {cat}: {cnt:,}  ({100*cnt/len(df):.1f}%)")


# ============================================================
# VEHICLE TRACKER  (feeds aggressiveness zones into the RL state)
# ============================================================
class VehicleTracker:
    def __init__(self, history_len=8):
        self.history_len = history_len
        self.tracked     = {}
        self.aggressiveness = {}
        self.next_id     = 0

    def update(self, vehicle_observations):
        unmatched   = list(enumerate(vehicle_observations))
        new_tracked = {}

        for vid, data in self.tracked.items():
            if not data["xy"]:
                continue
            last_x, last_y = data["xy"][-1]
            best_dist, best_idx = float("inf"), None

            for obs_idx, (ox, oy, _, _) in unmatched:
                d = np.sqrt((ox - last_x) ** 2 + (oy - last_y) ** 2)
                if d < best_dist and d < 15.0:
                    best_dist, best_idx = d, obs_idx

            if best_idx is not None:
                ox, oy, ovx, ovy = vehicle_observations[best_idx]
                data["xy"].append((ox, oy))
                data["vxy"].append((ovx, ovy))
                new_tracked[vid] = data
                unmatched = [(i, v) for i, v in unmatched if i != best_idx]

        for _, (ox, oy, ovx, ovy) in unmatched:
            vid = self.next_id
            self.next_id += 1
            new_tracked[vid] = {
                "xy":  deque([(ox, oy)],   maxlen=self.history_len),
                "vxy": deque([(ovx, ovy)], maxlen=self.history_len),
            }

        self.tracked = new_tracked
        self.aggressiveness = {
            vid: self._compute_aggressiveness(data)
            for vid, data in self.tracked.items()
        }

    def _compute_aggressiveness(self, data):
        vxy_list = list(data["vxy"])
        if not vxy_list:
            return 0.0
        speeds       = [np.sqrt(vx ** 2 + vy ** 2) for vx, vy in vxy_list]
        speed_factor = float(np.tanh(np.mean(speeds) / 10.0))
        accel_factor = float(np.tanh(np.std(speeds) / 3.0)) if len(speeds) > 1 else 0.0
        if len(speeds) >= 4:
            mid         = len(speeds) // 2
            delta       = np.mean(speeds[mid:]) - np.mean(speeds[:mid]) * 0.7
            persistence = float(np.tanh(max(0.0, delta) / 3.0))
        else:
            persistence = 0.0
        return float(np.clip(0.4 * speed_factor + 0.3 * accel_factor + 0.3 * persistence, 0.0, 1.0))

    def get_zone_aggressiveness(self, ego_x, ego_y, fwd_x, fwd_y, lat_x, lat_y):
        left_aggr = right_aggr = 0.0
        for vid, data in self.tracked.items():
            if not data["xy"]:
                continue
            vx, vy = data["xy"][-1]
            dx, dy = vx - ego_x, vy - ego_y
            dist   = np.sqrt(dx ** 2 + dy ** 2)
            if dist > 50.0:
                continue
            rel_forward = dx * fwd_x + dy * fwd_y
            rel_lateral = dx * lat_x + dy * lat_y
            aggr        = self.aggressiveness.get(vid, 0.0)
            if rel_lateral < -2.0 and abs(rel_forward) < 10.0:
                left_aggr  = max(left_aggr, aggr)
            elif rel_lateral > 2.0 and abs(rel_forward) < 10.0:
                right_aggr = max(right_aggr, aggr)
        return left_aggr, right_aggr


# ============================================================
# AI BRAIN  (6 inputs: speed, front_dist, cross×2, aggr×2 — 3 actions)
# ============================================================
class IntersectionAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


# ============================================================
# HELPERS
# ============================================================
def get_vxy(vid):
    """Convert SUMO angle + speed to (vx, vy) Cartesian."""
    speed     = traci.vehicle.getSpeed(vid)
    angle_rad = np.radians(traci.vehicle.getAngle(vid))
    return speed * np.sin(angle_rad), speed * np.cos(angle_rad)


def get_state_and_obs(tracker):
    ego_x, ego_y   = traci.vehicle.getPosition(EGO_ID)
    ego_vx, ego_vy = get_vxy(EGO_ID)

    speed  = np.sqrt(ego_vx ** 2 + ego_vy ** 2) + 1e-9
    fwd_x, fwd_y = ego_vx / speed, ego_vy / speed
    lat_x, lat_y = -fwd_y, fwd_x

    front_dist = cross_left = cross_right = 50.0
    vehicle_obs = []

    for vid in traci.vehicle.getIDList():
        if vid == EGO_ID:
            continue
        vx, vy     = traci.vehicle.getPosition(vid)
        v_vx, v_vy = get_vxy(vid)
        vehicle_obs.append((vx, vy, v_vx, v_vy))

        dx, dy      = vx - ego_x, vy - ego_y
        rel_forward = dx * fwd_x + dy * fwd_y
        rel_lateral = dx * lat_x + dy * lat_y
        dist        = np.sqrt(dx ** 2 + dy ** 2)

        approach_speed = -(v_vx * dx + v_vy * dy) / (dist + 1e-9)
        is_approaching = approach_speed > 1.0

        if dist < 50.0:
            if abs(rel_lateral) < 2.0 and rel_forward > 0:
                front_dist  = min(front_dist, rel_forward)
            elif rel_lateral < -2.0 and abs(rel_forward) < 10.0 and is_approaching:
                cross_left  = min(cross_left, dist)
            elif rel_lateral > 2.0 and abs(rel_forward) < 10.0 and is_approaching:
                cross_right = min(cross_right, dist)

    tracker.update(vehicle_obs)
    left_aggr, right_aggr = tracker.get_zone_aggressiveness(
        ego_x, ego_y, fwd_x, fwd_y, lat_x, lat_y
    )

    state = np.array([
        np.tanh(ego_vx / 10.0),
        np.tanh((front_dist  - 10.0) / 10.0),
        np.tanh((cross_left  - 10.0) / 10.0),
        np.tanh((cross_right - 10.0) / 10.0),
        left_aggr,
        right_aggr,
    ], dtype=np.float32)

    return state, ego_vx, front_dist, cross_left, cross_right, left_aggr, right_aggr, ego_x, ego_y


# ============================================================
# ACTION EXECUTION  (phantom-stop fix: full agent control via mode=0)
# ============================================================
def execute_action(action, ego_speed_target):
    if action == 0:    # Brake
        ego_speed_target = max(ego_speed_target - 3.0, 0.0)
    elif action == 2:  # Accelerate
        ego_speed_target = min(ego_speed_target + 3.0, 14.0)
    # action == 1: Idle — hold current target, but never let it drift to 0 silently
    traci.vehicle.setSpeed(EGO_ID, ego_speed_target)
    return ego_speed_target


# ============================================================
# REWARD FUNCTION
# ============================================================
def compute_reward(action, ego_vx, front_dist, cross_left, cross_right,
                   left_aggr, right_aggr, crashed):
    if crashed:
        return -50.0

    reward = 0.0

    if ego_vx < 3.0 and cross_left > 20.0 and cross_right > 20.0 and front_dist > 15.0:
        reward -= 3.0   # loitering tax

    left_threshold  = 8.0 + 7.0 * left_aggr
    right_threshold = 8.0 + 7.0 * right_aggr
    cross_threat    = cross_left < left_threshold or cross_right < right_threshold

    if cross_threat:
        threat_level = max(
            left_aggr  if cross_left  < left_threshold  else 0.0,
            right_aggr if cross_right < right_threshold else 0.0,
        )
        yield_bonus = 3.0 + 2.0 * threat_level
        run_penalty = 5.0 + 5.0 * threat_level
        if action == 0:
            reward += yield_bonus if ego_vx > 1.5 else -1.5
        elif action == 2:
            reward -= run_penalty
    else:
        if front_dist > 20.0:
            if action == 2:
                reward += 1.5
            reward += ego_vx / 10.0

    return reward


# ============================================================
# EPISODE RUNNER
# ============================================================
def run_episode(agent, exploration_noise, epoch, npc_collector, verbose_epoch):
    traci.load(["-c", CFG_PATH])

    for _ in range(100):
        traci.simulationStep()
        if EGO_ID in traci.vehicle.getIDList():
            break

    if EGO_ID not in traci.vehicle.getIDList():
        return [], []

    # --- Phantom-stop fix ---
    # Mode 0: SUMO applies NO safety constraints — agent has full speed authority.
    # This stops SUMO from ghost-braking the ego between setSpeed() calls.
    traci.vehicle.setSpeedMode(EGO_ID, 0)
    traci.vehicle.setLaneChangeMode(EGO_ID, 0)

    tracker = VehicleTracker(history_len=8)

    # Initialise to the depart speed from rou.xml (10 m/s), not getSpeed() which
    # may return 0 immediately after spawn — that was the phantom-stop trigger.
    ego_speed_target = 10.0

    log_probs, rewards = [], []
    step = 0

    while True:
        traci.simulationStep()
        step += 1

        crashed = EGO_ID in traci.simulation.getCollidingVehiclesIDList()
        alive   = EGO_ID in traci.vehicle.getIDList()

        if not alive or step > 1200:
            if crashed:
                rewards.append(-50.0)
                log_probs.append(log_probs[-1] if log_probs else torch.tensor(0.0))
            break

        if step % POLICY_FREQ == 0:
            (state_arr, ego_vx, front_dist, cross_left, cross_right,
             left_aggr, right_aggr, ego_x, ego_y) = get_state_and_obs(tracker)

            # Collect NPC aggressiveness; print full breakdown on the first policy
            # step of selected epochs so you can verify every intermediate value.
            show = verbose_epoch and (step == POLICY_FREQ)
            npc_collector.update(epoch, step, ego_x, ego_y,
                                 print_breakdown=show)

            state_tensor = torch.tensor(state_arr, dtype=torch.float32)
            action_probs = agent(state_tensor)

            noisy_probs = (action_probs + exploration_noise) / (1.0 + exploration_noise * 3)
            m           = torch.distributions.Categorical(probs=noisy_probs)
            action      = m.sample()
            log_prob    = m.log_prob(action)

            ego_speed_target = execute_action(action.item(), ego_speed_target)
            reward           = compute_reward(
                action.item(), ego_vx, front_dist, cross_left, cross_right,
                left_aggr, right_aggr, crashed
            )

            log_probs.append(log_prob)
            rewards.append(reward)

    return log_probs, rewards


# ============================================================
# TRAINING LOOP
# ============================================================
def train():
    print("=" * 60)
    print("  SUMO Intersection Agent — Aggressiveness Tracking")
    print("  Data collection linked to model.py AggressivenessModel")
    print("=" * 60)

    use_gui = "--no-gui" not in sys.argv          # GUI is ON by default
    binary  = sumo_binary_name(use_gui)
    sumo_bin = os.path.join(SUMO_HOME, "bin", binary)

    sumo_cmd = [sumo_bin, "-c", CFG_PATH]
    if use_gui:
        # --delay N  → N milliseconds between GUI frames (makes sim human-readable)
        sumo_cmd += ["--delay", "100"]

    traci.start(sumo_cmd)

    # Slow the GUI down via TraCI as well (belt-and-braces)
    if use_gui:
        try:
            traci.gui.setDelay("View #0", 100)
        except Exception:
            pass

    agent     = IntersectionAI()
    optimizer = optim.Adam(agent.parameters(), lr=0.005)

    epochs        = 200
    initial_noise = 5.0
    min_noise     = 0.01
    decay_rate    = 0.015
    history       = []

    npc_collector = NPCDataCollector()

    # Print ground-truth breakdown on epoch 1 and every 20 epochs after that
    verbose_epochs = {1} | {e for e in range(20, epochs + 1, 20)}

    print(f"\n[CONFIG] epochs={epochs}  GUI={'on' if use_gui else 'off'}")
    print(f"[CONFIG] Aggressiveness model weights from model.py:")
    print(f"         w_speed={npc_collector.model.w_speed}  w_accel={npc_collector.model.w_accel}")
    print(f"         w_prox={npc_collector.model.w_prox}   w_wave={npc_collector.model.w_wave}\n")

    for epoch in range(1, epochs + 1):
        noise = min_noise + (initial_noise - min_noise) * np.exp(-decay_rate * epoch)
        verbose = epoch in verbose_epochs

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  EPOCH {epoch}  — full NPC aggressiveness breakdown below")
            print(f"{'─'*60}")

        log_probs, rewards = run_episode(agent, noise, epoch, npc_collector, verbose)

        if not log_probs:
            continue

        total_reward = sum(rewards)
        history.append([epoch, total_reward])

        # REINFORCE backpropagation
        gamma, R, returns = 0.99, 0, []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = torch.stack(
            [-lp * R for lp, R in zip(log_probs, returns)]
        ).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:3d}/{epochs} | Reward: {total_reward:7.2f} | Noise: {noise:.3f}", end="")

        # Print NPC aggressiveness summary every epoch
        npc_collector.epoch_summary(epoch)

    traci.close()

    # ---- Save training history ----
    df_hist = pd.DataFrame(history, columns=["Epoch", "Reward"])
    df_hist.to_csv(os.path.join(DATA_DIR, "sumo_intersection_history.csv"), index=False)

    # ---- Save NPC aggressiveness dataset (linked to model.py) ----
    npc_collector.save_csv(os.path.join(DATA_DIR, "sumo_npc_aggressiveness.csv"))

    # ---- Learning curve plot ----
    plt.figure(figsize=(10, 5))
    smoothed = df_hist["Reward"].rolling(10, min_periods=1).mean()
    plt.plot(df_hist["Reward"], color="#bdc3c7", alpha=0.4, label="Raw Reward")
    plt.plot(smoothed,          color="#e74c3c", linewidth=2, label="10-Epoch Trend")
    plt.title("SUMO Intersection Agent — Learning Curve (200 epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "sumo_intersection_curve.png"))

    # ---- NPC aggressiveness distribution plot ----
    df_npc = pd.read_csv(os.path.join(DATA_DIR, "sumo_npc_aggressiveness.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_npc["ai_score"], bins=40, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].axvline(35, color="#f39c12", linestyle="--", label="Conservative/Normal boundary (35)")
    axes[0].axvline(70, color="#e74c3c", linestyle="--", label="Normal/Aggressive boundary (70)")
    axes[0].set_title("NPC AI Score Distribution\n(linked to model.py AggressivenessModel)")
    axes[0].set_xlabel("AI Score")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    cat_counts = df_npc["category"].value_counts()
    colors = {"Conservative": "#27ae60", "Normal": "#f39c12", "Aggressive": "#e74c3c"}
    axes[1].bar(cat_counts.index,
                cat_counts.values,
                color=[colors.get(c, "grey") for c in cat_counts.index],
                edgecolor="white")
    axes[1].set_title("NPC Aggressiveness Labels\nacross all 200 epochs")
    axes[1].set_ylabel("Observation Count")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Surrounding Vehicle Aggressiveness Data Collection", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "sumo_npc_aggressiveness_plot.png"))

    print("\n[SYSTEM] Done.")
    print("  sumo_intersection_curve.png")
    print("  sumo_npc_aggressiveness.csv")
    print("  sumo_npc_aggressiveness_plot.png")


if __name__ == "__main__":
    train()
