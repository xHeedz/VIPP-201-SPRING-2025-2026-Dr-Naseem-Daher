"""
SUMO Highway RL Agent with NPC Aggressiveness Data Collection
Ports rl_agent_26.py from highway-env to SUMO TraCI.

Usage:
    python sumo_highway_agent.py           # GUI mode with visualization (default)
    python sumo_highway_agent.py --no-gui  # headless / fast batch training

Run build_networks.py first to generate highway.net.xml.
"""
import os
import sys
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
CFG_PATH = os.path.join(BASE, "sumo", "highway", "highway.sumocfg")
EGO_ID   = "ego"

POLICY_FREQ = 4     # 0.1 s × 4 = 0.4 s between decisions (~5 Hz)


# ============================================================
# NPC AGGRESSIVENESS DATA COLLECTOR  (linked back to model.py)
# ============================================================
class NPCDataCollector:
    """
    Collects real-time telemetry from every NPC vehicle and computes the
    AI Aggressiveness Index using the exact same AggressivenessModel from
    model.py — the same weights, normalisation rules, and scoring formula
    used in agent.py, simulator.py, and finaltry_agent.py.

    Ground-truth linkage:
        SUMO TraCI sensor data  →  raw metrics (speed_kmh, accel_ms2, prox_m, wave_m)
        →  model.py normalize()  →  weighted squared/linear terms
        →  get_ai_score()  →  Conservative / Normal / Aggressive
    """

    def __init__(self):
        self.model   = AggressivenessModel()
        self.history = {}    # vid -> {'prev_speed_ms': float, 'ys': deque}
        self.records = []    # all rows accumulated for CSV

    # ----------------------------------------------------------
    def update(self, epoch, step, ego_x, ego_y, print_breakdown=False):
        for vid in traci.vehicle.getIDList():
            if vid == EGO_ID:
                continue

            speed_ms  = traci.vehicle.getSpeed(vid)
            speed_kmh = speed_ms * 3.6                        # m/s → km/h
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
                accel_ms2          = (speed_ms - h['prev_speed_ms']) / 0.1   # Δv / Δt
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
        can verify every intermediate step from raw sensor value to final label.
        """
        m = self.model

        # Step 1 – Normalise  (replicates model.py AggressivenessModel.normalize)
        n_speed = min(speed_kmh / 150.0, 1.0)
        n_accel = min(abs(accel_ms2) / 5.0, 1.0)
        n_prox  = (1.0 - prox_m / 50.0) if 0 < prox_m <= 50 else 0.0
        n_wave  = min(abs(wave_m) / 1.5, 1.0)

        # Step 2 – Weighted contributions  (replicates model.py get_ai_score)
        c_speed = n_speed ** 2 * m.w_speed
        c_accel = n_accel      * m.w_accel
        c_prox  = n_prox  ** 2 * m.w_prox
        c_wave  = n_wave       * m.w_wave
        raw_sum = c_speed + c_accel + c_prox + c_wave

        print(f"\n  ╔══ NPC Aggressiveness: {vid} ══╗")
        print(f"  │  Raw inputs  : speed={speed_kmh:.1f} km/h  "
              f"accel={accel_ms2:+.3f} m/s²  prox={prox_m:.1f} m  wave={wave_m:.3f} m")
        print(f"  │  Normalise   (model.py normalize)")
        print(f"  │    n_speed = min({speed_kmh:.1f}/150, 1)              = {n_speed:.4f}")
        print(f"  │    n_accel = min(|{accel_ms2:.3f}|/5, 1)             = {n_accel:.4f}")
        if 0 < prox_m <= 50:
            print(f"  │    n_prox  = 1 - ({prox_m:.1f}/50)                  = {n_prox:.4f}  [closer = higher risk]")
        else:
            print(f"  │    n_prox  = 0.0  [vehicle outside 50 m window]")
        print(f"  │    n_wave  = min({wave_m:.3f}/1.5, 1)               = {n_wave:.4f}")
        print(f"  │  Weights from model.py:")
        print(f"  │    w_speed={m.w_speed}  w_accel={m.w_accel}  w_prox={m.w_prox}  w_wave={m.w_wave}")
        print(f"  │  Weighted score  (model.py get_ai_score formula)")
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
    def save_csv(self, filename="sumo_npc_aggressiveness.csv"):
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
# AI BRAIN  (4 inputs, 5 actions — mirrors rl_agent_26.py)
# ============================================================
class SmartDriverAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=-1)


# ============================================================
# STATE EXTRACTION
# ============================================================
def get_state():
    ego_speed    = traci.vehicle.getSpeed(EGO_ID)
    ego_x, ego_y = traci.vehicle.getPosition(EGO_ID)

    prox          = 50.0
    v_ahead_speed = 20.0
    for vid in traci.vehicle.getIDList():
        if vid == EGO_ID:
            continue
        vx, vy = traci.vehicle.getPosition(vid)
        dx     = vx - ego_x
        if 0 < dx < prox and abs(vy - ego_y) < 4.0:
            prox          = dx
            v_ahead_speed = traci.vehicle.getSpeed(vid)

    norm_speed = float(np.tanh((ego_speed - 20.0) / 10.0))
    norm_accel = 0.0   # placeholder (no accelerometer in this config)
    norm_prox  = float(np.tanh((prox - 20.0) / 15.0))
    norm_wave  = float(np.tanh(ego_y / 2.0))

    return (np.array([norm_speed, norm_accel, norm_prox, norm_wave], dtype=np.float32),
            ego_speed, prox, v_ahead_speed, ego_x, ego_y)


# ============================================================
# ACTION EXECUTION  (phantom-stop fix)
# ============================================================
def execute_action(action, ego_speed_target):
    lane      = traci.vehicle.getLaneIndex(EGO_ID)
    road_id   = traci.vehicle.getRoadID(EGO_ID)
    num_lanes = traci.edge.getLaneNumber(road_id) if not road_id.startswith(":") else 4

    if action == 0 and lane > 0:
        traci.vehicle.changeLane(EGO_ID, lane - 1, 2.0)
    elif action == 2 and lane < num_lanes - 1:
        traci.vehicle.changeLane(EGO_ID, lane + 1, 2.0)
    elif action == 3:   # Faster
        ego_speed_target = min(ego_speed_target + 2.0, 30.0)
    elif action == 4:   # Slower — hard floor at 5 m/s; we never stop on a highway
        ego_speed_target = max(ego_speed_target - 2.0, 5.0)
    # action == 1: IDLE — no change to target

    traci.vehicle.setSpeed(EGO_ID, ego_speed_target)
    return ego_speed_target


# ============================================================
# REWARD FUNCTION
# ============================================================
def compute_reward(action, ego_speed, prox, v_ahead_speed, crashed):
    if crashed:
        return -30.0

    reward = 1.0
    if ego_speed > 22.0:
        reward += 1.5
    elif ego_speed < 15.0:
        reward -= 1.0

    rel_velocity = ego_speed - v_ahead_speed
    if rel_velocity > 0.5 and prox < 40.0:
        ttc     = prox / (rel_velocity + 1e-9)
        reward -= np.exp(-ttc / 2.5) * 4.0

    if action in (0, 2):   # steering tax
        reward -= 0.5

    return reward


# ============================================================
# EPISODE RUNNER
# ============================================================
def run_episode(agent, epoch, exploration_noise, npc_collector, verbose_epoch):
    traci.load(["-c", CFG_PATH])

    for _ in range(50):
        traci.simulationStep()
        if EGO_ID in traci.vehicle.getIDList():
            break

    if EGO_ID not in traci.vehicle.getIDList():
        return [], []

    # --- Phantom-stop fix ---
    # Mode 0: SUMO applies NO safety constraints — the agent has full speed
    # authority. This prevents SUMO from ghost-braking the ego between policy
    # steps and stops the vehicle from drifting to 0 due to safety overrides.
    traci.vehicle.setSpeedMode(EGO_ID, 0)
    traci.vehicle.setLaneChangeMode(EGO_ID, 0)

    # Initialise to the depart speed in rou.xml (20 m/s).
    # Using getSpeed() here sometimes returns 0 right after spawn,
    # which is the root cause of phantom stops in the first place.
    ego_speed_target = 20.0

    log_probs, rewards = [], []
    step = 0

    while True:
        traci.simulationStep()
        step += 1

        crashed = EGO_ID in traci.simulation.getCollidingVehiclesIDList()
        alive   = EGO_ID in traci.vehicle.getIDList()

        if not alive or step > 3000:
            if crashed:
                rewards.append(-30.0)
                log_probs.append(log_probs[-1] if log_probs else torch.tensor(0.0))
            break

        if step % POLICY_FREQ == 0:
            state_arr, ego_speed, prox, v_ahead_speed, ego_x, ego_y = get_state()

            # Collect NPC aggressiveness data; print full breakdown on the
            # first policy step of selected epochs for hand-calculation verification.
            show = verbose_epoch and (step == POLICY_FREQ)
            npc_collector.update(epoch, step, ego_x, ego_y, print_breakdown=show)

            state_tensor = torch.tensor(state_arr, dtype=torch.float32)
            action_probs = agent(state_tensor)

            noisy_probs = (action_probs + exploration_noise) / (1.0 + exploration_noise * 5)
            m           = torch.distributions.Categorical(probs=noisy_probs)
            action      = m.sample()
            log_prob    = m.log_prob(action)

            ego_speed_target = execute_action(action.item(), ego_speed_target)
            reward           = compute_reward(action.item(), ego_speed, prox,
                                              v_ahead_speed, crashed)

            log_probs.append(log_prob)
            rewards.append(reward)

    return log_probs, rewards


# ============================================================
# TRAINING LOOP
# ============================================================
def train():
    print("=" * 60)
    print("  SUMO Highway Agent — NPC Aggressiveness Data Collection")
    print("  Data collection linked to model.py AggressivenessModel")
    print("=" * 60)

    use_gui  = "--no-gui" not in sys.argv    # GUI is ON by default
    binary   = "sumo-gui.exe" if use_gui else "sumo.exe"
    sumo_bin = os.path.join(SUMO_HOME, "bin", binary)

    sumo_cmd = [sumo_bin, "-c", CFG_PATH]
    if use_gui:
        sumo_cmd += ["--delay", "100"]      # 100 ms between GUI frames

    traci.start(sumo_cmd)

    if use_gui:
        try:
            traci.gui.setDelay("View #0", 100)
        except Exception:
            pass

    agent     = SmartDriverAI()
    optimizer = optim.Adam(agent.parameters(), lr=0.005)

    epochs        = 200
    initial_noise = 5.0
    min_noise     = 0.01
    decay_rate    = 0.015
    entropy_beta  = 0.01
    history       = []

    npc_collector = NPCDataCollector()

    verbose_epochs = {1} | {e for e in range(20, epochs + 1, 20)}

    print(f"\n[CONFIG] epochs={epochs}  GUI={'on' if use_gui else 'off'}")
    print(f"[CONFIG] Aggressiveness model weights from model.py:")
    print(f"         w_speed={npc_collector.model.w_speed}  w_accel={npc_collector.model.w_accel}")
    print(f"         w_prox={npc_collector.model.w_prox}   w_wave={npc_collector.model.w_wave}\n")

    for epoch in range(1, epochs + 1):
        noise   = min_noise + (initial_noise - min_noise) * np.exp(-decay_rate * epoch)
        verbose = epoch in verbose_epochs

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  EPOCH {epoch}  — full NPC aggressiveness breakdown below")
            print(f"{'─'*60}")

        log_probs, rewards = run_episode(agent, epoch, noise, npc_collector, verbose)

        if not log_probs:
            continue

        total_reward = sum(rewards)
        history.append([epoch, total_reward])

        # REINFORCE + entropy regularisation
        gamma, R, returns = 0.99, 0, []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        action_probs_list = [torch.exp(lp) for lp in log_probs]
        policy_loss       = torch.stack(
            [-lp * R for lp, R in zip(log_probs, returns)]
        ).sum()

        action_tensor = torch.stack(action_probs_list)
        entropy       = -torch.sum(action_tensor * torch.log(action_tensor + 1e-9))
        total_loss    = policy_loss - entropy_beta * entropy

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:3d}/{epochs} | Reward: {total_reward:7.2f} | Noise: {noise:.3f}", end="")
        npc_collector.epoch_summary(epoch)

    traci.close()

    # ---- Save training history ----
    df_hist = pd.DataFrame(history, columns=["Epoch", "Reward"])
    df_hist.to_csv("sumo_highway_history.csv", index=False)

    # ---- Save NPC aggressiveness dataset (linked to model.py) ----
    npc_collector.save_csv("sumo_highway_npc_aggressiveness.csv")

    # ---- Learning curve plot ----
    plt.figure(figsize=(10, 5))
    smoothed = df_hist["Reward"].rolling(10, min_periods=1).mean()
    plt.plot(df_hist["Reward"], color="#bdc3c7", alpha=0.4, label="Raw Reward")
    plt.plot(smoothed,          color="#27ae60", linewidth=2, label="10-Epoch Trend")
    plt.title("SUMO Highway Agent — Learning Curve (200 epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sumo_highway_curve.png")

    # ---- NPC aggressiveness distribution plot ----
    df_npc = pd.read_csv("sumo_highway_npc_aggressiveness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_npc["ai_score"], bins=40, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].axvline(35, color="#f39c12", linestyle="--", label="Conservative/Normal (35)")
    axes[0].axvline(70, color="#e74c3c", linestyle="--", label="Normal/Aggressive (70)")
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

    plt.suptitle("Highway NPC Aggressiveness Data Collection", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("sumo_highway_npc_aggressiveness_plot.png")

    print("\n[SYSTEM] Done.")
    print("  sumo_highway_curve.png")
    print("  sumo_highway_npc_aggressiveness.csv")
    print("  sumo_highway_npc_aggressiveness_plot.png")


if __name__ == "__main__":
    train()
