"""
collect_data.py
---------------
Runs SUMO headless for N epochs, collects NPC aggressiveness data, and prints
full hand-calculation breakdowns so you can verify every intermediate step.

Outputs:
  quick_run_log.txt                       - all console output
  sumo_npc_aggressiveness.csv             - intersection NPC data
  sumo_highway_npc_aggressiveness.csv     - highway NPC data
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force UTF-8 on Windows so box chars don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from model import AggressivenessModel
from collections import deque

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci

N_EPOCHS = 5   # fast run for data collection + ground-truth verification


# ----------------------------------------------------------------
# Shared: AggressivenessModel-linked data collector
# ----------------------------------------------------------------
class NPCDataCollector:
    def __init__(self):
        self.model   = AggressivenessModel()
        self.history = {}
        self.records = []

    def update(self, epoch, step, ego_x, ego_y, ego_id, print_breakdown=False):
        for vid in traci.vehicle.getIDList():
            if vid == ego_id:
                continue
            speed_ms  = traci.vehicle.getSpeed(vid)
            speed_kmh = speed_ms * 3.6
            vx, vy    = traci.vehicle.getPosition(vid)
            prox_m    = float(np.sqrt((vx - ego_x)**2 + (vy - ego_y)**2))

            if vid not in self.history:
                self.history[vid] = {"prev": speed_ms, "ys": deque([vy], maxlen=30)}
                accel_ms2 = 0.0
            else:
                h = self.history[vid]
                accel_ms2  = (speed_ms - h["prev"]) / 0.1
                h["prev"]  = speed_ms
                h["ys"].append(vy)

            wave_m = float(np.std(list(self.history[vid]["ys"]))) \
                     if len(self.history[vid]["ys"]) > 3 else 0.0

            ai_score, label = self.model.get_ai_score(speed_kmh, accel_ms2, prox_m, wave_m)

            self.records.append({
                "epoch":      epoch,
                "step":       step,
                "vehicle_id": vid,
                "speed_kmh":  round(speed_kmh, 2),
                "accel_ms2":  round(accel_ms2,  3),
                "prox_m":     round(prox_m,     2),
                "wave_m":     round(wave_m,     3),
                "ai_score":   round(ai_score,   2),
                "category":   label,
            })

            if print_breakdown:
                _print_breakdown(self.model, vid, speed_kmh, accel_ms2,
                                 prox_m, wave_m, ai_score, label)

    def save(self, path):
        df = pd.DataFrame(self.records)
        df.to_csv(path, index=False)
        return df


def _print_breakdown(m, vid, speed_kmh, accel_ms2, prox_m, wave_m, ai_score, label):
    """Ground-truth hand-calculation trace -- every intermediate step shown."""
    # Step 1: Normalise  (mirrors model.py AggressivenessModel.normalize)
    n_speed = min(speed_kmh / 150.0, 1.0)
    n_accel = min(abs(accel_ms2) / 5.0, 1.0)
    n_prox  = (1.0 - prox_m / 50.0) if 0 < prox_m <= 50 else 0.0
    n_wave  = min(abs(wave_m) / 1.5, 1.0)

    # Step 2: Weighted contributions  (mirrors model.py get_ai_score)
    c_speed = n_speed**2 * m.w_speed
    c_accel = n_accel    * m.w_accel
    c_prox  = n_prox**2  * m.w_prox
    c_wave  = n_wave     * m.w_wave
    raw_sum = c_speed + c_accel + c_prox + c_wave

    print(f"\n  +-- {vid} --+")
    print(f"  | RAW      speed={speed_kmh:.2f} km/h  accel={accel_ms2:+.3f} m/s2  "
          f"prox={prox_m:.2f} m  wave={wave_m:.4f} m")
    print(f"  | NORMALISE  (model.py normalize)")
    print(f"  |   n_speed = min({speed_kmh:.2f}/150, 1.0)     = {n_speed:.5f}")
    print(f"  |   n_accel = min(|{accel_ms2:.3f}|/5.0, 1.0) = {n_accel:.5f}")
    if 0 < prox_m <= 50:
        print(f"  |   n_prox  = 1 - {prox_m:.2f}/50           = {n_prox:.5f}  (closer = riskier)")
    else:
        print(f"  |   n_prox  = 0.00000  (outside 50 m window)")
    print(f"  |   n_wave  = min({wave_m:.4f}/1.5, 1.0)   = {n_wave:.5f}")
    print(f"  | WEIGHTS from model.py:  "
          f"w_speed={m.w_speed}  w_accel={m.w_accel}  w_prox={m.w_prox}  w_wave={m.w_wave}")
    print(f"  | SCORE TERMS  (model.py get_ai_score)")
    print(f"  |   n_speed^2 * w_speed = {n_speed:.5f}^2 * {m.w_speed} = {c_speed:.6f}")
    print(f"  |   n_accel  * w_accel  = {n_accel:.5f}   * {m.w_accel} = {c_accel:.6f}")
    print(f"  |   n_prox^2 * w_prox   = {n_prox:.5f}^2 * {m.w_prox} = {c_prox:.6f}")
    print(f"  |   n_wave   * w_wave   = {n_wave:.5f}   * {m.w_wave} = {c_wave:.6f}")
    print(f"  | SUM  = {c_speed:.6f} + {c_accel:.6f} + {c_prox:.6f} + {c_wave:.6f}")
    print(f"  |      = {raw_sum:.6f}")
    print(f"  | FINAL AI Score = min({raw_sum:.6f} * 100, 100) = {ai_score:.3f}")
    print(f"  +-- LABEL: [{label}]  (score<35=Conservative  35-70=Normal  >=70=Aggressive) --+")


# ----------------------------------------------------------------
# Intersection run
# ----------------------------------------------------------------
def run_intersection():
    from sumo_urban_agent import (
        VehicleTracker, IntersectionAI, get_state_and_obs,
        execute_action, compute_reward, POLICY_FREQ
    )
    import torch
    import torch.optim as optim

    CFG = os.path.join(BASE, "sumo", "intersection", "intersection.sumocfg")
    EGO = "ego"
    BIN = os.path.join(SUMO_HOME, "bin", "sumo.exe")

    traci.start([BIN, "-c", CFG])

    agent     = IntersectionAI()
    optimizer = optim.Adam(agent.parameters(), lr=0.005)
    collector = NPCDataCollector()
    history   = []

    print("\n" + "="*60)
    print("  INTERSECTION -- hand-calculation trace (all epochs)")
    print("="*60)
    print(f"  Model weights from model.py:")
    print(f"    w_speed={collector.model.w_speed}  w_accel={collector.model.w_accel}"
          f"  w_prox={collector.model.w_prox}  w_wave={collector.model.w_wave}")
    print(f"  Scoring: score = (n_speed^2*w_speed + n_accel*w_accel"
          f" + n_prox^2*w_prox + n_wave*w_wave) * 100\n")

    for epoch in range(1, N_EPOCHS + 1):
        noise = 0.01 + (5.0 - 0.01) * np.exp(-0.015 * epoch)
        traci.load(["-c", CFG])

        for _ in range(100):
            traci.simulationStep()
            if EGO in traci.vehicle.getIDList():
                break

        if EGO not in traci.vehicle.getIDList():
            print(f"  [skip epoch {epoch}] ego did not spawn")
            continue

        traci.vehicle.setSpeedMode(EGO, 0)
        traci.vehicle.setLaneChangeMode(EGO, 0)

        tracker          = VehicleTracker(history_len=8)
        ego_speed_target = 10.0
        log_probs, rewards = [], []
        step = 0

        print(f"\n--- EPOCH {epoch}/{N_EPOCHS} ---")

        while True:
            traci.simulationStep()
            step += 1
            crashed = EGO in traci.simulation.getCollidingVehiclesIDList()
            alive   = EGO in traci.vehicle.getIDList()

            if not alive or step > 1200:
                if crashed:
                    rewards.append(-50.0)
                    log_probs.append(log_probs[-1] if log_probs else torch.tensor(0.0))
                break

            if step % POLICY_FREQ == 0:
                (state_arr, ego_vx, front_dist, cross_left, cross_right,
                 left_aggr, right_aggr, ego_x, ego_y) = get_state_and_obs(tracker)

                # Print full breakdown on first policy step of EVERY epoch
                show = (step == POLICY_FREQ)
                collector.update(epoch, step, ego_x, ego_y, EGO,
                                 print_breakdown=show)

                state_t      = torch.tensor(state_arr, dtype=torch.float32)
                action_probs = agent(state_t)
                noisy_probs  = (action_probs + noise) / (1.0 + noise * 3)
                m_dist       = torch.distributions.Categorical(probs=noisy_probs)
                action       = m_dist.sample()

                ego_speed_target = execute_action(action.item(), ego_speed_target)
                reward = compute_reward(action.item(), ego_vx, front_dist,
                                        cross_left, cross_right,
                                        left_aggr, right_aggr, crashed)
                log_probs.append(m_dist.log_prob(action))
                rewards.append(reward)

        total_reward = sum(rewards)
        history.append([epoch, total_reward])

        gamma, R, returns = 0.99, 0, []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)
        loss = torch.stack([-lp * R for lp, R in zip(log_probs, returns_t)]).sum()
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        epoch_rows = [r for r in collector.records if r["epoch"] == epoch]
        scores = [r["ai_score"] for r in epoch_rows]
        cats   = [r["category"] for r in epoch_rows]
        avg_ai = np.mean(scores) if scores else 0.0
        print(f"\n  Epoch {epoch} SUMMARY:  reward={total_reward:7.2f}  "
              f"NPC_obs={len(epoch_rows)}  avg_AI={avg_ai:.1f}  "
              f"C:{cats.count('Conservative')}  "
              f"N:{cats.count('Normal')}  "
              f"A:{cats.count('Aggressive')}")

    traci.close()
    out = os.path.join(BASE, "sumo_npc_aggressiveness.csv")
    df  = collector.save(out)
    pd.DataFrame(history, columns=["Epoch", "Reward"]).to_csv(
        os.path.join(BASE, "sumo_intersection_history.csv"), index=False)
    print(f"\n[Intersection] Saved {len(df)} rows -> {out}")
    return df


# ----------------------------------------------------------------
# Highway run
# ----------------------------------------------------------------
def run_highway():
    from sumo_highway_agent import (
        SmartDriverAI, get_state, execute_action, compute_reward, POLICY_FREQ
    )
    import torch
    import torch.optim as optim

    CFG = os.path.join(BASE, "sumo", "highway", "highway.sumocfg")
    EGO = "ego"
    BIN = os.path.join(SUMO_HOME, "bin", "sumo.exe")

    traci.start([BIN, "-c", CFG])

    agent     = SmartDriverAI()
    optimizer = optim.Adam(agent.parameters(), lr=0.005)
    collector = NPCDataCollector()
    history   = []

    print("\n" + "="*60)
    print("  HIGHWAY -- hand-calculation trace (all epochs)")
    print("="*60)
    print(f"  Model weights from model.py:")
    print(f"    w_speed={collector.model.w_speed}  w_accel={collector.model.w_accel}"
          f"  w_prox={collector.model.w_prox}  w_wave={collector.model.w_wave}")
    print(f"  Scoring: score = (n_speed^2*w_speed + n_accel*w_accel"
          f" + n_prox^2*w_prox + n_wave*w_wave) * 100\n")

    for epoch in range(1, N_EPOCHS + 1):
        noise = 0.01 + (5.0 - 0.01) * np.exp(-0.015 * epoch)
        traci.load(["-c", CFG])

        for _ in range(50):
            traci.simulationStep()
            if EGO in traci.vehicle.getIDList():
                break

        if EGO not in traci.vehicle.getIDList():
            print(f"  [skip epoch {epoch}] ego did not spawn")
            continue

        traci.vehicle.setSpeedMode(EGO, 0)
        traci.vehicle.setLaneChangeMode(EGO, 0)

        ego_speed_target = 20.0
        log_probs, rewards = [], []
        step = 0

        print(f"\n--- EPOCH {epoch}/{N_EPOCHS} ---")

        while True:
            traci.simulationStep()
            step += 1
            crashed = EGO in traci.simulation.getCollidingVehiclesIDList()
            alive   = EGO in traci.vehicle.getIDList()

            if not alive or step > 3000:
                if crashed:
                    rewards.append(-30.0)
                    log_probs.append(log_probs[-1] if log_probs else torch.tensor(0.0))
                break

            if step % POLICY_FREQ == 0:
                state_arr, ego_speed, prox, v_ahead_speed, ego_x, ego_y = get_state()

                show = (step == POLICY_FREQ)
                collector.update(epoch, step, ego_x, ego_y, EGO,
                                 print_breakdown=show)

                state_t      = torch.tensor(state_arr, dtype=torch.float32)
                action_probs = agent(state_t)
                noisy_probs  = (action_probs + noise) / (1.0 + noise * 5)
                m_dist       = torch.distributions.Categorical(probs=noisy_probs)
                action       = m_dist.sample()

                ego_speed_target = execute_action(action.item(), ego_speed_target)
                reward = compute_reward(action.item(), ego_speed, prox,
                                        v_ahead_speed, crashed)
                log_probs.append(m_dist.log_prob(action))
                rewards.append(reward)

        total_reward = sum(rewards)
        history.append([epoch, total_reward])

        gamma, R, returns = 0.99, 0, []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)

        action_probs_list = [torch.exp(lp) for lp in log_probs]
        policy_loss = torch.stack([-lp * R for lp, R in zip(log_probs, returns_t)]).sum()
        action_tensor = torch.stack(action_probs_list)
        entropy = -torch.sum(action_tensor * torch.log(action_tensor + 1e-9))
        loss = policy_loss - 0.01 * entropy
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        epoch_rows = [r for r in collector.records if r["epoch"] == epoch]
        scores = [r["ai_score"] for r in epoch_rows]
        cats   = [r["category"] for r in epoch_rows]
        avg_ai = np.mean(scores) if scores else 0.0
        print(f"\n  Epoch {epoch} SUMMARY:  reward={total_reward:7.2f}  "
              f"NPC_obs={len(epoch_rows)}  avg_AI={avg_ai:.1f}  "
              f"C:{cats.count('Conservative')}  "
              f"N:{cats.count('Normal')}  "
              f"A:{cats.count('Aggressive')}")

    traci.close()
    out = os.path.join(BASE, "sumo_highway_npc_aggressiveness.csv")
    df  = collector.save(out)
    pd.DataFrame(history, columns=["Epoch", "Reward"]).to_csv(
        os.path.join(BASE, "sumo_highway_history.csv"), index=False)
    print(f"\n[Highway] Saved {len(df)} rows -> {out}")
    return df


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
if __name__ == "__main__":
    log_path = os.path.join(BASE, "quick_run_log.txt")

    class Tee:
        def __init__(self, f):
            self.f = f
        def write(self, s):
            try:
                _real.write(s)
            except Exception:
                pass
            try:
                self.f.write(s)
            except Exception:
                pass
        def flush(self):
            try:
                _real.flush()
            except Exception:
                pass
            try:
                self.f.flush()
            except Exception:
                pass

    _real = sys.stdout

    with open(log_path, "w", encoding="utf-8") as logf:
        sys.stdout = Tee(logf)

        df_int = None
        try:
            df_int = run_intersection()
        except Exception as e:
            import traceback
            print(f"\n[ERROR intersection] {e}")
            traceback.print_exc()
            try:
                traci.close()
            except Exception:
                pass

        df_hw = None
        try:
            df_hw = run_highway()
        except Exception as e:
            import traceback
            print(f"\n[ERROR highway] {e}")
            traceback.print_exc()
            try:
                traci.close()
            except Exception:
                pass

        sys.stdout = _real

    print(f"\n[DONE] Log saved -> {log_path}")
    if df_int is not None:
        print(f"  Intersection: {len(df_int)} NPC observations")
    if df_hw is not None:
        print(f"  Highway:      {len(df_hw)} NPC observations")
