"""
Quick data-collection run: 5 epochs, headless, verbose hand-calculations on every epoch.
Outputs:
  - sumo_npc_aggressiveness.csv
  - sumo_highway_npc_aggressiveness.csv
  - quick_run_log.txt   (captured stdout with all hand-calc traces)
"""
import os, sys, io, contextlib

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ── patch both agents to run 5 epochs with verbose on every epoch ──
import importlib, types

def patch_and_run(module_name, epochs=5):
    mod = importlib.import_module(module_name)

    orig_train = mod.train

    def patched_train():
        import torch, numpy as np, pandas as pd, matplotlib
        matplotlib.use("Agg")   # no display needed

        import traci as _traci
        import torch.optim as optim

        SUMO_HOME = mod.SUMO_HOME
        CFG_PATH  = mod.CFG_PATH
        EGO_ID    = mod.EGO_ID

        use_gui  = False
        binary   = "sumo.exe"
        sumo_bin = os.path.join(SUMO_HOME, "bin", binary)
        sumo_cmd = [sumo_bin, "-c", CFG_PATH]

        _traci.start(sumo_cmd)

        agent     = mod.IntersectionAI() if hasattr(mod, "IntersectionAI") else mod.SmartDriverAI()
        optimizer = optim.Adam(agent.parameters(), lr=0.005)

        initial_noise = 5.0
        min_noise     = 0.01
        decay_rate    = 0.015
        history       = []

        npc_collector = mod.NPCDataCollector()
        verbose_epochs = set(range(1, epochs + 1))   # verbose on every epoch

        print(f"\n[{module_name}] epochs={epochs}  GUI=off  verbose=ALL")
        print(f"[{module_name}] Weights: w_speed={npc_collector.model.w_speed}  "
              f"w_accel={npc_collector.model.w_accel}  "
              f"w_prox={npc_collector.model.w_prox}  "
              f"w_wave={npc_collector.model.w_wave}\n")

        for epoch in range(1, epochs + 1):
            noise   = min_noise + (initial_noise - min_noise) * np.exp(-decay_rate * epoch)
            verbose = epoch in verbose_epochs

            print(f"\n{'═'*60}")
            print(f"  EPOCH {epoch}/{epochs}  —  NPC aggressiveness hand-calculations")
            print(f"{'═'*60}")

            if hasattr(mod, "run_episode"):
                # intersection agent signature: (agent, noise, epoch, npc_collector, verbose)
                import inspect
                sig = inspect.signature(mod.run_episode)
                params = list(sig.parameters.keys())
                if "epoch" in params:
                    log_probs, rewards = mod.run_episode(agent, noise, epoch, npc_collector, verbose)
                else:
                    log_probs, rewards = mod.run_episode(agent, noise, npc_collector, verbose)

            if not log_probs:
                print(f"  [skip] ego did not spawn in epoch {epoch}")
                continue

            total_reward = sum(rewards)
            history.append([epoch, total_reward])

            gamma, R, returns = 0.99, 0, []
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)

            import torch as _torch
            returns_t = _torch.tensor(returns, dtype=_torch.float32)
            if len(returns_t) > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)

            policy_loss = _torch.stack(
                [-lp * R for lp, R in zip(log_probs, returns_t)]
            ).sum()
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            print(f"\nEpoch {epoch:2d} | Reward: {total_reward:7.2f} | Noise: {noise:.3f}", end="  ")
            npc_collector.epoch_summary(epoch)

        _traci.close()

        csv_name = f"sumo_npc_aggressiveness.csv" if "urban" in module_name else "sumo_highway_npc_aggressiveness.csv"
        npc_collector.save_csv(csv_name)

        import pandas as _pd
        df = _pd.DataFrame(history, columns=["Epoch", "Reward"])
        hist_name = "sumo_intersection_history.csv" if "urban" in module_name else "sumo_highway_history.csv"
        df.to_csv(hist_name, index=False)
        print(f"[{module_name}] Done.\n")

    return patched_train


if __name__ == "__main__":
    log_path = os.path.join(BASE, "quick_run_log.txt")
    print(f"Writing all output to {log_path}  (also echoed to console)\n")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    with open(log_path, "w", encoding="utf-8") as logf:
        tee = Tee(sys.stdout, logf)
        old_stdout = sys.stdout
        sys.stdout = tee

        try:
            print("=" * 60)
            print("  QUICK RUN — Intersection Agent (5 epochs)")
            print("=" * 60)
            import sumo_urban_agent as urban
            urban_runner = patch_and_run("sumo_urban_agent", epochs=5)
            urban_runner()
        except Exception as e:
            print(f"[ERROR intersection] {e}")
            import traceback; traceback.print_exc()

        try:
            print("=" * 60)
            print("  QUICK RUN — Highway Agent (5 epochs)")
            print("=" * 60)
            import sumo_highway_agent as highway
            highway_runner = patch_and_run("sumo_highway_agent", epochs=5)
            highway_runner()
        except Exception as e:
            print(f"[ERROR highway] {e}")
            import traceback; traceback.print_exc()

        sys.stdout = old_stdout

    print(f"\nLog saved to {log_path}")
