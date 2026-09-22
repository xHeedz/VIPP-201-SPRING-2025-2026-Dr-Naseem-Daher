import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from paths import DATA_DIR, FIG_DIR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. DATASET (Adding slight correlations so the agent actually has something to learn)
def generate_synthetic_data(filename=os.path.join(DATA_DIR, "telemetry_data.csv")):
    np.random.seed(42)
    data = []
    for _ in range(1000):
        # We inject a slight hidden rule: Metric 1 is highly linked to the truth in Highway, etc.
        m1, m2, m3, m4 = np.random.rand(4)
        data.append(['highway', m1, m2, m3, m4, (m1*0.7) + (m2*0.3)])
        
        m1, m2, m3, m4 = np.random.rand(4)
        data.append(['urban', m1, m2, m3, m4, (m2*0.6) + (m3*0.4)])
        
        m1, m2, m3, m4 = np.random.rand(4)
        data.append(['weather', m1, m2, m3, m4, (m3*0.8) + (m4*0.2)])

    df = pd.DataFrame(data, columns=['environment', 'metric_1', 'metric_2', 'metric_3', 'metric_4', 'ground_truth'])
    df.to_csv(filename, index=False)

if not os.path.exists(os.path.join(DATA_DIR, "telemetry_data.csv")):
    generate_synthetic_data()

# 2. THE TUNED AGENT
class DynamicWeightAgent(nn.Module):
    def __init__(self):
        super(DynamicWeightAgent, self).__init__()
        torch.manual_seed(42)
        # TUNE 1: Start weights uniformly at 0 so Softmax begins perfectly balanced at 0.25
        self.raw_weights_highway = nn.Parameter(torch.zeros(4))
        self.raw_weights_urban = nn.Parameter(torch.zeros(4))
        self.raw_weights_weather = nn.Parameter(torch.zeros(4))

    def get_weights(self, env_type):
        if env_type == 'highway': return F.softmax(self.raw_weights_highway, dim=0)
        elif env_type == 'urban': return F.softmax(self.raw_weights_urban, dim=0)
        elif env_type == 'weather': return F.softmax(self.raw_weights_weather, dim=0)

    def forward(self, env_type, trial_metrics):
        weights = self.get_weights(env_type)
        return torch.sum(weights * trial_metrics)


# 3. THE TUNED TRAINING LOOP
agent = DynamicWeightAgent()
optimizer = optim.Adam(agent.parameters(), lr=0.1)

# TUNE 2: Add a Learning Rate Scheduler. Every 5 epochs, cut the learning rate by half.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

df = pd.read_csv(os.path.join(DATA_DIR, "telemetry_data.csv"))
epochs = 20

history = {"loss": [], "highway": {0:[], 1:[], 2:[], 3:[]}, "urban": {0:[], 1:[], 2:[], 3:[]}, "weather": {0:[], 1:[], 2:[], 3:[]}}

print("[SYSTEM] Training Tuned Agent using Full-Batch Gradient Descent...")

for epoch in range(1, epochs + 1):
    total_loss = 0
    
    # TUNE 3: Zero the gradients at the START of the epoch, not inside the loop
    optimizer.zero_grad() 
    
    for index, row in df.iterrows():
        env_type = row['environment']
        metrics_tensor = torch.tensor([row['metric_1'], row['metric_2'], row['metric_3'], row['metric_4']], dtype=torch.float32)
        truth_tensor = torch.tensor(row['ground_truth'], dtype=torch.float32)
        
        predicted_ai = agent(env_type, metrics_tensor)
        loss = F.mse_loss(predicted_ai, truth_tensor)
        
        # Accumulate the gradients (add them up for all 1000 rows)
        loss.backward() 
        total_loss += loss.item()
    
    # TUNE 4: Take ONE precise optimizer step using the average of all rows
    optimizer.step()
    scheduler.step()
    
    # --- ANALYZER TRACKING ---
    avg_loss = total_loss / len(df)
    history["loss"].append(avg_loss)
    
    hw = agent.get_weights('highway').detach().numpy()
    uw = agent.get_weights('urban').detach().numpy()
    ww = agent.get_weights('weather').detach().numpy()
    
    for i in range(4):
        history["highway"][i].append(hw[i])
        history["urban"][i].append(uw[i])
        history["weather"][i].append(ww[i])

    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.4f}")

# 4. THE TUNED ANALYZER (Graphing the Results)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Agent Learning Trajectory (Stabilized via Batch Learning)", fontsize=16)

axs[0, 0].plot(range(1, epochs+1), history["loss"], 'k-o', linewidth=2)
axs[0, 0].set_title("Smoothed Model Loss"); axs[0, 0].grid(True, alpha=0.3)

for i in range(4): axs[0, 1].plot(range(1, epochs+1), history["highway"][i], label=f"Metric {i+1}", linewidth=2)
axs[0, 1].set_title("Highway Weights"); axs[0, 1].legend(); axs[0, 1].grid(True, alpha=0.3)

for i in range(4): axs[1, 0].plot(range(1, epochs+1), history["urban"][i], label=f"Metric {i+1}", linewidth=2)
axs[1, 0].set_title("Urban Weights"); axs[1, 0].legend(); axs[1, 0].grid(True, alpha=0.3)

for i in range(4): axs[1, 1].plot(range(1, epochs+1), history["weather"][i], label=f"Metric {i+1}", linewidth=2)
axs[1, 1].set_title("Weather Weights"); axs[1, 1].legend(); axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(FIG_DIR, "tuned_agent_results.png"))
print("\n[SYSTEM] Check 'tuned_agent_results.png' - Notice how smooth the lines are now!")
plt.show()