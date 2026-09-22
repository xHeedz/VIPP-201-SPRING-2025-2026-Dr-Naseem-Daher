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

# --- 1. THE TRUTH (Our hidden rules that the AI must discover) ---
# Notice how these all sum exactly to 1.0 (100%)
TRUE_W_HW = np.array([0.60, 0.25, 0.10, 0.05])
TRUE_W_UR = np.array([0.10, 0.50, 0.30, 0.10])
TRUE_W_WE = np.array([0.25, 0.25, 0.25, 0.25])

def generate_strict_data(filename=os.path.join(DATA_DIR, "strict_telemetry.csv")):
    np.random.seed(42)
    data = []
    
    for _ in range(1000):
        m = np.random.rand(4)
        data.append(['highway', m[0], m[1], m[2], m[3], np.sum(m * TRUE_W_HW)])
        
        m = np.random.rand(4)
        data.append(['urban', m[0], m[1], m[2], m[3], np.sum(m * TRUE_W_UR)])
        
        m = np.random.rand(4)
        data.append(['weather', m[0], m[1], m[2], m[3], np.sum(m * TRUE_W_WE)])

    df = pd.DataFrame(data, columns=['environment', 'metric_1', 'metric_2', 'metric_3', 'metric_4', 'ground_truth'])
    df.to_csv(filename, index=False)

generate_strict_data()

# --- 2. THE AGENT ---
class DynamicWeightAgent(nn.Module):
    def __init__(self):
        super(DynamicWeightAgent, self).__init__()
        torch.manual_seed(42)
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

# --- 3. TRAINING LOOP (Tuned for Exact Convergence) ---
agent = DynamicWeightAgent()
# Increased learning rate and epochs to ensure it reaches the exact target
optimizer = optim.Adam(agent.parameters(), lr=0.2) 
df = pd.read_csv(os.path.join(DATA_DIR, "strict_telemetry.csv"))
epochs = 80

history = {"loss": [], "highway": {0:[], 1:[], 2:[], 3:[]}, "urban": {0:[], 1:[], 2:[], 3:[]}, "weather": {0:[], 1:[], 2:[], 3:[]}}

print("[SYSTEM] Agent is attempting to reverse-engineer the hidden rules...")

for epoch in range(1, epochs + 1):
    total_loss = 0
    optimizer.zero_grad() 
    
    for index, row in df.iterrows():
        env_type = row['environment']
        metrics_tensor = torch.tensor([row['metric_1'], row['metric_2'], row['metric_3'], row['metric_4']], dtype=torch.float32)
        truth_tensor = torch.tensor(row['ground_truth'], dtype=torch.float32)
        
        predicted = agent(env_type, metrics_tensor)
        loss = F.mse_loss(predicted, truth_tensor)
        loss.backward() 
        total_loss += loss.item()
    
    optimizer.step()
    
    # Record data
    history["loss"].append(total_loss / len(df))
    hw = agent.get_weights('highway').detach().numpy()
    uw = agent.get_weights('urban').detach().numpy()
    ww = agent.get_weights('weather').detach().numpy()
    for i in range(4):
        history["highway"][i].append(hw[i])
        history["urban"][i].append(uw[i])
        history["weather"][i].append(ww[i])

# --- 4. THE ANALYZER (The Proof) ---
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("The Proof: Agent's Guesses (Solid) vs Ground Truth Rules (Dashed)", fontsize=16)

# Colors for consistency
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

axs[0, 0].plot(range(1, epochs+1), history["loss"], 'k-', linewidth=2)
axs[0, 0].set_title("Model Loss (Reaching near absolute zero)"); axs[0, 0].grid(True, alpha=0.3)

# Plot Highway
for i in range(4): 
    axs[0, 1].axhline(y=TRUE_W_HW[i], color=colors[i], linestyle='--', alpha=0.5) # The Truth
    axs[0, 1].plot(range(1, epochs+1), history["highway"][i], color=colors[i], label=f"M{i+1}", linewidth=2) # Agent
axs[0, 1].set_title("Highway: Target is [0.60, 0.25, 0.10, 0.05]"); axs[0, 1].legend(); axs[0, 1].grid(True, alpha=0.3)

# Plot Urban
for i in range(4): 
    axs[1, 0].axhline(y=TRUE_W_UR[i], color=colors[i], linestyle='--', alpha=0.5) 
    axs[1, 0].plot(range(1, epochs+1), history["urban"][i], color=colors[i], label=f"M{i+1}", linewidth=2)
axs[1, 0].set_title("Urban: Target is [0.10, 0.50, 0.30, 0.10]"); axs[1, 0].legend(); axs[1, 0].grid(True, alpha=0.3)

# Plot Weather
for i in range(4): 
    axs[1, 1].axhline(y=TRUE_W_WE[i], color=colors[i], linestyle='--', alpha=0.5) 
    axs[1, 1].plot(range(1, epochs+1), history["weather"][i], color=colors[i], label=f"M{i+1}", linewidth=2)
axs[1, 1].set_title("Weather: Target is [0.25, 0.25, 0.25, 0.25]"); axs[1, 1].legend(); axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(FIG_DIR, "weight_convergence_strict.png"))
print("\n[SYSTEM] Check 'weight_convergence_strict.png'.")
plt.show()