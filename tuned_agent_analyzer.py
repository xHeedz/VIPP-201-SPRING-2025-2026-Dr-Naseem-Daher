import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. THE FIX: MEANINGFUL SYNTHETIC DATA ---
def generate_logical_data(filename="logical_telemetry.csv"):
    """
    Instead of random targets, we establish SECRET RULES. 
    If the agent works, it will discover these exact weights.
    """
    np.random.seed(42)
    data = []
    
    # SECRET RULE 1 (Highway): 80% Metric 1, 20% Metric 2. M3 and M4 are useless.
    for _ in range(500):
        m1, m2, m3, m4 = np.random.rand(4)
        truth = (0.8 * m1) + (0.2 * m2) + (0.0 * m3) + (0.0 * m4)
        data.append(['highway', m1, m2, m3, m4, truth])

    # SECRET RULE 2 (Urban): 60% Metric 2, 40% Metric 3. M1 and M4 are useless.
    for _ in range(500):
        m1, m2, m3, m4 = np.random.rand(4)
        truth = (0.0 * m1) + (0.6 * m2) + (0.4 * m3) + (0.0 * m4)
        data.append(['urban', m1, m2, m3, m4, truth])

    # SECRET RULE 3 (Weather): All metrics matter equally (25% each).
    for _ in range(500):
        m1, m2, m3, m4 = np.random.rand(4)
        truth = (0.25 * m1) + (0.25 * m2) + (0.25 * m3) + (0.25 * m4)
        data.append(['weather', m1, m2, m3, m4, truth])

    df = pd.DataFrame(data, columns=['environment', 'metric_1', 'metric_2', 'metric_3', 'metric_4', 'ground_truth'])
    df.to_csv(filename, index=False)

generate_logical_data()

# --- 2. THE AGENT (Tuned) ---
class DynamicWeightAgent(nn.Module):
    def __init__(self):
        super(DynamicWeightAgent, self).__init__()
        torch.manual_seed(42)
        # Initialize raw weights closer to 0 for a cleaner start
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

# --- 3. TRAINING & NEW ANALYZER ---
agent = DynamicWeightAgent()
# TUNING: Lowered learning rate slightly for smoother, predictable convergence
optimizer = optim.Adam(agent.parameters(), lr=0.05) 
df = pd.read_csv("logical_telemetry.csv")

epochs = 15
history = {"loss": [], "highway": {0:[], 1:[], 2:[], 3:[]}, "urban": {0:[], 1:[], 2:[], 3:[]}, "weather": {0:[], 1:[], 2:[], 3:[]}}

print("[SYSTEM] Training Agent on Logical Data...")

for epoch in range(1, epochs + 1):
    df = df.sample(frac=1).reset_index(drop=True)
    total_loss = 0
    
    for index, row in df.iterrows():
        env_type = row['environment']
        metrics_tensor = torch.tensor([row['metric_1'], row['metric_2'], row['metric_3'], row['metric_4']], dtype=torch.float32)
        truth_tensor = torch.tensor(row['ground_truth'], dtype=torch.float32)
        
        optimizer.zero_grad()
        predicted_ai = agent(env_type, metrics_tensor)
        loss = F.mse_loss(predicted_ai, truth_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(df)
    history["loss"].append(avg_loss)
    
    hw = agent.get_weights('highway').detach().numpy()
    uw = agent.get_weights('urban').detach().numpy()
    ww = agent.get_weights('weather').detach().numpy()
    
    for i in range(4):
        history["highway"][i].append(hw[i])
        history["urban"][i].append(uw[i])
        history["weather"][i].append(ww[i])

# --- 4. THE PROOF (Plotting) ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Agent Verification: Discovering the Secret Rules", fontsize=16)

# 1. Loss
axs[0, 0].plot(range(1, epochs+1), history["loss"], color='black', marker='o', linewidth=2)
axs[0, 0].set_title("Model Loss (Should drop to near 0)")
axs[0, 0].grid(True, alpha=0.3)

# 2. Highway (Secret: 0.8, 0.2, 0, 0)
for i in range(4): axs[0, 1].plot(range(1, epochs+1), history["highway"][i], label=f"Metric {i+1}", linewidth=2)
axs[0, 1].set_title("Highway Weights (Target: M1=0.8, M2=0.2, M3=0, M4=0)")
axs[0, 1].legend(); axs[0, 1].grid(True, alpha=0.3)

# 3. Urban (Secret: 0, 0.6, 0.4, 0)
for i in range(4): axs[1, 0].plot(range(1, epochs+1), history["urban"][i], label=f"Metric {i+1}", linewidth=2)
axs[1, 0].set_title("Urban Weights (Target: M2=0.6, M3=0.4)")
axs[1, 0].legend(); axs[1, 0].grid(True, alpha=0.3)

# 4. Weather (Secret: 0.25 all)
for i in range(4): axs[1, 1].plot(range(1, epochs+1), history["weather"][i], label=f"Metric {i+1}", linewidth=2)
axs[1, 1].set_title("Weather Weights (Target: All exactly 0.25)")
axs[1, 1].legend(); axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("tuned_agent_results.png")
print("\n[SYSTEM] Done! Check 'tuned_agent_results.png'")
plt.show()