import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. DATA GENERATION (from your original code) ---
def generate_synthetic_data(filename="telemetry_data.csv"):
    np.random.seed(42)
    data = []
    # Creating obvious patterns so we can visually confirm the agent finds them
    for _ in range(1000):
        # Highway: Metric 1 is highly correlated to truth
        data.append(['highway', np.random.rand(), np.random.rand()*0.5, np.random.rand(), np.random.rand()*0.2, np.random.choice([0.0, 0.2, 0.8, 1.0])])
        # Urban: Metric 2 is highly correlated
        data.append(['urban', np.random.rand(), np.random.rand(), np.random.rand()*0.5, np.random.rand()*0.3, np.random.choice([0.0, 0.3, 0.7, 1.0])])
        # Weather: Metric 3 is highly correlated
        data.append(['weather', np.random.rand(), np.random.rand(), np.random.rand()*0.4, np.random.rand(), np.random.choice([0.0, 0.1, 0.9, 1.0])])

    df = pd.DataFrame(data, columns=['environment', 'metric_1', 'metric_2', 'metric_3', 'metric_4', 'ground_truth'])
    df.to_csv(filename, index=False)

if not os.path.exists("telemetry_data.csv"):
    generate_synthetic_data()

# --- 2. AGENT DEFINITION ---
class DynamicWeightAgent(nn.Module):
    def __init__(self):
        super(DynamicWeightAgent, self).__init__()
        torch.manual_seed(42)
        self.raw_weights_highway = nn.Parameter(torch.randn(4))
        self.raw_weights_urban = nn.Parameter(torch.randn(4))
        self.raw_weights_weather = nn.Parameter(torch.randn(4))

    def get_weights(self, env_type):
        if env_type == 'highway': return F.softmax(self.raw_weights_highway, dim=0)
        elif env_type == 'urban': return F.softmax(self.raw_weights_urban, dim=0)
        elif env_type == 'weather': return F.softmax(self.raw_weights_weather, dim=0)

    def forward(self, env_type, trial_metrics):
        weights = self.get_weights(env_type)
        return torch.sum(weights * trial_metrics)

# --- 3. TRAINING & TRACKING LOGIC ---
agent = DynamicWeightAgent()
optimizer = optim.Adam(agent.parameters(), lr=0.01)
df = pd.read_csv("telemetry_data.csv")

epochs = 15
history = {
    "loss": [],
    "highway": {0:[], 1:[], 2:[], 3:[]},
    "urban": {0:[], 1:[], 2:[], 3:[]},
    "weather": {0:[], 1:[], 2:[], 3:[]}
}

print("[SYSTEM] Starting Training with Visualization Tracking...")

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
    
    # Track the weights at the end of each epoch
    hw = agent.get_weights('highway').detach().numpy()
    uw = agent.get_weights('urban').detach().numpy()
    ww = agent.get_weights('weather').detach().numpy()
    
    for i in range(4):
        history["highway"][i].append(hw[i])
        history["urban"][i].append(uw[i])
        history["weather"][i].append(ww[i])

    print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

# --- 4. PLOTTING THE DASHBOARD ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Dynamic Weight Agent: Training Verification Dashboard", fontsize=16)

# Plot 1: Overall Loss
axs[0, 0].plot(range(1, epochs+1), history["loss"], color='black', marker='o', linewidth=2)
axs[0, 0].set_title("Model Loss (Mean Squared Error)")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].grid(True, alpha=0.3)

# Plot 2: Highway Weights
for i in range(4):
    axs[0, 1].plot(range(1, epochs+1), history["highway"][i], label=f"Metric {i+1}")
axs[0, 1].set_title("Highway Weights Convergence")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Weight Value (Softmax)")
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# Plot 3: Urban Weights
for i in range(4):
    axs[1, 0].plot(range(1, epochs+1), history["urban"][i], label=f"Metric {i+1}")
axs[1, 0].set_title("Urban Weights Convergence")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Weight Value (Softmax)")
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# Plot 4: Weather Weights
for i in range(4):
    axs[1, 1].plot(range(1, epochs+1), history["weather"][i], label=f"Metric {i+1}")
axs[1, 1].set_title("Weather Weights Convergence")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Weight Value (Softmax)")
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("agent_verification_dashboard.png")
print("\n[SYSTEM] Dashboard saved as 'agent_verification_dashboard.png'. Displaying now...")
plt.show()