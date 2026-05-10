import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os


# 1. CREATE A DUMMY CSV DATASET
def generate_synthetic_data(filename="telemetry_data.csv"):
    np.random.seed(42)
    data = []
    
    for _ in range(1000):
        # Highway Data
        data.append(['highway', np.random.rand(), np.random.rand()*0.5, np.random.rand(), np.random.rand()*0.2, np.random.choice([0.0, 0.2, 0.8, 1.0])])
        # Urban Data 
        data.append(['urban', np.random.rand(), np.random.rand(), np.random.rand()*0.5, np.random.rand()*0.3, np.random.choice([0.0, 0.3, 0.7, 1.0])])
        # Weather Data 
        data.append(['weather', np.random.rand(), np.random.rand(), np.random.rand()*0.4, np.random.rand(), np.random.choice([0.0, 0.1, 0.9, 1.0])])

    df = pd.DataFrame(data, columns=['environment', 'metric_1', 'metric_2', 'metric_3', 'metric_4', 'ground_truth'])
    df.to_csv(filename, index=False)
    print(f"[SYSTEM] Created synthetic dataset: {filename} with {len(df)} rows.\n")

if not os.path.exists("telemetry_data.csv"):
    generate_synthetic_data()


# 2. THE AGENT ARCHITECTURE

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


# 3. SETUP & HELPER FUNCTIONS

agent = DynamicWeightAgent()
optimizer = optim.Adam(agent.parameters(), lr=0.01) 

def print_weights(epoch, title=""):
    print(f"--- {title} (Epoch {epoch}) ---")
    hw = agent.get_weights('highway').detach().numpy()
    uw = agent.get_weights('urban').detach().numpy()
    ww = agent.get_weights('weather').detach().numpy()
    
    print(f"Highway -> THW: {hw[0]:.3f} | Weave: {hw[1]:.3f} | SpeedDelta: {hw[2]:.3f} | Brake: {hw[3]:.3f}")
    print(f"Urban   -> Occl: {uw[0]:.3f} | VRU: {uw[1]:.3f} | PET: {uw[2]:.3f} | Dilma: {uw[3]:.3f}")
    print(f"Weather -> Trac: {ww[0]:.3f} | Wet_HW: {ww[1]:.3f} | Steer: {ww[2]:.3f} | Hydro: {ww[3]:.3f}\n")


# 4. TRAIN DIRECTLY FROM CSV

df = pd.read_csv("telemetry_data.csv")

print_weights(0, "INITIAL RANDOM WEIGHTS")

epochs = 10 
# Note: An "epoch" now means reading through the ENTIRE 3000-row CSV file.
for epoch in range(1, epochs + 1):
    
    # Shuffle the dataset so the agent doesn't memorize the order
    df = df.sample(frac=1).reset_index(drop=True)
    
    total_loss = 0
    
  
    for index, row in df.iterrows():
        env_type = row['environment']
        metrics = [row['metric_1'], row['metric_2'], row['metric_3'], row['metric_4']]
        truth = row['ground_truth']
        metrics_tensor = torch.tensor(metrics, dtype=torch.float32)
        truth_tensor = torch.tensor(truth, dtype=torch.float32)
        optimizer.zero_grad()
        predicted_ai = agent(env_type, metrics_tensor)
        loss = F.mse_loss(predicted_ai, truth_tensor)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
   
    if epoch % 2 == 0:
        print(f"[Epoch {epoch}] Average Loss: {total_loss/len(df):.4f}")

print("\n[SYSTEM] TRAINING COMPLETE ON DATASET.\n")
print_weights(epochs, "FINAL DISTRIBUTED WEIGHTS")