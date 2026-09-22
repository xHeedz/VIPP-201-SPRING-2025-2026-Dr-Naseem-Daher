import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. Define the Agent
class DynamicWeightAgent(nn.Module):
    def __init__(self):
        super(DynamicWeightAgent, self).__init__()
        # Initialize random raw weights for 4 parameters per environment
        torch.manual_seed(42) # For reproducible results
        self.raw_weights_highway = nn.Parameter(torch.randn(4))
        self.raw_weights_urban = nn.Parameter(torch.randn(4))
        self.raw_weights_weather = nn.Parameter(torch.randn(4))

    def get_weights(self, env_type):
        # Softmax forces weights to be between 0 and 1, and sum to 1.0
        if env_type == 'highway':
            return F.softmax(self.raw_weights_highway, dim=0)
        elif env_type == 'urban':
            return F.softmax(self.raw_weights_urban, dim=0)
        elif env_type == 'weather':
            return F.softmax(self.raw_weights_weather, dim=0)

    def forward(self, env_type, trial_metrics):
        weights = self.get_weights(env_type)
        # Calculate AI: w1*m1 + w2*m2 + w3*m3 + w4*m4
        aggressiveness_index = torch.sum(weights * trial_metrics)
        return aggressiveness_index

# 2. Setup Training
agent = DynamicWeightAgent()
optimizer = optim.Adam(agent.parameters(), lr=0.05)

def print_weights(epoch, title=""):
    print(f"\n--- {title} (Epoch {epoch}) ---")
    hw = agent.get_weights('highway').detach().numpy()
    uw = agent.get_weights('urban').detach().numpy()
    ww = agent.get_weights('weather').detach().numpy()
    
    print(f"Highway -> THW: {hw[0]:.3f} | Weave: {hw[1]:.3f} | SpeedDelta: {hw[2]:.3f} | Brake: {hw[3]:.3f}")
    print(f"Urban   -> Occl: {uw[0]:.3f} | VRU: {uw[1]:.3f} | PET: {uw[2]:.3f} | Dilma: {uw[3]:.3f}")
    print(f"Weather -> Trac: {ww[0]:.3f} | Wet_HW: {ww[1]:.3f} | Steer: {ww[2]:.3f} | Hydro: {ww[3]:.3f}")

def train_step(env_type, metrics, truth):
    metrics_tensor = torch.tensor(metrics, dtype=torch.float32)
    truth_tensor = torch.tensor(truth, dtype=torch.float32)
    
    optimizer.zero_grad()
    predicted_ai = agent(env_type, metrics_tensor)
    loss = F.mse_loss(predicted_ai, truth_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()

# 3. Execution Loop
print_weights(0, "INITIAL RANDOM WEIGHTS")

epochs = 1000
for epoch in range(epochs):
    
    # We feed it data where specific parameters strongly correlate with crashes (1.0) or safety (0.0)
    
    # Highway: THW [index 0] and Weave [index 1] are highly dangerous here
    train_step('highway', [0.9, 0.8, 0.2, 0.1], 1.0) # Crash
    train_step('highway', [0.1, 0.1, 0.8, 0.8], 0.2) # High speed/brake, but good spacing = minor penalty
    
    # Urban: Occlusion [index 0] and VRU [index 1] are highly dangerous here
    train_step('urban', [0.9, 0.9, 0.1, 0.1], 1.0) # Near miss with pedestrian
    train_step('urban', [0.1, 0.1, 0.9, 0.9], 0.3) # Annoying driving, but no VRU risk
    
    # Weather: Traction [index 0] and Wet Headway [index 1] cause spins/crashes
    train_step('weather', [0.9, 0.8, 0.2, 0.1], 1.0) # Spinout
    train_step('weather', [0.1, 0.1, 0.5, 0.5], 0.1) # Safe driving

    # Print progress halfway
    if epoch == 500:
        print_weights(epoch, "MIDWAY TRAINING")

print_weights(epochs, "FINAL OPTIMIZED WEIGHTS")