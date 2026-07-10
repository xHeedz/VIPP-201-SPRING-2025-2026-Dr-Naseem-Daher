import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time 

# --- 1. THE AI BRAIN (Direct Control Unlocked) ---
class SmartDriverAI(nn.Module):
    def __init__(self, input_size=4, num_actions=5):
        super(SmartDriverAI, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        # Outputs probabilities for: 0:Left, 1:Idle, 2:Right, 3:Faster, 4:Slower
        return F.softmax(logits, dim=-1) 

# --- 2. THE TRAINING LOOP ---
def train_agent():
    print("--- Booting Incentive-Shaped RL Environment ---")
    
    env = gym.make("highway-v0", render_mode="human") 
    env.unwrapped.configure({
        "vehicles_count": 15, 
        "duration": 40,
        "simulation_frequency": 15,
        "policy_frequency": 5,
    })
    
    agent = SmartDriverAI()
    optimizer = optim.Adam(agent.parameters(), lr=0.01)
    
    epochs = 300
    history_rewards = []
    full_history = [] 

    for epoch in range(1, epochs + 1):
        obs, info = env.reset()
        done = truncated = False
        
        log_probs = []
        rewards = []

        # --- DYNAMIC TRAFFIC (Blue Cars) ---
        ego_vehicle = env.unwrapped.vehicle
        for v in env.unwrapped.road.vehicles:
            if v is not ego_vehicle:
                # Random speeds force the blue cars to weave and act naturally
                v.target_speed = np.random.uniform(15, 28) 

        while not (done or truncated):
            
            # Cinematic camera speed
            time.sleep(0.04) 
            
            ego_x, ego_y, ego_vx, ego_vy = obs[0][1], obs[0][2], obs[0][3], obs[0][4]
            
            prox = 50.0 
            for i in range(1, len(obs)):
                v_x = obs[i][1]
                if v_x > ego_x and (v_x - ego_x) < prox:
                    prox = v_x - ego_x

            state_tensor = torch.tensor([ego_vx/30.0, ego_vy/5.0, prox/50.0, abs(ego_y)/4.0], dtype=torch.float32)
            
            # AI outputs the 5 action probabilities directly
            action_probs = agent(state_tensor)
            
            # Categorical sampling lets the AI explore confidently
            m = torch.distributions.Categorical(action_probs) 
            action = m.sample() 
            log_prob = m.log_prob(action)
            
            obs, env_reward, done, truncated, info = env.step(action.item())
            
            # --- INCENTIVE SHAPING (The Behavior Design) ---
            custom_reward = 0.0
            if info.get('crashed', False):
                custom_reward -= 30.0 
            else:
                custom_reward += 1.0  # Base survival
                
                # The Need for Speed
                if ego_vx > 22.0:
                    custom_reward += 1.5
                elif ego_vx < 15.0:
                    custom_reward -= 1.0
                    
                # The Tailgating Penalty
                if prox < 12.0:        
                    custom_reward -= 2.0 
                    
                # THE STEERING TAX: Tiny penalty for changing lanes (0=Left, 2=Right)
                # Stops random swerving, but encourages it when blocked!
                if action.item() in [0, 2]:
                    custom_reward -= 0.5
            
            log_probs.append(log_prob)
            rewards.append(custom_reward)

        total_reward = sum(rewards)
        history_rewards.append(total_reward)
        
        # --- MATH FIX (Stabilized Backpropagation) ---
        gamma = 0.99
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean()
        
        loss = []
        for lp, R in zip(log_probs, returns):
            loss.append(-lp * R) 
        
        loss = torch.stack(loss).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        full_history.append([epoch, total_reward, loss.item()])

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Avg Reward (Last 10): {np.mean(history_rewards[-10:]):6.2f} | Loss: {loss.item():6.2f}")

    df_history = pd.DataFrame(full_history, columns=['Epoch', 'Reward', 'Loss'])
    df_history.to_csv("rl_training_history.csv", index=False)

    env.close()
    
    # Graphing
    plt.figure(figsize=(10, 5))
    smoothed_rewards = pd.Series(history_rewards).rolling(window=10, min_periods=1).mean()
    plt.plot(history_rewards, color="#bdc3c7", alpha=0.4, label="Raw Epoch Reward")
    plt.plot(smoothed_rewards, color="#3498db", linewidth=2, label="10-Epoch Trend")
    plt.title("Smart Agent Performance (Incentive Shaped)")
    plt.xlabel("Epochs")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("rl_learning_curve.png")
    print("\n[SYSTEM] Run complete. Saved to 'rl_learning_curve.png'.")

if __name__ == "__main__":
    train_agent()