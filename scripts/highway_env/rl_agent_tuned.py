import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from paths import DATA_DIR, FIG_DIR
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

# --- 1. THE AI BRAIN ---
class SmartDriverAI(nn.Module):
    def __init__(self, input_size=4, num_actions=5):
        super(SmartDriverAI, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1) 

# --- 2. THE TRAINING LOOP ---
def train_agent():
    print("--- Booting Academically Tuned RL Environment ---")
    
    # Set to "human" to watch, None for fast training
    env = gym.make("highway-v0", render_mode="human") 
    env.unwrapped.configure({
        "vehicles_count": 15, 
        "duration": 40,
        "simulation_frequency": 15,
        "policy_frequency": 5,
    })
    
    agent = SmartDriverAI()
    optimizer = optim.Adam(agent.parameters(), lr=0.005) # Lowered LR for stability
    
    epochs = 300
    history_rewards = []
    full_history = [] 
    
    # Academic Hyperparameters
    initial_noise = 5.0
    min_noise = 0.01
    decay_rate = 0.015
    entropy_beta = 0.01

    for epoch in range(1, epochs + 1):
        obs, info = env.reset()
        done = truncated = False
        
        log_probs = []
        rewards = []
        action_probs_list = []

        # Exponential Decay of Exploration Noise
        exploration_noise = min_noise + (initial_noise - min_noise) * np.exp(-decay_rate * epoch)

        # Dynamic Traffic
        ego_vehicle = env.unwrapped.vehicle
        for v in env.unwrapped.road.vehicles:
            if v is not ego_vehicle:
                v.target_speed = np.random.uniform(15, 28) 

        while not (done or truncated):
            time.sleep(0.04) # Cinematic camera speed
            
            ego_x, ego_y, ego_vx, ego_vy = obs[0][1], obs[0][2], obs[0][3], obs[0][4]
            
            prox = 50.0 
            v_ahead_speed = 20.0
            for i in range(1, len(obs)):
                v_x, v_vx = obs[i][1], obs[i][3]
                if v_x > ego_x and (v_x - ego_x) < prox:
                    prox = v_x - ego_x
                    v_ahead_speed = v_vx

            # 1. State Tensor Standardization (Tanh)
            norm_speed = np.tanh((ego_vx - 20.0) / 10.0) 
            norm_accel = np.tanh(ego_vy / 2.0)
            norm_prox = np.tanh((prox - 20.0) / 15.0)     
            norm_wave = np.tanh(ego_y / 2.0)

            state_tensor = torch.tensor([norm_speed, norm_accel, norm_prox, norm_wave], dtype=torch.float32)
            
            action_probs = agent(state_tensor)
            action_probs_list.append(action_probs)
            
            # Apply decaying noise
            m = torch.distributions.Categorical(probs=(action_probs + exploration_noise) / (1.0 + exploration_noise * 5)) 
            action = m.sample() 
            log_prob = m.log_prob(action)
            
            obs, env_reward, done, truncated, info = env.step(action.item())
            
            # 2. Risk-Aware Reward Shaping (TTC Penalty)
            custom_reward = 0.0
            if info.get('crashed', False):
                custom_reward -= 30.0 
            else:
                custom_reward += 1.0  
                if ego_vx > 22.0: custom_reward += 1.5
                elif ego_vx < 15.0: custom_reward -= 1.0
                
                # Time-To-Collision exponential math
                rel_velocity = ego_vx - v_ahead_speed
                if rel_velocity > 0.5 and prox < 40.0:
                    ttc = prox / rel_velocity
                    tau = 2.5 
                    custom_reward -= np.exp(-ttc / tau) * 4.0 
                    
                # Steering Tax
                if action.item() in [0, 2]:
                    custom_reward -= 0.5
            
            log_probs.append(log_prob)
            rewards.append(custom_reward)

        total_reward = sum(rewards)
        history_rewards.append(total_reward)
        
        # Backpropagation
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
        
        policy_loss = []
        for lp, R in zip(log_probs, returns):
            policy_loss.append(-lp * R) 
        policy_loss = torch.stack(policy_loss).sum()
        
        # 3. Entropy Regularization Bonus
        action_probs_tensor = torch.stack(action_probs_list)
        entropy = -torch.sum(action_probs_tensor * torch.log(action_probs_tensor + 1e-9))
        
        # Final combined loss
        total_loss = policy_loss - (entropy_beta * entropy)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        full_history.append([epoch, total_reward, total_loss.item()])

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Avg Reward: {np.mean(history_rewards[-10:]):6.2f} | Noise: {exploration_noise:.3f}")

    df_history = pd.DataFrame(full_history, columns=['Epoch', 'Reward', 'Loss'])
    df_history.to_csv(os.path.join(DATA_DIR, "rl_training_history.csv"), index=False)
    env.close()
    
    # Graphing
    plt.figure(figsize=(10, 5))
    smoothed_rewards = pd.Series(history_rewards).rolling(window=10, min_periods=1).mean()
    plt.plot(history_rewards, color="#bdc3c7", alpha=0.4, label="Raw Reward")
    plt.plot(smoothed_rewards, color="#27ae60", linewidth=2, label="10-Epoch Trend (Tuned)")
    plt.title("Academically Tuned RL Agent Performance")
    plt.xlabel("Epochs")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "rl_learning_curve.png"))
    print("\n[SYSTEM] Run complete. Saved to 'rl_learning_curve.png'.")

if __name__ == "__main__":
    train_agent()