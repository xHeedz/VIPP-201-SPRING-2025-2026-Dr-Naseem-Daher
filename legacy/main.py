import numpy as np
import random
import matplotlib.pyplot as plt
from simple_env import MockHighwayEnv  # <--- Importing the file you just made

# ==========================================
# 1. THE AGENT CLASS
# ==========================================
class AggressiveDriverAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.lr = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-Table: [Speed(3), Prox(3), Lane(2), Actions(3)]
        self.q_table = np.zeros((3, 3, 2, len(actions)))

        # --- THE WEIGHTS TO TEST ---
        # Adjust these to tune the "Aggressiveness"
        self.w_speed = 0.5      
        self.w_proximity = 0.8  
        self.w_accel = 0.1      
        self.w_waviness = 0.1   

    def normalize_metrics(self, speed, acceleration, distance_to_lead, lane_deviation):
        # 1. Normalize Speed (0-180 km/h)
        norm_speed = min(speed / 180, 1.0)

        # 2. Normalize Accel (0-5 m/s^2)
        norm_accel = min(abs(acceleration) / 5, 1.0)

        # 3. Normalize Proximity (0-50m safe distance)
        # Closer = Higher Score (1.0 = Bumper to Bumper)
        safe_dist = 50
        if distance_to_lead > safe_dist:
            norm_prox = 0.0
        else:
            norm_prox = 1.0 - (distance_to_lead / safe_dist)

        # 4. Normalize Waviness (0-1.75m deviation)
        norm_waviness = min(abs(lane_deviation) / 1.75, 1.0)

        return norm_speed, norm_accel, norm_prox, norm_waviness

    def get_discrete_state(self, n_speed, n_prox, n_wave):
        # Speed Bucket
        if n_speed < 0.3: s_s = 0
        elif n_speed < 0.7: s_s = 1
        else: s_s = 2
        
        # Proximity Bucket
        if n_prox < 0.3: s_p = 0
        elif n_prox < 0.7: s_p = 1
        else: s_p = 2
        
        # Waviness Bucket
        if n_wave < 0.5: s_w = 0
        else: s_w = 1
        
        return (s_s, s_p, s_w)

    def calculate_reward(self, n_speed, n_accel, n_prox, n_wave, crashed):
        if crashed: return -100 # Crash Penalty

        # The Aggressiveness Formula
        r = (n_speed**2 * self.w_speed) + \
            (n_accel * self.w_accel) + \
            (n_prox * self.w_proximity) - \
            (n_wave * self.w_waviness)
        return r

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state + (action,)]
        max_future_q = np.max(self.q_table[next_state])
        new_q = (1 - self.lr) * current_q + self.lr * (reward + self.gamma * max_future_q)
        self.q_table[state + (action,)] = new_q

# ==========================================
# 2. THE TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    env = MockHighwayEnv()
    agent = AggressiveDriverAgent(actions=[0, 1, 2]) # 0:Maintain, 1:Accel, 2:Brake

    proximity_history = []
    
    print("Training Agent to be Aggressive...")
    
    for episode in range(500):
        raw_state = env.reset()
        
        # Process State
        norm = agent.normalize_metrics(*raw_state)
        state = agent.get_discrete_state(*norm[:3])
        
        done = False
        while not done:
            action = agent.choose_action(state)
            
            # Step
            next_raw, crash, done = env.step(action)
            
            # Reward & Learn
            next_norm = agent.normalize_metrics(*next_raw)
            reward = agent.calculate_reward(*next_norm, crash)
            next_state = agent.get_discrete_state(*next_norm[:3])
            
            agent.learn(state, action, reward, next_state)
            
            state = next_state

            # Record distance for the last 50 episodes to visualize
            if episode > 450:
                proximity_history.append(next_raw[2])

    print("Done! Displaying results...")

    # Plot Results
    plt.figure(figsize=(10,4))
    plt.plot(proximity_history)
    plt.title("Distance to Lead Car (Last 50 Episodes)")
    plt.xlabel("Time Steps")
    plt.ylabel("Distance (m)")
    plt.axhline(y=0, color='r', linestyle='--', label="Crash")
    plt.axhline(y=10, color='g', linestyle='--', label="Target: Aggressive Zone")
    plt.legend()
    plt.grid(True)
    plt.show()



    