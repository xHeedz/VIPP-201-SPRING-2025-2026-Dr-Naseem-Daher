import gymnasium as gym
import highway_env
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. THE AGENT (Updated for Gym Actions)
# ==========================================
class AggressiveDriverAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        # Highway-Env Actions:
        # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
        self.actions = [0, 1, 2, 3, 4] 
        self.lr = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-Table: [Speed(3), Prox(3), Lane(2), Actions(5)]
        self.q_table = np.zeros((3, 3, 2, len(self.actions)))

        # --- TUNING WEIGHTS (Your Research) ---
        # We want the agent to be aggressive!
        self.w_speed = 0.8       # Love Speed
        self.w_proximity = 1.2   # Love Tailgating
        self.w_waviness = 0.1    # Don't care about weaving

    def get_discrete_state(self, speed_kmh, distance_m, lane_id):
        # 1. Bucket Speed (0-150 km/h)
        if speed_kmh < 60: s_speed = 0
        elif speed_kmh < 100: s_speed = 1
        else: s_speed = 2 # Fast
        
        # 2. Bucket Proximity (0-100m)
        if distance_m < 20: s_prox = 2   # Tailgating (Aggressive!)
        elif distance_m < 50: s_prox = 1 # Close
        else: s_prox = 0                 # Far
        
        # 3. Bucket Lane (Are we centered?)
        # In this env, lane_id is an integer (0, 1, 2), so we treat it as 'state'
        # Simplified: We just track if we are in the 'fast' lane (0) or others
        s_lane = 0 if lane_id == 0 else 1 

        return (s_speed, s_prox, s_lane)

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
# 2. HELPER: DECODE GYM OBSERVATIONS
# ==========================================
def parse_observation(obs):
    """
    Highway-Env returns a matrix of cars. We need to find 'us' and the 'lead car'.
    obs values are normalized: 
    - Position X is along the road. 
    - Position Y is the lane.
    - Speed Vx is forward speed.
    """
    # Row 0 is always the "Ego" (Our) Vehicle
    ego_x = obs[0, 1] 
    ego_y = obs[0, 2]
    ego_vx = obs[0, 3]
    
    # Real-world approximate conversions (Environment specific)
    speed_kmh = ego_vx * 100 # Approx conversion
    
    # Find the closest car in front of us (in the same lane)
    min_dist = 100.0 # Default max distance
    
    for i in range(1, len(obs)): # Loop through other cars
        car_x = obs[i, 1]
        car_y = obs[i, 2]
        
        # Check if car is in front of us (x > ego_x) 
        # AND in the same lane (abs(car_y - ego_y) is small)
        if car_x > ego_x and abs(car_y - ego_y) < 0.1:
            dist = (car_x - ego_x) * 100 # Approx meters
            if dist < min_dist:
                min_dist = dist
                
    return speed_kmh, min_dist, int(ego_y)

# ==========================================
# 3. MAIN SIMULATION LOOP
# ==========================================
if __name__ == "__main__":
    # Setup the Highway Environment
    # render_mode='human' opens a window so you can SEE it!
    env = gym.make("highway-v0", render_mode='rgb_array') 
    
    # Configure: More traffic, longer duration
    env.unwrapped.config.update({
        "simulation_frequency": 15,
        "duration": 40,
        "vehicles_count": 15, 
        "lanes_count": 3,
        "show_trajectories": False,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5, # Track us + 4 nearest cars
            "features": ["presence", "x", "y", "vx", "vy"],
        }
    })

    agent = AggressiveDriverAgent()
    rewards = []
    
    # We will just run 10 episodes to visualize, because 'render' is slow
    episodes = 10 
    
    print("Starting Highway Simulation...")
    print("Look for the window displaying the cars!")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # 1. Parse Data
            speed, dist, lane = parse_observation(obs)
            state = agent.get_discrete_state(speed, dist, lane)
            
            # 2. Decide
            action = agent.choose_action(state)
            
            # 3. Step
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 4. Custom Aggressive Reward (The Secret Sauce)
            # Highway-env gives points for safety. We OVERWRITE that.
            # We give points for Speed and Tailgating!
            speed, dist, lane = parse_observation(next_obs)
            
            # AGGRESSIVE REWARD FUNCTION
            custom_reward = (speed / 100.0) * agent.w_speed
            if dist < 20: 
                custom_reward += agent.w_proximity # Bonus for being close!
            
            if done: custom_reward = -10 # Crash penalty
            
            # 5. Learn
            next_state = agent.get_discrete_state(speed, dist, lane)
            agent.learn(state, action, custom_reward, next_state)
            
            obs = next_obs
            total_reward += custom_reward
            
            # Render the environment
            env.render()
        
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")

    env.close()