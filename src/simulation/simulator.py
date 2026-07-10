import gymnasium as gym
import highway_env
import pandas as pd
import numpy as np

from model import AggressivenessModel

def run_demo():
    print("--- Starting VIP Controlled Flow Simulation ---")

    env = gym.make("highway-v0", render_mode="human")

    # 1. UPDATED CONFIG: Added rendering and control limits to stop culling
    env.unwrapped.configure({
        "vehicles_count": 15,
        "duration": 100,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "screen_width": 1200,
        "screen_height": 400,
        "controlled_vehicles": 1,
        "offscreen_rendering": False,
    })

    model = AggressivenessModel()
    env.reset()
    dataset = []

    for step in range(2000):
        ego = env.unwrapped.vehicle
        
        # 1. SET EGO SPEED (The "Anchor")
        ego.target_speed = 20  # Constant 72 km/h
        
        # 2. CONTROL THE SURROUNDING TRAFFIC (The "Actors")
        for v in env.unwrapped.road.vehicles:
            if v is ego:
                continue
            
            dist = v.position[0] - ego.position[0]
            
            # Base speed adjustments (keep them close!)
            if dist > 0:  # Car is in front of you
                v.target_speed = ego.speed + 1.0
            else:         # Car is behind you
                v.target_speed = ego.speed + 1.5
                
            # Emergency Distance Management (The Safety Bubble)
            if 0 < dist < 15:      # Front car is too close? Match ego speed
                v.target_speed = ego.speed
            elif -15 < dist < 0:   # Back car is falling behind? Speed it up slightly
                v.target_speed = ego.speed + 2

            # HARD SAFETY CLAMP: Ensures no runaway cars and no despawning
            v.target_speed = np.clip(v.target_speed, 18, 22)

        # Ego Logic: Just stay in the lane and maintain target speed
        action = 1  # IDLE
        obs, reward, done, truncated, info = env.step(action)
        
        # 3. DATA COLLECTION
        for vehicle in env.unwrapped.road.vehicles:
            if vehicle is ego:
                continue 
            
            v_speed = vehicle.speed * 3.6
            v_prox = abs(vehicle.position[0] - ego.position[0])
            lane_center = round(vehicle.position[1] / 4) * 4
            v_wave = abs(vehicle.position[1] - lane_center)
            
            score, label = model.get_ai_score(v_speed, 0.1, v_prox, v_wave)
            dataset.append({"Waviness": v_wave, "AI_Score": score, "Category": label})
        
        if done or truncated:
            break

    env.close()
    pd.DataFrame(dataset).to_csv("demo_data.csv", index=False)
    print("--- Simulation Complete: Smooth Data Captured ---")

if __name__ == "__main__":
    run_demo()