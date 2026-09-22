import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR

# Ensure we can import modules from the test2 directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import gymnasium as gym
import highway_env
import pandas as pd
from aggressiveness_index import AggressivenessModel

def get_proximity(vehicle, all_vehicles):
    """Finds the distance to the closest car directly in front of this vehicle."""
    min_dist = 100.0 # Max tracking distance
    for other in all_vehicles:
        if other is vehicle:
            continue
            
        # Check if the other car is in the same lane (Y axis) and ahead (X axis)
        if abs(other.position[1] - vehicle.position[1]) < 2.0: 
            dist = other.position[0] - vehicle.position[0]
            if 0 < dist < min_dist:
                min_dist = dist
    return min_dist

def run_collection():
    print("Initializing Simulation...")
    # Using 'rgb_array' so it runs silently and fast in the background
    env = gym.make("highway-v0", render_mode="rgb_array")
    
    env.unwrapped.config.update({
        "simulation_frequency": 15,  # 15 frames per second
        "duration": 60,              # Run for 60 simulated seconds
        "vehicles_count": 20,        # Traffic density
        "lanes_count": 3
    })

    dt = 1.0 / 15.0 # Time step
    model = AggressivenessModel()
    dataset = []
    
    # Dictionary to remember the previous speed of each car to calculate acceleration
    previous_speeds = {} 

    obs, info = env.reset()
    done = truncated = False
    step = 0

    print("Collecting data. Please wait...")
    
    while not (done or truncated):
        # We bypass the RL observation and look directly at the physics engine
        vehicles = env.unwrapped.road.vehicles
        
        for v in vehicles:
            v_id = id(v) # Unique memory address for each car
            
            speed_kmh = v.speed * 3.6
            wave_m = v.velocity[1] # Lateral speed (weaving)
            prox_m = get_proximity(v, vehicles)
            
            # Calculate Acceleration: a = (v_current - v_previous) / dt
            accel_ms2 = 0.0
            if v_id in previous_speeds:
                accel_ms2 = (v.speed - previous_speeds[v_id]) / dt
            
            previous_speeds[v_id] = v.speed
            
            # Skip the first frame for each car since we don't have an acceleration value yet
            if accel_ms2 != 0.0: 
                ai_score, category = model.calculate_index(speed_kmh, accel_ms2, prox_m, wave_m)
                
                dataset.append({
                    "Timestamp": step,
                    "Vehicle_ID": str(v_id)[-5:], # Shorten ID for readability
                    "Speed_kmh": round(speed_kmh, 2),
                    "Accel_ms2": round(accel_ms2, 2),
                    "Proximity_m": round(prox_m, 2),
                    "Waviness_m": round(wave_m, 2),
                    "AI_Score": ai_score,
                    "Category": category
                })
        
        # The agent does nothing, we just let traffic flow naturally
        action = 1 # IDLE
        obs, reward, done, truncated, info = env.step(action)
        step += 1

    env.close()
    
    # Save the deliverables
    df = pd.DataFrame(dataset)
    output_path = os.path.join(DATA_DIR, "driving_behaviors_dataset.csv")
    df.to_csv(output_path, index=False)
    print("✅ Collection Complete! Saved to 'driving_behaviors_dataset.csv'")
    print(df.head())

if __name__ == "__main__":
    run_collection()