import numpy as np

class MockHighwayEnv:
    def __init__(self):
        # Physics Constants
        self.dt = 1.0  # Time step (1 second)
        self.lead_car_speed = 100 / 3.6  # 100 km/h in m/s (Constant)
        
        self.reset()

    def reset(self):
        """ Resets the car to a safe starting state """
        self.ego_speed = 80 / 3.6    # Start at 80 km/h
        self.distance = 100.0        # Start 100m behind
        self.lane_deviation = 0.0    # Start centered
        self.steps = 0
        return self.get_state_metrics()

    def step(self, action):
        """ 
        Simulates one second of driving. 
        Action 0: Maintain
        Action 1: Accelerate (+2 m/s^2)
        Action 2: Brake (-2 m/s^2)
        """
        self.steps += 1
        
        # 1. Apply Physics based on Action
        accel = 0
        if action == 1: accel = 2.0
        elif action == 2: accel = -2.0
        
        # Update Speed (v = u + at)
        self.ego_speed += accel * self.dt
        self.ego_speed = max(0, min(self.ego_speed, 180/3.6)) # Clamp 0-180 km/h

        # Update Distance (d = d_old + (v_lead - v_ego) * t)
        relative_speed = self.lead_car_speed - self.ego_speed
        self.distance += relative_speed * self.dt
        
        # Update Lane Waviness (Random drift for simulation)
        # In real life, this depends on steering. Here, we just add noise.
        self.lane_deviation += np.random.uniform(-0.1, 0.1)

        # 2. Check for "Done" (Crash or too far)
        done = False
        crash = False
        
        if self.distance <= 0:
            crash = True
            done = True
            self.distance = 0 # Crash!
            
        if self.steps > 100: # End episode after 100 seconds
            done = True

        # 3. Return Raw Metrics
        return self.get_state_metrics(), crash, done

    def get_state_metrics(self):
        # Return raw values: speed (km/h), accel, distance (m), deviation (m)
        return self.ego_speed * 3.6, 0.0, self.distance, self.lane_deviation