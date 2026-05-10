import numpy as np

class AggressivenessModel:
    def __init__(self):
        # Mathematical Framework Weights (Tweak these for your report)
        self.w_speed = 0.5      
        self.w_accel = 0.2      
        self.w_prox = 0.8       
        self.w_wave = 0.3       

    def normalize_metrics(self, speed_kmh, accel_ms2, prox_m, wave_m):
        # 1. Normalize Speed (0 to 150 km/h)
        n_speed = np.clip(speed_kmh / 150.0, 0, 1.0)

        # 2. Normalize Acceleration (0 to 5 m/s^2)
        n_accel = np.clip(abs(accel_ms2) / 5.0, 0, 1.0)

        # 3. Normalize Proximity (Closer = Higher Aggression)
        safe_dist = 50.0
        if prox_m > safe_dist or prox_m < 0:
            n_prox = 0.0
        else:
            n_prox = 1.0 - (prox_m / safe_dist)

        # 4. Normalize Waviness (0 to 2 meters of lane drift)
        n_wave = np.clip(abs(wave_m) / 2.0, 0, 1.0)

        return n_speed, n_accel, n_prox, n_wave

    def calculate_index(self, speed, accel, prox, wave):
        n_speed, n_accel, n_prox, n_wave = self.normalize_metrics(speed, accel, prox, wave)
        
        # Calculate the raw Aggressiveness Index (AI)
        ai_score = (n_speed**2 * self.w_speed) + \
                   (n_accel * self.w_accel) + \
                   (n_prox * self.w_prox) + \
                   (n_wave * self.w_wave)
        
        # Scale to a cleaner 0-100 metric for your dataset
        max_possible_score = self.w_speed + self.w_accel + self.w_prox + self.w_wave
        ai_scaled = (ai_score / max_possible_score) * 100

        # Categorize the intent
        if ai_scaled < 30:
            category = "Conservative"
        elif ai_scaled < 65:
            category = "Normal"
        else:
            category = "Aggressive"

        return round(ai_scaled, 2), category