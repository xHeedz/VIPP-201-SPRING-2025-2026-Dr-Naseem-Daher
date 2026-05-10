import numpy as np

class AggressivenessModel:
    def __init__(self):
        self.w_speed = 0.5      
        self.w_accel = 0.2      
        self.w_prox = 0.8       
        self.w_wave = 0.4 # Slightly higher weight for the presentation plot

    def normalize(self, speed_kmh, accel_ms2, prox_m, wave_m):
        n_speed = min(speed_kmh / 150.0, 1.0)
        n_accel = min(abs(accel_ms2) / 5.0, 1.0)
        n_prox = 1.0 - (prox_m / 50.0) if 0 < prox_m <= 50 else 0.0
        n_wave = min(abs(wave_m) / 1.5, 1.0) 
        return n_speed, n_accel, n_prox, n_wave

    def get_ai_score(self, speed, accel, prox, wave):
        ns, na, np_dist, nw = self.normalize(speed, accel, prox, wave)
        score = (ns**2 * self.w_speed) + (na * self.w_accel) + (np_dist**2 * self.w_prox) + (nw * self.w_wave)
        final_score = min(score * 100, 100)
        
        if final_score < 35: return final_score, "Conservative"
        elif final_score < 70: return final_score, "Normal"
        else: return final_score, "Aggressive"