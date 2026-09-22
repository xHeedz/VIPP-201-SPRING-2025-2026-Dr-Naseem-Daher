import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- 1. THE SIMULATED BRAIN MATH ---
def simulate_neural_network(speed, distance):
    # Normalize inputs (just like your state_tensor)
    n_speed = speed / 30.0
    n_prox = distance / 50.0
    
    # Simulate the Neural Network's hidden logic
    # If distance gets tiny, the safety instinct spikes massively
    logit_prox = (1.0 - n_prox) * 6.0 
    
    # If the road is clear and we are moving, speed instinct takes over
    logit_speed = (n_speed * 2.0) + (n_prox * 4.0)
    
    logit_accel = 1.0  # Baseline
    logit_wave = 0.5   # Baseline
    
    logits = np.array([logit_speed, logit_accel, logit_prox, logit_wave])
    
    # Softmax function (forces them all to equal 1.0 or 100%)
    exp_logits = np.exp(logits)
    weights = exp_logits / np.sum(exp_logits)
    return weights

# --- 2. SET UP THE DASHBOARD ---
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.35) # Make room for sliders

labels = ['Speed Weight', 'Accel Weight', 'Safety (Prox) Weight', 'Lane (Wave) Weight']
colors = ['#3498db', '#f39c12', '#e74c3c', '#9b59b6']

# Starting conditions: Going 25 m/s, car ahead is 40 meters away
init_speed = 25.0
init_dist = 40.0
init_weights = simulate_neural_network(init_speed, init_dist)

bars = ax.bar(labels, init_weights, color=colors)
ax.set_ylim(0, 1.0)
ax.set_ylabel('Weight Probability (0.0 to 1.0)', fontsize=12)
ax.set_title('Live AI Brain Simulation: How Sensors Alter Weights', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

#  3. add interactive sliders 
ax_speed = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_dist = plt.axes([0.15, 0.1, 0.7, 0.03])

slider_speed = Slider(ax_speed, 'Ego Speed (m/s)', 0.0, 30.0, valinit=init_speed, color='#2ecc71')
slider_dist = Slider(ax_dist, 'Distance to Car Ahead (m)', 0.0, 50.0, valinit=init_dist, color='#e74c3c')

# 4. update loop 
def update(val):
    # When you move a slider, recalculate the neural network!
    weights = simulate_neural_network(slider_speed.val, slider_dist.val)
    for bar, w in zip(bars, weights):
        bar.set_height(w)
    fig.canvas.draw_idle()

slider_speed.on_changed(update)
slider_dist.on_changed(update)

print("[SYSTEM] Booting Interactive Dashboard...")
plt.show()