import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR
import pandas as pd
import matplotlib.pyplot as plt

def generate_presentation_graphs():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "rl_training_history.csv"))
    except FileNotFoundError:
        print("[Error] Could not find 'rl_training_history.csv'. Make sure the agent finished training!")
        return

    # Create a 1x2 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Reinforcement Learning: Autonomous Driving Agent Performance", fontsize=18, fontweight='bold', y=1.05)

    # --- Plot 1: The Learning Curve (Rewards) ---
    # Using a rolling average to smooth out the noisy RL data
    window_size = 10
    smoothed_rewards = df['Reward'].rolling(window=window_size, min_periods=1).mean()
    
    axs[0].plot(df['Epoch'], df['Reward'], color='#bdc3c7', alpha=0.5, label='Raw Reward per Epoch')
    axs[0].plot(df['Epoch'], smoothed_rewards, color='#2ecc71', linewidth=3, label=f'{window_size}-Epoch Moving Avg')
    
    axs[0].set_title("Agent Survival & Efficiency (Reward)", fontsize=14)
    axs[0].set_xlabel("Training Epochs", fontsize=12)
    axs[0].set_ylabel("Cumulative Reward", fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(loc="lower right")

    # --- Plot 2: Evolution of Dynamic Weights (or training loss) ---
    # Weight columns are only present in dynamic-weight training histories;
    # the highway-env policy scripts log Epoch, Reward and Loss instead.
    weight_cols = ['W_Speed', 'W_Accel', 'W_Prox', 'W_Wave']
    if all(c in df.columns for c in weight_cols):
        axs[1].plot(df['Epoch'], df['W_Speed'], color='#e74c3c', linewidth=2, label='Weight: Speed')
        axs[1].plot(df['Epoch'], df['W_Accel'], color='#f39c12', linewidth=2, label='Weight: Acceleration')
        axs[1].plot(df['Epoch'], df['W_Prox'], color='#3498db', linewidth=2, label='Weight: Proximity (Safety)')
        axs[1].plot(df['Epoch'], df['W_Wave'], color='#9b59b6', linewidth=2, label='Weight: Waviness')
        axs[1].set_title("How the AI Shifted its Priorities", fontsize=14)
        axs[1].set_ylabel("Weight Value Distribution", fontsize=12)
        axs[1].legend(loc="center right")
    else:
        smoothed_loss = df['Loss'].rolling(window=window_size, min_periods=1).mean()
        axs[1].plot(df['Epoch'], df['Loss'], color='#bdc3c7', alpha=0.5, label='Raw Loss per Epoch')
        axs[1].plot(df['Epoch'], smoothed_loss, color='#3498db', linewidth=3, label=f'{window_size}-Epoch Moving Avg')
        axs[1].set_title("Policy Loss", fontsize=14)
        axs[1].set_ylabel("Loss", fontsize=12)
        axs[1].legend(loc="upper right")
    axs[1].set_xlabel("Training Epochs", fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "presentation_dashboard.png"), dpi=300, bbox_inches='tight')
    print("[Success] Presentation dashboard saved as 'presentation_dashboard.png'")
    plt.show()

if __name__ == "__main__":
    generate_presentation_graphs()