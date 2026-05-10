import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation():
    print("Loading dataset...")
    try:
        df = pd.read_csv("driving_behaviors_dataset.csv")
    except FileNotFoundError:
        print("❌ Error: 'driving_behaviors_dataset.csv' not found.")
        print("Make sure you run data_collector.py first to generate the data!")
        return

    # 1. Extract the relevant columns
    # We take the absolute value of waviness because drifting left (-y) 
    # or right (+y) both indicate unstable driving.
    waviness = np.abs(df["Waviness_m"]) 
    ai_scores = df["AI_Score"]
    categories = df["Category"]

    # 2. Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Color-code by category for a better visual in the report
    colors = {'Conservative': '#2ecc71', 'Normal': '#3498db', 'Aggressive': '#e74c3c'}
    
    # 3. Create the scatter plot
    plt.scatter(waviness, ai_scores, c=categories.map(colors), 
                alpha=0.6, edgecolors='w', s=60, label="Vehicle Data Points")

    # 4. Calculate and plot the trendline (Linear Regression)
    # np.polyfit calculates the slope (m) and intercept (b)
    m, b = np.polyfit(waviness, ai_scores, 1)
    trendline_x = np.linspace(waviness.min(), waviness.max(), 100)
    trendline_y = m * trendline_x + b
    
    plt.plot(trendline_x, trendline_y, color='black', linestyle='--', linewidth=2, 
             label=f"Trendline ($y = {m:.2f}x + {b:.2f}$)")

    # 5. Formatting for the Technical Report
    plt.title("Correlation Between Lane Waviness and Aggressiveness Index (AI)", fontsize=14, fontweight='bold')
    plt.xlabel("Absolute Lane Waviness (Lateral Velocity in m/s)", fontsize=12)
    plt.ylabel("Computed Aggressiveness Index (AI Score)", fontsize=12)
    
    # Customizing the legend to show the categories
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Aggressive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='Conservative'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Trendline')
    ]
    plt.legend(handles=custom_lines, loc="upper left")
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 6. Save and Show
    output_filename = "waviness_vs_ai_correlation.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Plot saved as '{output_filename}' ready for your report.")
    
    plt.show()

if __name__ == "__main__":
    plot_correlation()