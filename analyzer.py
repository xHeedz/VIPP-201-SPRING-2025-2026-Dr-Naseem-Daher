import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_results():
    try:
        df = pd.read_csv("demo_data.csv")
    except:
        print("Error: Run simulator.py first!")
        return

    plt.figure(figsize=(10, 6))
    
    # Using a scatter plot with specific colors for AUB presentation
    sns.scatterplot(data=df, x="Waviness", y="AI_Score", hue="Category", 
                    palette={"Aggressive": "#e74c3c", "Normal": "#3498db", "Conservative": "#2ecc71"},
                    alpha=0.6)
    
    plt.title("Correlation: Lateral Waviness vs. Aggressiveness Index", fontsize=14)
    plt.xlabel("Lane Deviation (Waviness in Meters)", fontsize=12)
    plt.ylabel("Computed AI Score (0-100)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("results_graph.png")
    print("--- Graph saved as results_graph.png ---")
    plt.show()

if __name__ == "__main__":
    show_results()