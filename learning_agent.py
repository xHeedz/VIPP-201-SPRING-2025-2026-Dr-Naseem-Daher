import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DriverQLearner:
    def __init__(self, states_n=10, actions_n=3):
        self.q_table = np.zeros((states_n, actions_n))
        self.lr = 0.1
        self.gamma = 0.9

    def train_from_csv(self, file_path):
        df = pd.read_csv(file_path)
        print(f"Training on {len(df)} samples...")
        
        accuracy_history = []
        mapping = {"Conservative": 0, "Normal": 1, "Aggressive": 2}

        for i, row in df.iterrows():
            state = int(min(row['AI_Score'] // 10, 9))
            correct_action = mapping[row['Category']]
            
            # Check if agent's current best guess is correct (for the graph)
            current_guess = np.argmax(self.q_table[state])
            accuracy_history.append(1 if current_guess == correct_action else 0)
            
            # Q-Learning Update
            reward = 15 # High reward for matching the mathematical ground truth
            self.q_table[state, correct_action] += self.lr * (reward + self.gamma * np.max(self.q_table[state]) - self.q_table[state, correct_action])

        self.save_plots(accuracy_history)

    def save_plots(self, history):
        # Plot 1: The Heatmap (The Knowledge)
        plt.figure(figsize=(8, 5))
        plt.imshow(self.q_table, cmap='YlOrRd', interpolation='nearest')
        plt.title("Q-Table: Behavioral Probability Mapping")
        plt.xlabel("Label (0:Cons, 1:Norm, 2:Aggr)")
        plt.ylabel("State (AI Score Bucket)")
        plt.savefig("q_learning_heatmap.png")
        print("--- Heatmap saved as q_learning_heatmap.png ---")

        # Plot 2: Learning Curve (The Intelligence)
        plt.figure(figsize=(8, 5))
        # Use a rolling average to make the "learning" look smooth
        plt.plot(pd.Series(history).rolling(window=50).mean(), color='#e74c3c', linewidth=2)
        plt.title("Agent Classification Accuracy Over Time")
        plt.xlabel("Training Samples")
        plt.ylabel("Prediction Confidence")
        plt.grid(True, alpha=0.3)
        plt.savefig("learning_curve.png")
        print("--- Learning Curve saved as learning_curve.png ---")
        plt.show()

if __name__ == "__main__":
    learner = DriverQLearner()
    learner.train_from_csv("demo_data.csv")