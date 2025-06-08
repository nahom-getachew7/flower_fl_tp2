import os
import json
import matplotlib.pyplot as plt
from typing import Dict

def load_results(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"{file_path} is empty.")
        return json.loads(content)

def compare_multiple_results(file_label_map: Dict[str, str], fig_dir: str = "./Compare/alpha") -> None:
    os.makedirs(fig_dir, exist_ok=True)
    all_results = {}

    for file_path, label in file_label_map.items():
        all_results[label] = load_results(file_path)

    # Create a figure with 3 subplots (one for each alpha)
    plt.figure(figsize=(18, 6))
    
    # Sort the alphas to ensure consistent ordering (high to low)
    sorted_labels = sorted(file_label_map.values(), key=lambda x: float(x.split('=')[1]), reverse=True)
    
    for i, label in enumerate(sorted_labels, 1):
        data = all_results[label]
        
        # Get the rounds and metrics
        rounds = [r[0] for r in data.get("metrics_distributed", [])]
        val_acc = [r[1].get("val_accuracy", 0.0) for r in data.get("metrics_distributed", [])]
        train_acc = [r[1].get("train_accuracy", 0.0) for r in data.get("metrics_distributed_fit", [])]
        
        # Create subplot
        plt.subplot(1, 3, i)
        plt.plot(rounds, val_acc, label='Validation Accuracy', marker='o', markersize=4)
        plt.plot(rounds, train_acc, label='Training Accuracy', marker='s', markersize=4)
        
        plt.title(f"Accuracy Comparison ({label})")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.0)  # Set consistent y-axis for comparison
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "train_val_accuracy_comparison.png"))
    plt.close()

if __name__ == "__main__":
    print("This is compare_train_val_accuracy")
    file_label_map = {
        "results_alpha_10.json": "α=10.0",
        "results_alpha_1.json": "α=1.0",
        "results_alpha_01.json": "α=0.1"
    }

    compare_multiple_results(file_label_map)