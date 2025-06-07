import os
import json
import matplotlib.pyplot as plt
from typing import Dict

# Set base directory relative to the script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_results(file_path: str) -> Dict:
    full_path = os.path.join(RESULTS_DIR, file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    with open(full_path, "r") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"{file_path} is empty.")
        return json.loads(content)

def compare_multiple_results(file_label_map: Dict[str, str], fig_dir: str = "Comparison_figures") -> None:
    full_fig_dir = os.path.join(BASE_DIR, fig_dir, "alpha_10")
    os.makedirs(full_fig_dir, exist_ok=True)
    
    all_results = {}
    for file_path, label in file_label_map.items():
        try:
            all_results[label] = load_results(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue

    plt.figure(figsize=(16, 6))
    
    # Subplot 1: Training Accuracy
    plt.subplot(1, 2, 1)
    for label, data in all_results.items():
        rounds = [r[0] for r in data.get("metrics_distributed_fit", [])]
        train_acc = [r[1].get("train_accuracy", 0.0) for r in data.get("metrics_distributed_fit", [])]
        plt.plot(rounds, train_acc, label=f"{label}")
    
    plt.title("Training Accuracy Comparison (α=0.1)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Validation Accuracy
    plt.subplot(1, 2, 2)
    for label, data in all_results.items():
        rounds = [r[0] for r in data.get("metrics_distributed", [])]
        val_acc = [r[1].get("val_accuracy", 0.0) for r in data.get("metrics_distributed", [])]
        plt.plot(rounds, val_acc, label=f"{label}")
    
    plt.title("Validation Accuracy Comparison (α=0.1)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(full_fig_dir, "train_val_accuracy_comparison.png"))
    plt.close()

if __name__ == "__main__":
    file_label_map = {
        "FedAvg_alpha_10.json": "FedAvg",
        "FedProx_alpha_10_mu_01.json": "FedProx(μ=0.1)",
        "SCAFFOLD_alpha_10.json": "SCAFFOLD"
    }
    compare_multiple_results(file_label_map)