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


def compare_train_val_accuracy(file_label_map: Dict[str, str], fig_dir: str = "./Compare/mu") -> None:
    os.makedirs(fig_dir, exist_ok=True)
    all_results = {}

    for file_path, label in file_label_map.items():
        all_results[label] = load_results(file_path)

    # Create a figure with subplots for each alpha value
    plt.figure(figsize=(18, 6))
    
    sorted_labels= sorted(file_label_map.values(), key=lambda x: float(x.split('=')[1]), reverse=True)


    # Plot for each alpha value
    for i, label in enumerate(sorted_labels, 1):
        data = all_results[label]

        # Get rounds
        rounds = [r[0] for r in data.get("metrics_distributed", [])]
        # Validation accuracy
        val_acc = [r[1].get("val_accuracy", 0.0) for r in data.get("metrics_distributed", [])]
        # Training accuracy
        train_acc = [r[1].get("train_accuracy", 0.0) for r in data.get("metrics_distributed_fit", [])]

        #create subplot
        plt.subplot(1, 3, i)
        plt.plot(rounds, val_acc,label='Validation Accuracy', marker = 'o', markersize = 4)
        plt.plot(rounds, train_acc,label='Training Accuracy', marker = 'o', markersize = 4)
        
        plt.title(f"Accuracy Comparison μ={label.split('=')[1]} (α=10)")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.0)  # Set y-axis limits to 0-1 for accuracy
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "train_val_accuracy_comparison_mu_fixed_alpha.png"))
    plt.close()


if __name__ == "__main__":
    file_label_map = {
        "result_alpha_10_mu_01.json": "μ=0.1",
        "result_alpha_10_mu_05.json": "μ=0.5",
        "result_alpha_10_mu_1.json": "μ=1"
    }

    compare_train_val_accuracy(file_label_map)