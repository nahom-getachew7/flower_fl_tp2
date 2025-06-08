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


def compare_fedprox_results(file_label_map: Dict[str, str], fig_dir: str = "./Compare/mu") -> None:
    os.makedirs(fig_dir, exist_ok=True)
    all_results = {}

    for file_path, label in file_label_map.items():
        all_results[label] = load_results(file_path)

    # Create a figure with two subplots (one for accuracy, one for loss)
    plt.figure(figsize=(16, 6))
    
    # Subplot 1: Validation Accuracy
    plt.subplot(1, 2, 1)
    for label, data in all_results.items():
        rounds = [r[0] for r in data.get("metrics_distributed", [])]
        val_acc = [r[1].get("val_accuracy", 0.0) for r in data.get("metrics_distributed", [])]
        plt.plot(rounds, val_acc, label=f"{label}")
    
    plt.title("FedProx Validation Accuracy Comparison (α=10.0)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Validation Loss
    plt.subplot(1, 2, 2)
    for label, data in all_results.items():
        rounds = [r[0] for r in data.get("metrics_distributed", [])]
        val_loss = [r[1].get("val_loss", 0.0) for r in data.get("metrics_distributed", [])]
        plt.plot(rounds, val_loss, label=f"{label}")
    
    plt.title("FedProx Validation Loss Comparison (α=10.0)")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fedprox_mu_comparison.png"))
    plt.close()


if __name__ == "__main__":
    file_label_map = {
        "result_alpha_10_mu_01.json": "μ=0.1",
        "result_alpha_10_mu_05.json": "μ=0.5",
        "result_alpha_10_mu_1.json": "μ=1.0"
    }

    compare_fedprox_results(file_label_map)