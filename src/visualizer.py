import json
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from typing import Dict, List, Tuple


class ResultsVisualizer:
    def init(self) -> None:
        self.results: Dict = {}

    def load_simulation_results(self, file_name: str):
        with open(file_name, "r") as f:
            content = f.read().strip()
        if not content:
            raise ValueError(f"{file_name} is empty.")
        self.results = json.loads(content)

    def plot_results(self, fig_directory: str = "./figures") -> None:
        os.makedirs(fig_directory, exist_ok=True)

        
        loss_rounds = [r[0] for r in self.results.get("losses_distributed", [])]
        loss_values = [r[1] for r in self.results.get("losses_distributed", [])]

       
        val_data = self.results.get("metrics_distributed", [])
        val_rounds = [r[0] for r in val_data]
        val_accuracies = [r[1].get("val_accuracy", 0.0) for r in val_data]

        
        train_data = self.results.get("metrics_distributed_fit", [])
        train_rounds = [r[0] for r in train_data]
        train_accuracies = [r[1].get("train_accuracy", 0.0) for r in train_data]

        
        drift_values = [r[1].get("client_drift", 0.0) for r in train_data]

        # Loss plot
        plt.figure()
        plt.plot(loss_rounds, loss_values, 'b-o', label="Loss")
        plt.title("Loss per Round")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{fig_directory}/loss.png")
        plt.close()

        
        plt.figure()
        plt.plot(val_rounds, val_accuracies, 'g-o', label="Validation Accuracy")
        plt.title("Validation Accuracy per Round")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{fig_directory}/val_accuracy.png")
        plt.close()

        
        plt.figure()
        plt.plot(train_rounds, train_accuracies, 'r-o', label="Training Accuracy")
        plt.title("Training Accuracy per Round")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{fig_directory}/train_accuracy.png")
        plt.close()

       
        if any(drift_values):
            plt.figure()
            plt.plot(train_rounds, drift_values, 'm-o', label="Client Drift")
            plt.title("Client Drift per Round")
            plt.xlabel("Round")
            plt.ylabel("Drift Value")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{fig_directory}/client_drift.png")
            plt.close()