import flwr as fl
from flwr.server import Server
from flwr.server.history import History
from typing import Optional, Dict, List, Tuple
import json
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from .strategy import ScaffoldStrategy
from .client_manager import CustomClientManager
from .run_client import client_fn
from .strategy import ScaffoldStrategy

def convert_metrics(metrics: Dict[str, List[tuple[int, float]]]) -> List[tuple[int, Dict[str, float]]]:
   
    converted: Dict[int, Dict[str, float]] = {}
    for metric_name, metric_list in metrics.items():
        for round_num, val in metric_list:
            if round_num not in converted:
                converted[round_num] = {}
            converted[round_num][metric_name] = float(val)
    return sorted(converted.items())

def save_results(history: History, filename: str = "results.json") -> None:
   
    results = {
        "losses_distributed": [(rnd, float(loss)) for rnd, loss in history.losses_distributed],
        "metrics_distributed": convert_metrics(getattr(history, "metrics_distributed", {})),
        "metrics_distributed_fit": convert_metrics(getattr(history, "metrics_distributed_fit", {})),
        "losses_centralized": [(rnd, float(loss)) for rnd, loss in getattr(history, "losses_centralized", [])],
        "metrics_centralized": convert_metrics(getattr(history, "metrics_centralized", {}))
    }

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f" Results is saved to {filename}")

def run_server(
    server_address: str = "127.0.0.1:8080",
    num_rounds: int = 3,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    output_file: str = "results.json"
) -> History:
    
    
    config = fl.server.ServerConfig(num_rounds=num_rounds)

    if strategy is None:
        strategy = ScaffoldStrategy()
    if client_manager is None:
        client_manager = CustomClientManager()

    print(f"Starting server for {num_rounds} rounds...")
    history = fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
        client_manager=client_manager
    )

    
    save_results(history, output_file)

    return history

def run_simulation(
    num_clients: int,
    num_rounds: int,
    output_file: str = "results.json",
) -> History:
    """Run a Flower simulation with SCAFFOLD."""

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
        strategy=ScaffoldStrategy(),
    )
    save_results(history, output_file)
    return history
