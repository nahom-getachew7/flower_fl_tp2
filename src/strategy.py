from flwr.server.strategy import Strategy
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
    Scalar
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import time

class FedAvgStrategy(Strategy):
    def __init__(self, initial_parameters: Optional[Parameters] = None):
        self.initial_parameters = initial_parameters

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        print("Waiting for clients to connect...")
        timeout = 300  # Increased timeout to 5 minutes
        start_time = time.time()
        
        while client_manager.num_available() < 1:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise RuntimeError(f"Timeout after {timeout} seconds: No clients connected to server.")
            print(f"Waiting... ({elapsed:.1f}s elapsed, {client_manager.num_available()} clients connected)")
            time.sleep(5)  # Check every 5 seconds instead of 1
            
        print(f"Client connected! Proceeding with initialization.")
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
       
        
        sample_size = max(1, int(0.3 * client_manager.num_available()))
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=1,  
            criterion=None  
        )
        
        
        config = {
            "server_round": server_round,
            "epochs": 3,  
            "batch_size": 64,  
            "learning_rate": 0.01  
        }
        
        fit_ins = FitIns(parameters=parameters, config=config)
        
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        weights = []
        num_examples = []
        
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weights.append(ndarrays)
            num_examples.append(fit_res.num_examples)

        total_examples = sum(num_examples)
        averaged_weights = [
            sum(w[i] * n for w, n in zip(weights, num_examples)) / total_examples
            for i in range(len(weights[0]))
        ]
        
        avg_loss = np.mean([res.metrics["train_loss"] for _, res in results])
        avg_acc = np.mean([res.metrics["train_accuracy"] for _, res in results])

        return ndarrays_to_parameters(averaged_weights), {
            "train_loss": float(avg_loss),
            "train_accuracy": float(avg_acc)
        }

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        
        
        eval_clients = client_manager.sample(
            num_clients=max(2, int(0.2 * client_manager.num_available())),
            min_num_clients=1
        )
        
        config = {
            "server_round": server_round,
            "batch_size": 32  
        }
        
        evaluate_ins = EvaluateIns(parameters=parameters, config=config)
        return [(client, evaluate_ins) for client in eval_clients]
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = sum([res.num_examples for _, res in results])
        weighted_loss = sum([res.loss * res.num_examples for _, res in results]) / total_examples
        avg_accuracy = np.mean([res.metrics["val_accuracy"] for _, res in results])

        return float(weighted_loss), {
            "val_accuracy": float(avg_accuracy)
        }
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
       
        return None
    
    
class FedProxStrategy(FedAvgStrategy):
    def __init__(
        self, 
        initial_parameters: Optional[Parameters] = None,
        mu: float = 0.1  # Proximal term coefficient
    ):
        super().__init__(initial_parameters)
        self.mu = mu

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = super().configure_fit(server_round, parameters, client_manager)
        
        # Add mu to config for all clients
        return [
            (
                client,
                FitIns(
                    parameters=fit_ins.parameters,
                    config={**fit_ins.config, "mu": self.mu}
                )
            )
            for client, fit_ins in clients
        ]