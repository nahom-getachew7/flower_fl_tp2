from typing import Dict, List, Optional, Tuple
import numpy as np
from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes, GetParametersIns, Parameters, Scalar
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy

class ScaffoldStrategy(Strategy):
    """Flower strategy implementing SCAFFOLD (Stochastic Controlled Averaging)."""

    def __init__(
        self,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 0,
        min_evaluate_clients: int = 0,
    ):
        """Initialize the SCAFFOLD strategy."""
        # Client sampling parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients

        # Control variates
        self.server_control: List[np.ndarray] = []
        # Client-specific control variates (ci for each client id)
        self.client_controls: Dict[str, List[np.ndarray]] = {}

        # Number of model parameter arrays (to split parameters vs control variates)
        self.num_model_params: Optional[int] = None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Wait until at least one client is connected, then request parameters."""
        import time
        timeout = 60  # seconds
        start_time = time.time()

        while client_manager.num_available() < 1:
            if time.time() - start_time > timeout:
                raise RuntimeError("Timeout: No clients connected.")
            time.sleep(1)

        client = client_manager.sample(1)[0]
        get_parameters_ins = GetParametersIns(config={})
        parameters_res = client.get_parameters(ins=get_parameters_ins,
                                                timeout=10.0,
                                                group_id=""
                                            )
        weights = parameters_to_ndarrays(parameters_res.parameters)

        self.server_control = [np.zeros_like(w) for w in weights]
        self.num_model_params = len(weights)

        return parameters_res.parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training by sampling clients and sending current model and control variates."""
        # Convert current global model parameters to numpy arrays
        weights = parameters_to_ndarrays(parameters)

        # Initialize server control if empty
        if not self.server_control:
            self.server_control = [np.zeros_like(w) for w in weights]
        # Ensure num_model_params is set
        if self.num_model_params is None:
            self.num_model_params = len(weights)

        # Sample clients for training
        num_clients = client_manager.num_available()
        num_sample = max(int(self.fraction_fit * num_clients), self.min_fit_clients)
        num_sample = min(num_sample, num_clients)
        
        clients = client_manager.sample(
                num_clients=num_sample,
                min_num_clients=self.min_fit_clients,
                criterion=None
        )
        # Prepare FitIns for each selected client
        instructions = []
        for client in clients:
            cid = client.cid
            # Initialize client control variate if seeing client first time
            if cid not in self.client_controls:
                self.client_controls[cid] = [np.zeros_like(w) for w in weights]
            client_control = self.client_controls[cid]
            # Combine parameters: [model_weights, server_control, client_control]
            combined_weights = weights + self.server_control + client_control
            parameters_with_control = ndarrays_to_parameters(combined_weights)
            # Create FitIns with combined parameters
            fit_ins = FitIns(parameters=parameters_with_control, config={})
            instructions.append((client, fit_ins))
        return instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results and update global model and control variates."""
        if not results:
            return None, {}

        # Sum number of examples for weighting
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        # Prepare accumulators
        sum_model_params = [np.zeros_like(c) for c in self.server_control]
        sum_control_updates = [np.zeros_like(c) for c in self.server_control]

        # Iterate over client results
        for client, fit_res in results:
            cid = client.cid
            res_weights = parameters_to_ndarrays(fit_res.parameters)
            
            # Split parameters
            model_params = res_weights[:self.num_model_params]
            client_control_update = res_weights[2*self.num_model_params:3*self.num_model_params]
            
            # Accumulate model parameters (example-weighted)
            for idx, w in enumerate(model_params):
                sum_model_params[idx] += w * fit_res.num_examples
                
            # Accumulate control updates (uniform weighting)
            for idx, cv in enumerate(client_control_update):
                sum_control_updates[idx] += cv
                
            # Update stored client control
            if cid in self.client_controls:
                for idx in range(len(self.client_controls[cid])):
                    self.client_controls[cid][idx] += client_control_update[idx]

        # Compute weighted average of model parameters
        new_global_weights = [param_sum / total_examples for param_sum in sum_model_params]
        
        # Compute average control update (uniform average)
        avg_control_update = [
            cv_sum / len(results)
            for cv_sum in sum_control_updates
        ]
        
        # Update server control variate
        total_clients = len(self.client_controls)
        cv_multiplier = len(results) / total_clients if total_clients > 0 else 1.0
        for idx in range(len(self.server_control)):
            self.server_control[idx] += cv_multiplier * avg_control_update[idx]

        # Create Parameters object
        aggregated_parameters = ndarrays_to_parameters(new_global_weights)

        # Aggregate metrics
        aggregated_metrics: Dict[str, float] = {}
        for _, fit_res in results:
            if fit_res.metrics is None:
                continue
            for key, value in fit_res.metrics.items():
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + value * fit_res.num_examples
        for key in aggregated_metrics:
            aggregated_metrics[key] /= total_examples

        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        num_clients = client_manager.num_available()
        num_sample = max(int(self.fraction_evaluate * num_clients), self.min_evaluate_clients)
        num_sample = min(num_sample, num_clients)
        
        clients = client_manager.sample(
            num_clients=num_sample,
            min_num_clients=self.min_evaluate_clients,
            criterion=None
        )
        # Send only model parameters for evaluation
        evaluate_instructions = [(client, EvaluateIns(parameters, {})) for client in clients]
        return evaluate_instructions

    def aggregate_evaluate(
        self,
        server_round: int,  # Add this
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes]],  # Add this
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy and loss across clients."""
        if not results:
            return None, {}

        # Aggregate losses
        total_loss = 0.0
        total_examples = 0
        for client_proxy, evaluate_res in results:
            if evaluate_res.loss is not None:
                total_loss += evaluate_res.loss * evaluate_res.num_examples
                total_examples += evaluate_res.num_examples

        # Aggregate metrics (including val_accuracy)
        metrics_aggregated = {}
        if total_examples > 0:
            metrics_aggregated["val_loss"] = total_loss / total_examples

            # Calculate weighted average of val_accuracy
            val_accuracies = [r.metrics.get("val_accuracy", 0) * r.num_examples 
                            for _, r in results if "val_accuracy" in r.metrics]
            total_val_examples = sum(r.num_examples for _, r in results 
                                if "val_accuracy" in r.metrics)
            
            if total_val_examples > 0:
                metrics_aggregated["val_accuracy"] = sum(val_accuracies) / total_val_examples

        return total_loss / total_examples if total_examples > 0 else None, metrics_aggregated
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None