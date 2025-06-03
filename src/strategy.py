from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes, GetParametersIns, Parameters, Scalar
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

class ScaffoldStrategy(Strategy):
    """Flower strategy implementing SCAFFOLD (Stochastic Controlled Averaging)."""

    def __init__(
        self,
        fraction_fit: float = 1.0,
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
    ) -> List[Tuple]:
        """Configure the next round of training by sampling clients and sending current model and control variates."""
        # Convert current global model parameters to numpy arrays
        weights = parameters_to_ndarrays(parameters)

        # Initialize server control if empty (e.g., if no initial parameters were set in initialize_parameters)
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
            # Create FitIns with combined parameters (control variates included)
            fit_ins = FitIns(parameters=parameters_with_control, config={})
            instructions.append((client, fit_ins))
        return instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Tuple],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results and update global model and control variates."""
        if not results:
            return None, {}

        # Sum number of examples for weighting
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        # Prepare accumulators for weighted sum of model parameters and control updates
        # We use the server_control list as a template for shapes
        sum_model_params = [np.zeros_like(c) for c in self.server_control]
        sum_control_updates = [np.zeros_like(c) for c in self.server_control]

        # Iterate over client results
        for client, fit_res in results:
            cid = client.cid
            # Extract returned parameter arrays
            res_weights = parameters_to_ndarrays(fit_res.parameters)
            # The returned list is [model_weights, server_control, client_control_update]
            # Split them according to num_model_params
            model_params = res_weights[: self.num_model_params]
            # The server control part (middle) is ignored (clients typically leave it unchanged)
            client_control_update = res_weights[self.num_model_params * 2 : self.num_model_params * 3]
            # Accumulate weighted sum of model parameters
            for idx, w in enumerate(model_params):
                sum_model_params[idx] += w * fit_res.num_examples
            # Accumulate weighted sum of client control updates
            for idx, cv in enumerate(client_control_update):
                sum_control_updates[idx] += cv * fit_res.num_examples
            # Update stored client control variate (ci = old_ci + cv_update)
            if cid in self.client_controls:
                for idx in range(len(self.client_controls[cid])):
                    self.client_controls[cid][idx] += client_control_update[idx]

        # Compute weighted average of model parameters (global update)
        new_global_weights = [param_sum / total_examples for param_sum in sum_model_params]

        # Compute weighted average of control updates
        avg_control_update = [cv_sum / total_examples for cv_sum in sum_control_updates]
        # Update server control variate: c = c + (fraction)*avg_control_update
        total_clients = len(self.client_controls) if self.client_controls else len(results)
        cv_multiplier = len(results) / total_clients if total_clients > 0 else 1.0
        for idx in range(len(self.server_control)):
            self.server_control[idx] = self.server_control[idx] + cv_multiplier * avg_control_update[idx]

        # Create Parameters object for new global model
        aggregated_parameters = ndarrays_to_parameters(new_global_weights)

        # Aggregate metrics (e.g., training loss) by weighted average if provided
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
    ) -> List[Tuple]:
        """Configure the next round of evaluation."""
        num_clients = client_manager.num_available()
        num_sample = max(int(self.fraction_evaluate * num_clients), self.min_evaluate_clients)
        num_sample = min(num_sample, num_clients)
        
        clients = client_manager.sample(
            num_clients=num_sample,
            min_num_clients=self.min_evaluate_clients,
            criterion=None
    )
        # Send only model parameters for evaluation (no control variates)
        evaluate_instructions = [(client, EvaluateIns(parameters, {})) for client in clients]
        return evaluate_instructions

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Tuple],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Sum weighted loss over clients
        total_examples = 0
        sum_loss = 0.0
        for _, eval_res in results:
            total_examples += eval_res.num_examples
            if eval_res.loss is not None:
                sum_loss += eval_res.loss * eval_res.num_examples

        # Compute averaged loss
        averaged_loss = sum_loss / total_examples if total_examples > 0 else None
        return averaged_loss, {"num_examples": total_examples}
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None