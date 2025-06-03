from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import numpy as np
from flwr.common import (
    GetPropertiesIns, GetPropertiesRes,
    GetParametersIns, GetParametersRes,
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    Parameters, ndarrays_to_parameters,
    parameters_to_ndarrays, Scalar, Code, Status
)
from flwr.client import Client
from torch.utils.data import DataLoader


class CustomClient(Client):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        # Initialize client control variate
        self.client_control: Optional[List[np.ndarray]] = None

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={}
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )

    def fit(self, ins: FitIns) -> FitRes:
        # Unpack parameters: [model_weights, server_control, client_control]
        full_params = parameters_to_ndarrays(ins.parameters)
        num_model_params = len(full_params) // 3
        model_weights = full_params[:num_model_params]
        server_control = full_params[num_model_params:2 * num_model_params]
        client_control_old = full_params[2 * num_model_params:]

        # Set model weights and store initial parameters
        self.model.set_model_parameters(model_weights)
        initial_weights = self.model.get_model_parameters()  # w0

        # Initialize client control if None
        if self.client_control is None:
            self.client_control = [np.zeros_like(w) for w in model_weights]

        # Convert control variates to tensors for gradient correction
        correction_tensors = [
            torch.tensor(c_i - c_s, dtype=torch.float32, device=self.device)
            for c_i, c_s in zip(client_control_old, server_control)
        ]

        # Train model with per-batch gradient correction
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        num_batches = 0
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Apply gradient correction
            with torch.no_grad():
                for param, corr in zip(self.model.parameters(), correction_tensors):
                    if param.grad is not None:
                        param.grad -= corr
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            num_batches += 1

        # Compute updated weights (wT)
        updated_weights = self.model.get_model_parameters()
        
        # Calculate control update: Δc_i = -c_s + (w0 - wT)/(Kη)
        K = num_batches  # Number of local steps
        eta = 0.01  # Learning rate
        control_update = [
            (w0 - wT) / (K * eta) - c_s
            for w0, wT, c_s in zip(initial_weights, updated_weights, server_control)
        ]
        
        # Update client control: c_i^{new} = c_i^{old} + Δc_i
        self.client_control = [
            c_old + delta 
            for c_old, delta in zip(client_control_old, control_update)
        ]

        # Return [model_weights, server_control, control_update]
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(
                updated_weights + server_control + control_update
            ),
            num_examples=len(self.train_loader.dataset),
            metrics={"train_loss": total_loss/num_batches, "train_accuracy": correct/total}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        loss, accuracy = self.model.test_epoch(
            self.test_loader, criterion, self.device
        )

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.test_loader.dataset),
            metrics={"val_accuracy": accuracy}
        )

    def to_client(self) -> 'CustomClient':
        return self
