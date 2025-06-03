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
        full_params = parameters_to_ndarrays(ins.parameters)

        num_model_params = len(full_params) // 3
        model_weights = full_params[:num_model_params]
        server_control = full_params[num_model_params:2 * num_model_params]
        client_control = full_params[2 * num_model_params:]

        # Set model weights
        self.model.set_model_parameters(model_weights)

        # Store previous client control
        if self.client_control is None:
            self.client_control = [np.zeros_like(p) for p in model_weights]

        # SCAFFOLD-corrected gradient (FedDyn-style)
        for p, c_s, c_c in zip(self.model.parameters(), server_control, client_control):
            p.grad = None  # clear existing gradient
            if p.requires_grad:
                p.data -= 0.01 * (c_s - c_c).astype(np.float32)  # SCAFFOLD correction

        # Train model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        loss, accuracy = self.model.train_epoch(
            self.train_loader, criterion, optimizer, self.device
        )

        # Get updated model weights
        new_weights = self.model.get_model_parameters()

        # Compute client control update: ci_new - ci_old
        control_update = [new - old for new, old in zip(self.client_control, client_control)]

        # Update stored control variate
        self.client_control = [c + u for c, u in zip(client_control, control_update)]

        # Return combined parameters: [model_weights, server_control, control_update]
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(new_weights + server_control + control_update),
            num_examples=len(self.train_loader.dataset),
            metrics={"train_loss": loss, "train_accuracy": accuracy}
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
