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
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        
        # Extract mu from config (default to 0 for FedAvg)
        mu = ins.config.get("mu", 0.0)
        
        # Save global parameters BEFORE training (convert to numpy)
        global_params = self.model.get_model_parameters()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=ins.config.get("learning_rate", 0.01)
        )
        
        loss, accuracy = self.model.train_epoch(
            self.train_loader,
            criterion,
            optimizer,
            self.device,
            global_params=global_params,  # Pass global params
            mu=mu                       # Pass mu
        )
        
        parameters = self.model.get_model_parameters()
        parameters_prime = ndarrays_to_parameters(parameters)
        
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters_prime,
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
            metrics={"val_loss": loss, "val_accuracy": accuracy}
        )

    def to_client(self) -> 'CustomClient':
        return self