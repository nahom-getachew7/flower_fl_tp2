import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader
import torch

class CustomFashionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        global_params: Optional[List[np.ndarray]] = None,  # New param
        mu: float = 0.0                                    # New param
    ) -> Tuple[float, float]:
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Convert global params to tensors if provided
        global_tensors = None
        if global_params is not None and mu > 0:
            global_tensors = [
                torch.tensor(arr).to(device) 
                for arr in global_params
            ]
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            
            # Add proximal term if mu > 0
            if mu > 0 and global_tensors is not None:
                proximal_term = 0.0
                for param, global_t in zip(self.parameters(), global_tensors):
                    proximal_term += torch.norm(param - global_t, p=2) ** 2
                loss = loss + (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        return avg_loss, accuracy

    def test_epoch(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[float, float]:
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        return avg_loss, accuracy

    def get_model_parameters(self) -> List[np.ndarray]:
        return [param.cpu().detach().numpy() for param in self.parameters()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict)