import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from typing import Tuple

def generate_distributed_datasets(k: int, alpha: float, save_dir: str) -> None:

    np.random.seed(42)
    torch.manual_seed(42)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    os.makedirs(save_dir, exist_ok=True)
    labels = full_dataset.targets.numpy()
    num_classes = len(full_dataset.classes)
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    client_indices = [[] for _ in range(k)]
    
    for class_idx in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, k))
        indices = class_indices[class_idx]
        np.random.shuffle(indices)
        splits = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        class_splits = np.split(indices, splits)
        
        for client_idx in range(k):
            client_indices[client_idx].extend(class_splits[client_idx])
    
    for client_idx in range(k):
        torch.save({
            'indices': client_indices[client_idx],
            'targets': labels[client_indices[client_idx]]
        }, os.path.join(save_dir, f'client_{client_idx}.pt'))

def load_client_data(cid: int, data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    
    torch.manual_seed(42)
    
    full_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transforms.ToTensor()
    )
    
    client_data = torch.load(os.path.join(data_dir, f'client_{cid}.pt'), weights_only=False)
    client_dataset = Subset(full_dataset, client_data['indices'])
    
    train_size = int(0.8 * len(client_dataset))
    val_size = len(client_dataset) - train_size
    train_dataset, val_dataset = random_split(
        client_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = transform
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size)
    )