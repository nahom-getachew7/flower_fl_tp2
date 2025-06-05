import argparse
from .data_utils import load_client_data
from .model import CustomFashionModel
from .client import CustomClient
import torch
import flwr as fl
from flwr.common import Context  # Fix deprecation

BATCH_SIZE = 64  # Standardized

def run_client(cid: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_client_data(
        cid=cid,
        data_dir="./data/client_data",
        batch_size=BATCH_SIZE
    )
    model = CustomFashionModel().to(device)
    client = CustomClient(model, train_loader, val_loader, device)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )

# Updated to use Context
def client_fn(cid: str) -> fl.client.Client:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_client_data(
        cid=int(cid),
        data_dir="./data/client_data",
        batch_size=BATCH_SIZE
    )
    model = CustomFashionModel().to(device)
    return CustomClient(model, train_loader, val_loader, device).to_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FL client")
    parser.add_argument("--cid", type=int, required=True)
    args = parser.parse_args()
    run_client(args.cid)