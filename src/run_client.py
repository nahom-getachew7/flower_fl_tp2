import argparse
from .data_utils import load_client_data
from .model import CustomFashionModel
from .client import CustomClient
import torch
import flwr as fl

def run_client(cid: int) -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    train_loader, val_loader = load_client_data(
        cid=cid,
        data_dir="./data/client_data",
        batch_size=32
    )
    
   
    model = CustomFashionModel().to(device)
    client = CustomClient(model, train_loader, val_loader, device)
    
    
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )
def client_fn(cid: str):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = load_client_data(
        cid=int(cid),
        data_dir="./data/client_data",
        batch_size=64
    )

    model = CustomFashionModel().to(device)
    client = CustomClient(model, train_loader, val_loader, device)
    return client.to_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FL client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    args = parser.parse_args()
    
    print(f"Starting client {args.cid}...")
    run_client(args.cid)