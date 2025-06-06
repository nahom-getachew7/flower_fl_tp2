import argparse
from src.data_utils import generate_distributed_datasets
from src.server import run_server, run_simulation
from src.visualizer import ResultsVisualizer
from src.run_client import client_fn, run_client  # Import client_fn for simulation
from src.strategy import ScaffoldStrategy  # Import your custom strategy
import flwr as fl
from torch import manual_seed
from numpy.random import seed as np_seed

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    manual_seed(seed)          # PyTorch
    np_seed(seed)              # NumPy

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Federated Learning TP2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    #Generate data
    data_parser = subparsers.add_parser("generate-data")
    data_parser.add_argument("--num-clients", type=int, default=10)
    data_parser.add_argument("--alpha", type=float, default=1.0)
    data_parser.add_argument("--data-dir", type=str, default="./data/client_data")
    
    # Server 
    server_parser = subparsers.add_parser("run-server")
    server_parser.add_argument("--address", type=str, default="127.0.0.1:8080")
    server_parser.add_argument("--rounds", type=int, default=3)
    server_parser.add_argument("--output", type=str, default="results.json")
    
    # Client 
    client_parser = subparsers.add_parser("run-client")
    client_parser.add_argument("--cid", type=int, required=True, help="Client ID (0 to num-clients-1)")
    

    # New: Simulation command
    sim_parser = subparsers.add_parser("simulate")
    sim_parser.add_argument("--num-clients", type=int, default=10)
    sim_parser.add_argument("--rounds", type=int, default=3)
    sim_parser.add_argument("--output", type=str, default="results.json")
    
    args = parser.parse_args()
    
    if args.command == "generate-data":
        generate_distributed_datasets(args.num_clients, args.alpha, args.data_dir)
        print(f"Generated datasets for {args.num_clients} clients")

    elif args.command == "run-server":
        
        run_server(args.address, args.rounds,output_file=args.output)

    elif args.command == "run-client":
        run_client(args.cid)

    elif args.command == "simulate":
        run_simulation(args.num_clients, args.rounds, args.output)

if __name__ == "__main__":
    main()