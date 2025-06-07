import argparse
from src.data_utils import generate_distributed_datasets
from src.server import run_server, run_simulation  # Add simulation
from src.run_client import run_client, client_fn  # Add client_fn

def main():
    parser = argparse.ArgumentParser(description="Federated Learning - FedProx")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Generate data
    data_parser = subparsers.add_parser("generate-data")
    data_parser.add_argument("--num-clients", type=int, default=10)
    data_parser.add_argument("--alpha", type=float, default=1.0)
    data_parser.add_argument("--data-dir", type=str, default="./data/client_data")
    
    # Server (Real deployment)
    server_parser = subparsers.add_parser("run-server")
    server_parser.add_argument("--address", type=str, default="127.0.0.1:8080")
    server_parser.add_argument("--rounds", type=int, default=3)
    server_parser.add_argument("--output", type=str, default="results.json")
    server_parser.add_argument("--mu", type=float, default=0.1)
    
    # Client (Real deployment)
    client_parser = subparsers.add_parser("run-client")
    client_parser.add_argument("--cid", type=int, required=True)
    
    
    # NEW: Simulation command
    sim_parser = subparsers.add_parser("simulate")
    sim_parser.add_argument("--num-clients", type=int, default=10)
    sim_parser.add_argument("--rounds", type=int, default=3)
    sim_parser.add_argument("--output", type=str, default="results.json")
    sim_parser.add_argument("--mu", type=float, default=0.1)
    
    args = parser.parse_args()
    
    if args.command == "generate-data":
        generate_distributed_datasets(args.num_clients, args.alpha, args.data_dir)
    
    elif args.command == "run-server":
        run_server(
            args.address, 
            args.rounds,
            output_file=args.output,
            mu=args.mu
        )
    
    elif args.command == "run-client":
        run_client(args.cid)
    
    elif args.command == "simulate":  # NEW: Handle simulation
        run_simulation(
            args.num_clients,
            args.rounds,
            args.output,
            args.mu
        )

if __name__ == "__main__":
    main()