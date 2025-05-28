import argparse
from os import name
from src.data_utils import generate_distributed_datasets
from src.server import run_server
from src.visualizer import ResultsVisualizer
from src.run_client import run_client

def main():
    
    parser = argparse.ArgumentParser(description="Federated Learning TP1")
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
    
    
    # Visualization 
    vis_parser = subparsers.add_parser("visualize")
    vis_parser.add_argument("--results-file", type=str, default="results.json")
    vis_parser.add_argument("--output-dir", type=str, default="./figures")

    # Simulation + Visualization command (NEW)
    simvis_parser = subparsers.add_parser("simulate-and-visualize")
    simvis_parser.add_argument("--num-clients", type=int, default=2)
    simvis_parser.add_argument("--rounds", type=int, default=3)
    simvis_parser.add_argument("--results-file", type=str, default="results.json")
    simvis_parser.add_argument("--output-dir", type=str, default="./figures")
    
    args = parser.parse_args()
    
    if args.command == "generate-data":
        generate_distributed_datasets(args.num_clients, args.alpha, args.data_dir)
        print(f"Generated datasets for {args.num_clients} clients")

    elif args.command == "run-server":
        
        run_server(args.address, args.rounds,output_file=args.output)

    elif args.command == "run-client":
        run_client(args.cid)

    elif args.command == "visualize":
        visualizer = ResultsVisualizer()
        visualizer.load_simulation_results(args.results_file)
        visualizer.plot_results(args.output_dir)

if __name__ == "__main__":
    main()