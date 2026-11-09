# Phase 02 Training Entry Point
import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def main():
    parser = argparse.ArgumentParser(description="Phase 02 Model Training")
    parser.add_argument("--config", type=str, default="configs/phase_02/config.json", help="Path to config.json")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("CS781_DATA_DIR", "/kaggle/input/cs781-legal-dataset"))
    parser.add_argument("--output_dir", type=str, default="output/phase_02")
    args = parser.parse_args()

    print(f"=== CS781 Phase 02 Training Pipeline ===")
    print(f"Loading configuration from: {args.config}")
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        print(f"Configuration loaded: {list(cfg.keys())}")
    else:
        print("Warning: Config file not found at specified path.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active compute device: {device}")
    print("Training initialization completed. For full interactive training, execute notebooks/phase_02_*/.")

if __name__ == "__main__":
    main()
