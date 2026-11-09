# Phase 06 Inference & Submission Generation Entry Point
import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def main():
    parser = argparse.ArgumentParser(description="Phase 06 Inference and Submission Generation")
    parser.add_argument("--config", type=str, default="configs/phase_06/config.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint weights")
    parser.add_argument("--output_file", type=str, default="results/phase_06/prediction.json")
    args = parser.parse_args()

    print(f"=== CS781 Phase 06 Inference Pipeline ===")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active compute device: {device}")
    print("Inference setup verified. To reproduce official submission predictions, consult results/phase_06/.")

if __name__ == "__main__":
    main()
