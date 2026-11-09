"""Phase 06: Student Knowledge Distillation Training CLI."""
import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.legal_losses import DistillationLoss

def main():
    parser = argparse.ArgumentParser(description="Phase 06: Deep-16 Student Distillation")
    parser.add_argument("--config", type=str, default="configs/phase_06/config.json")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    print("=== CS781 Phase 06: Deep-16 Student Knowledge Distillation ===")
    loss_fn = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
    print(f"DistillationLoss configured with T={args.temperature}, alpha={args.alpha}")

if __name__ == "__main__":
    main()
