"""Phase 04: Court Judgment Prediction and Explanation Training CLI."""
import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.phase_04.model import build_classifier, CaseClassifier

def main():
    parser = argparse.ArgumentParser(description="Phase 04: CJPE Hierarchical Classifier Training")
    parser.add_argument("--config", type=str, default="configs/phase_04/config.json")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("CS781_DATA_DIR", "./data"))
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase_04")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    print("=== CS781 Phase 04: Hierarchical Local-Global SwiGLU-LoRMA Training ===")
    os.makedirs(args.output_dir, exist_ok=True)
    classifier = build_classifier()
    print("Successfully built CaseClassifier with hierarchical attention and LoRMA adapter.")

if __name__ == "__main__":
    main()
