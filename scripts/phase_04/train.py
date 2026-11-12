"""Phase 04: Court Judgment Prediction and Explanation Model Setup CLI Scaffold.

Initializes the Hierarchical Local-Global SwiGLU-LoRMA CaseClassifier architecture.
Full hierarchical document training across long judicial decisions was executed in the
Phase 4 Kaggle notebook environment.
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.phase_04.model import HybridLegalLM, apply_lora_and_head, CaseClassifier


def main():
    parser = argparse.ArgumentParser(description="Phase 04: CJPE Hierarchical Classifier Setup")
    parser.add_argument("--config", type=str, default="configs/phase_04/config.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional Phase 1 base checkpoint")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("CS781_DATA_DIR", "./data"))
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase_04")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    print("=== CS781 Phase 04: Hierarchical Local-Global SwiGLU-LoRMA Setup ===")
    os.makedirs(args.output_dir, exist_ok=True)

    backbone = HybridLegalLM(
        vocab_size=25000,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_kv_head=4,
        max_seq_len=1024,
        ffn_multiplier=4,
        dropout=0.0,
    )
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        backbone.load_state_dict(state, strict=False)
        print(f"Loaded backbone weights from: {args.checkpoint}")

    classifier = apply_lora_and_head(backbone, rank=8, alpha=16.0)
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Successfully built CaseClassifier: {total_params:,} total parameters ({trainable_params:,} trainable LoRA/head params).")
    print("See notebooks/phase_04_judgment_prediction/00_BEST_OFFICIAL_IMPLEMENTATION.ipynb for full hierarchical training.")


if __name__ == "__main__":
    main()
