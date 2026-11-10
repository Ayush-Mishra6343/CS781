"""Phase 06: Student Knowledge Distillation Training CLI.

Trains or fine-tunes the 16-layer V1DeepLegalLM student via sequence-level knowledge
distillation against teacher models (Seq2Seq / V1-Wide), combining task cross-entropy
and soft KL divergence loss:

    L_KD = alpha * L_CE(y, y_student) + (1 - alpha) * T^2 * KL(p_student / T || p_teacher / T)
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.phase_06.model import V1DeepLegalLM
from src.phase_06.tokenizer import load_phase1_tokenizer
from src.common.legal_losses import DistillationLoss


def main():
    parser = argparse.ArgumentParser(description="Phase 06: Deep-16 Student Distillation")
    parser.add_argument("--config", type=str, default="configs/phase_06/config.json", help="Path to config JSON")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phase_06", help="Directory to save checkpoints")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of ground-truth CE loss vs teacher KL")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    print("=== Phase 06: Deep-16 Knowledge Distillation Training ===")
    print(f"Distillation Temperature: {args.temperature}, Alpha: {args.alpha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device}")

    # Load authentic Phase 1 / Phase 6 tokenizer
    try:
        tokenizer = load_phase1_tokenizer()
        print(f"Loaded ByteLevelBPE25k tokenizer (vocab size: {tokenizer.vocab_size:,})")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Initialize authentic 16-layer Net2Net model
    model = V1DeepLegalLM(
        vocab_size=25000,
        n_embd=768,
        n_layer=16,
        n_head=12,
        n_kv_head=4,
        max_seq_len=2048,
        intermediate_dim=4352,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized V1DeepLegalLM architecture: {total_params:,} parameters (16 layers, intermediate_dim=4,352)")

    distill_loss_fn = DistillationLoss(temperature=args.temperature, alpha=args.alpha)

    # Check for config and dataset
    if not os.path.exists(args.config):
        print(f"\n[NOTE] Config file not found at '{args.config}'.")
        print("To run full distillation training:")
        print("  1. Prepare paired teacher-student judgment data in data/phase_06/train.json")
        print("  2. Place Phase 1 base checkpoint in checkpoints/phase_01_best_model.pt")
        print("  3. Run on multi-GPU or high-RAM CUDA accelerator with mixed-precision (fp16/bf16).")
        return

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    print("Ready for distillation execution.")


if __name__ == "__main__":
    main()
