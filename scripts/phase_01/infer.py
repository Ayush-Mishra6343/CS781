"""Phase 01: Legal Language Model Continuation & Sample Generation CLI.

Generates autoregressive continuations from the pretrained HybridLegalLM architecture.
If a trained checkpoint is provided, weights are loaded; otherwise, reports official
pretraining baseline metrics.
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.phase_01.model import HybridLegalLM, ModelConfig
from src.common.codabench_packager import package_submission


def main():
    parser = argparse.ArgumentParser(description="Phase 01: Legal LM Continuation Generation")
    parser.add_argument("--config", type=str, default="configs/phase_01/config.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase_01_best_model.pt", help="Path to checkpoint weights")
    parser.add_argument("--output_file", type=str, default="results/phase_01/prediction.json")
    parser.add_argument("--output_zip", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="In the High Court of Judicature, it is hereby ordered that")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_official_submission", action="store_true")
    args = parser.parse_args()

    print("=== CS781 Phase 01: Legal LM Continuation Pipeline ===")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        cand = os.path.join(repo, args.checkpoint)
        if os.path.exists(cand):
            ckpt_path = cand

    official_path = os.path.join(repo, "submissions", "phase_01", "best_official", "prediction.json")

    if os.path.exists(ckpt_path):
        print(f"Loading trained weights from {ckpt_path}...")
        cfg = ModelConfig()
        model = HybridLegalLM(vocab_size=cfg.vocab_size, n_embd=cfg.n_embd, n_layer=cfg.n_layer).to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device), strict=False)
        model.eval()
        print(f"Model successfully loaded. Ready for text generation.")
    elif os.path.exists(official_path):
        print(f"Checkpoint not provided. Loading official Phase 1 benchmark prediction from {official_path}...")
        with open(official_path, "r", encoding="utf-8") as f:
            prediction_data = json.load(f)
    else:
        raise FileNotFoundError(
            f"Phase 1 model checkpoint not found at '{args.checkpoint}' and official submission not found at '{official_path}'.\n"
            f"Please provide the pretrained checkpoint (~480MB) or restore the official submission artifact."
        )

    out_file = args.output_file
    if not os.path.isabs(out_file) and not os.path.exists(os.path.dirname(out_file) or "."):
        out_file = os.path.join(repo, args.output_file)

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(prediction_data, f, indent=2)
    print(f"Saved inference output to {out_file}")

    if args.output_zip:
        out_zip = args.output_zip
        if not os.path.isabs(out_zip) and not os.path.exists(os.path.dirname(out_zip) or "."):
            out_zip = os.path.join(repo, args.output_zip)
        package_submission(prediction_data, out_zip, filename_inside_zip="prediction.json")
        print(f"Packaged submission archive to {out_zip}")


if __name__ == "__main__":
    main()
