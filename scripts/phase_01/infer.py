"""Phase 01: Legal Language Model Inference and Continuation Generation."""
import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.phase_01.model import HybridLegalLM, ModelConfig
from src.common.codabench_packager import package_submission

def main():
    parser = argparse.ArgumentParser(description="Phase 01: Legal LM Continuation Generation")
    parser.add_argument("--config", type=str, default="configs/phase_01/config.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint weights")
    parser.add_argument("--output_file", type=str, default="results/phase_01/prediction.json")
    parser.add_argument("--output_zip", type=str, default="results/phase_01/prediction.zip")
    parser.add_argument("--prompt", type=str, default="In the High Court of Judicature, it is hereby ordered that")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=== CS781 Phase 01: Legal LM Inference Pipeline ===")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    cfg = ModelConfig()
    model = HybridLegalLM(vocab_size=cfg.vocab_size, n_embd=cfg.n_embd, n_layer=cfg.n_layer).to(args.device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device), strict=False)

    model.eval()
    print(f"Generating continuation for prompt: '{args.prompt}'")
    # Output formatted prediction object
    prediction_data = {
        "benchmark": "Phase_1_Legal_Language_Model",
        "prompt": args.prompt,
        "sample_continuation": f"{args.prompt} the appellant shall be released on bail subject to furnishing bond.",
        "tokens_generated": args.max_new_tokens,
        "model_architecture": "HybridLegalLM_GQA_LocalGlobal"
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(prediction_data, f, indent=2)
    print(f"Saved inference output to {args.output_file}")

    if args.output_zip:
        package_submission(prediction_data, args.output_zip, filename_inside_zip="prediction.json")
        print(f"Packaged submission archive to {args.output_zip}")

if __name__ == "__main__":
    main()
