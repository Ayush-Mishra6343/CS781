"""Phase 06: Summarization Inference CLI.

Executes live autoregressive generation if model checkpoint weights and test inputs
are provided, or loads the verified official competition submission for packaging.
Placeholder strings are strictly prohibited.
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.phase_06.model import V1DeepLegalLM
from src.phase_06.tokenizer import load_phase1_tokenizer
from src.phase_06.mbr_decoder import compute_mbr_consensus
from src.common.codabench_packager import package_submission


def generate_summary(model, tokenizer, prompt_text: str, max_new_tokens: int = 128, device: str = "cpu") -> str:
    """Performs greedy autoregressive token generation."""
    prompt_tokens = tokenizer.encode(prompt_text)[:1024]
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            next_token_id = int(torch.argmax(next_token_logits).item())
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
            # Break on EOS if applicable (e.g. token 2)
            if next_token_id == 2:
                break

    generated_ids = input_ids[0, len(prompt_tokens):].tolist()
    return tokenizer.decode(generated_ids).strip()


def main():
    parser = argparse.ArgumentParser(description="Phase 06 Inference")
    parser.add_argument("--test_file", type=str, default=None, help="Path to test dataset JSON")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/phase_06_best_model.pt", help="Path to checkpoint weights")
    parser.add_argument("--output_file", type=str, default="results/phase_06/prediction.json", help="Destination path for predictions")
    parser.add_argument("--output_zip", type=str, default=None, help="Optional zip packaging path")
    parser.add_argument("--use_official_submission", action="store_true", help="Load verified official submission predictions")
    args = parser.parse_args()

    print("=== Phase 06: Deep-16 Summarization Inference ===")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Mode 1: Load and inspect official verified submission
    if args.use_official_submission or (args.test_file is None and not os.path.exists(args.checkpoint_path)):
        best_pred_path = os.path.join(repo, "submissions", "phase_06", "best_official", "prediction.json")
        if os.path.exists(best_pred_path):
            print(f"[INFO] Loading verified official submission predictions from {best_pred_path}...")
            with open(best_pred_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            print(f"Loaded {len(predictions)} verified test predictions.")

            out_file = args.output_file
            if not os.path.isabs(out_file) and not os.path.exists(os.path.dirname(out_file) or "."):
                out_file = os.path.join(repo, args.output_file)
            os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2)
            print(f"Wrote predictions to {out_file}")

            if args.output_zip:
                out_zip = args.output_zip
                if not os.path.isabs(out_zip) and not os.path.exists(os.path.dirname(out_zip) or "."):
                    out_zip = os.path.join(repo, args.output_zip)
                os.makedirs(os.path.dirname(out_zip) or ".", exist_ok=True)
                package_submission(predictions, out_zip, filename_inside_zip="prediction.json")
                print(f"Packaged submission archive to {out_zip}")
            return
        else:
            raise FileNotFoundError(
                f"Official submission artifact not found at '{best_pred_path}'.\n"
                f"To run live inference, provide test data via --test_file and model weights via --checkpoint_path."
            )

    # Mode 2: Live generation from checkpoint
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test dataset not found at '{args.test_file}'.")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(
            f"Phase 6 model checkpoint not found at '{args.checkpoint_path}'.\n"
            f"Trained Deep-16 weights (~250MB) are archived externally.\n"
            f"Please supply the checkpoint file or use --use_official_submission to inspect the competition output."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizer and model on device: {device}")
    tokenizer = load_phase1_tokenizer()

    model = V1DeepLegalLM(vocab_size=25000, n_layer=16, intermediate_dim=4352).to(device)
    state = torch.load(args.checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    predictions = {}
    print(f"Generating live summaries for {len(test_data)} cases...")
    for item in test_data:
        qid = str(item.get("id", item.get("judgment_id", "")))
        text = item.get("text", item.get("judgment_text", ""))
        summary = generate_summary(model, tokenizer, text, max_new_tokens=128, device=device)
        predictions[qid] = summary

    out_file = args.output_file
    if not os.path.isabs(out_file) and not os.path.exists(os.path.dirname(out_file) or "."):
        out_file = os.path.join(repo, args.output_file)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"Wrote generated predictions to {out_file}")

    if args.output_zip:
        out_zip = args.output_zip
        if not os.path.isabs(out_zip) and not os.path.exists(os.path.dirname(out_zip) or "."):
            out_zip = os.path.join(repo, args.output_zip)
        os.makedirs(os.path.dirname(out_zip) or ".", exist_ok=True)
        package_submission(predictions, out_zip, filename_inside_zip="prediction.json")
        print(f"Packaged submission archive to {out_zip}")


if __name__ == "__main__":
    main()
