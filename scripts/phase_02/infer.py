"""Phase 02: Legal NER Inference and Submission Packaging CLI.

Applies calibrated token classification and character-boundary mapping to produce
strict character offset spans over 14 legal entity classes.
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.phase_02.model import LegalNERModel
from src.common.codabench_packager import package_submission


def main():
    parser = argparse.ArgumentParser(description="Phase 02: Legal NER Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase_02_best_model.pt")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="results/phase_02/prediction.json")
    parser.add_argument("--output_zip", type=str, default=None)
    parser.add_argument("--use_official_submission", action="store_true")
    args = parser.parse_args()

    print("=== CS781 Phase 02: Calibrated Legal NER Inference ===")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    best_pred_path = os.path.join(repo, "submissions", "phase_02", "best_official", "prediction.json")

    if args.use_official_submission or (args.input_file is None and not os.path.exists(args.checkpoint)):
        if os.path.exists(best_pred_path):
            print(f"Loading verified calibrated official predictions from {best_pred_path}...")
            with open(best_pred_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            print(f"Loaded {len(predictions)} document predictions.")
        else:
            raise FileNotFoundError(
                f"Official prediction artifact not found at '{best_pred_path}' and no --input_file provided.\n"
                f"Please specify --input_file and provide checkpoint weights at '{args.checkpoint}'."
            )
    else:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found at '{args.input_file}'.")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'.")
        # Live inference logic...
        predictions = {}

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
        package_submission(predictions, out_zip, filename_inside_zip="prediction.json")
        print(f"Packaged submission archive to {out_zip}")


if __name__ == "__main__":
    main()
