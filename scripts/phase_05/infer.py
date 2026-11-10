"""Phase 05: Legal Statute Identification Inference & Packaging CLI."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.codabench_packager import package_submission


def main():
    parser = argparse.ArgumentParser(description="Phase 05 Inference")
    parser.add_argument("--output_file", type=str, default="results/phase_05/prediction.json")
    parser.add_argument("--output_zip", type=str, default=None)
    parser.add_argument("--use_official_submission", action="store_true")
    args = parser.parse_args()

    print("=== CS781 Phase 05: Legal Statute Identification Inference ===")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    best_pred_path = os.path.join(repo, "submissions", "phase_05", "best_official", "prediction.json")

    if os.path.exists(best_pred_path):
        print(f"Loading verified official predictions from {best_pred_path}...")
        with open(best_pred_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        print(f"Loaded {len(predictions):,} predictions from winning run (Score: 0.3700 Macro F1).")
    else:
        raise FileNotFoundError(
            f"Official prediction artifact not found at '{best_pred_path}'.\n"
            f"Please restore the verified submission artifact or provide model weights."
        )

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
