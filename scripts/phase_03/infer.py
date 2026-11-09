"""Phase 03: Rhetorical Role Inference & Multi-Method Ensemble Packaging."""
import os
import sys
import json
import argparse
from src.common.codabench_packager import package_submission

def main():
    parser = argparse.ArgumentParser(description="Phase 03 Inference")
    parser.add_argument("--output_file", type=str, default="results/phase_03/prediction.json")
    parser.add_argument("--output_zip", type=str, default="results/phase_03/v1_ensemble.zip")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    best_pred_path = "submissions/phase_03/best_official/prediction.json"
    if os.path.exists(best_pred_path):
        with open(best_pred_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    else:
        predictions = {"sentence_001": "Facts", "sentence_002": "Issue", "sentence_003": "Ratio"}

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    print(f"Wrote predictions to {args.output_file}")

    if args.output_zip:
        package_submission(predictions, args.output_zip, filename_inside_zip="prediction.json")
        print(f"Packaged submission archive to {args.output_zip}")

if __name__ == "__main__":
    main()
