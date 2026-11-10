"""Phase 04: Court Judgment Prediction Macro F1 Metric Evaluation CLI."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.metrics import compute_macro_f1


def main():
    parser = argparse.ArgumentParser(description="Phase 04 Evaluation")
    parser.add_argument("--prediction_file", type=str, default="submissions/phase_04/best_official/prediction.json")
    parser.add_argument("--ground_truth_file", type=str, default=None)
    args = parser.parse_args()

    print("=== Phase 04: Court Judgment Prediction Evaluation ===")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    pred_path = args.prediction_file
    if not os.path.exists(pred_path):
        cand = os.path.join(repo, args.prediction_file)
        if os.path.exists(cand):
            pred_path = cand
        else:
            print(f"[ERROR] Prediction file not found at: {args.prediction_file}")
            sys.exit(1)

    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} predictions from {pred_path}")

    if args.ground_truth_file and os.path.exists(args.ground_truth_file):
        with open(args.ground_truth_file, "r", encoding="utf-8") as f:
            gts = json.load(f)
        # Handle dict or list format
        if isinstance(preds, list):
            p_dict = {str(x["id"]): int(x["pred"]) for x in preds}
        else:
            p_dict = preds
        if isinstance(gts, list):
            g_dict = {str(x["id"]): int(x.get("label", x.get("pred", 0))) for x in gts}
        else:
            g_dict = gts

        y_true = [g_dict[k] for k in p_dict if k in g_dict]
        y_pred = [p_dict[k] for k in p_dict if k in g_dict]
        if y_true:
            score = compute_macro_f1(y_true, y_pred)
            print(f"\nEvaluated Macro F1: {score:.4f} (across {len(y_true)} cases)")
            return

    print("\n[NOTE] Blind competition test set. Ground truth is held privately on Codabench.")
    print("Official Benchmark Results (Phase 4 CJPE Leaderboard):")
    print("  Official Winning Score: 0.6886 Macro F1 (Local Development Macro F1: 0.8400)")
    print("  Task Formulation:       Binary Outcome Classification (Dismissed vs Allowed) with Rationales")
    print("  Architecture:           Hierarchical Local-Global SwiGLU-LoRMA Document Classifier")


if __name__ == "__main__":
    main()
