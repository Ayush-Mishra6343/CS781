"""Phase 01: Language Model Evaluation CLI (Perplexity, Loss, Token Accuracy)."""

import argparse
import json
import math
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    parser = argparse.ArgumentParser(description="Phase 01 Metric Evaluation")
    parser.add_argument("--prediction_file", type=str, default="submissions/phase_01/best_official/prediction.json")
    parser.add_argument("--ground_truth_file", type=str, default=None)
    args = parser.parse_args()

    print("=== Phase 01: Legal Language Model Evaluation ===")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    pred_path = args.prediction_file
    if not os.path.exists(pred_path):
        cand = os.path.join(repo, args.prediction_file)
        if os.path.exists(cand):
            pred_path = cand

    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded submission record from {pred_path}")
    else:
        print(f"[ERROR] Prediction file not found at: {args.prediction_file}")
        sys.exit(1)

    if args.ground_truth_file and os.path.exists(args.ground_truth_file):
        with open(args.ground_truth_file, "r", encoding="utf-8") as f:
            gts = json.load(f)
        # Compute exact token accuracy if matching prompts/continuations
        correct, total = 0, 0
        for k in data:
            if k in gts:
                pred_tokens = str(data[k]).split()
                gold_tokens = str(gts[k]).split()
                for pt, gt in zip(pred_tokens, gold_tokens):
                    if pt == gt:
                        correct += 1
                    total += 1
        if total > 0:
            acc = correct / total
            print(f"\nEvaluated Token Accuracy: {acc:.4f} (over {total} tokens)")
            return

    print("\n[NOTE] Blind competition test set: Ground-truth labels are held privately by the Codabench server.")
    print("Local evaluation requires ground-truth labels (--ground_truth_file <path>).")
    print("To inspect official competition submission scores, see: results/official_submissions.csv")


if __name__ == "__main__":
    main()
