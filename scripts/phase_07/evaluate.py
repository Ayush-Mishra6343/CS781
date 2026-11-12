"""Phase 07: Prior Case Retrieval Metric Evaluation CLI (Micro-F1@K).

Calculates exact retrieval metrics:
- Precision@K
- Recall@K
- Micro-F1@K
- True Positives, False Positives, False Negatives
from prediction JSON and ground-truth qrels.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.phase_07.model import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser(description="Phase 07 Evaluation")
    parser.add_argument("--prediction_file", type=str, default="submissions/phase_07/best_official/prediction.json", help="Path to predictions JSON")
    parser.add_argument("--qrels_file", type=str, default=None, help="Path to ground truth qrels JSON ({qid: [relevant_cids...]})")
    parser.add_argument("--k", type=int, default=7, help="Cut-off rank K")
    args = parser.parse_args()

    print("=== Phase 07 Prior Case Retrieval Evaluation ===")
    print(f"Cut-off rank: K={args.k}")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    pred_path = args.prediction_file
    if not os.path.exists(pred_path):
        cand = os.path.join(repo_root, args.prediction_file)
        if os.path.exists(cand):
            pred_path = cand
        else:
            print(f"[ERROR] Prediction file not found at: {args.prediction_file}")
            sys.exit(1)

    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} query predictions from {pred_path}")

    qrels_path = args.qrels_file
    if qrels_path and not os.path.exists(qrels_path):
        cand_q = os.path.join(repo_root, qrels_path)
        if os.path.exists(cand_q):
            qrels_path = cand_q

    if qrels_path and os.path.exists(qrels_path):
        with open(qrels_path, "r", encoding="utf-8") as f:
            qrels = json.load(f)

        metrics = evaluate_retrieval(qrels, preds, k=args.k)
        print("\nEvaluation Results against Ground Truth Qrels:")
        print(f"  Total Queries:     {metrics['total_queries']}")
        print(f"  Gold Pairs:        {metrics['gold_pairs']}")
        print(f"  True Positives:    {metrics['tp']}")
        print(f"  False Positives:   {metrics['fp']}")
        print(f"  False Negatives:   {metrics['fn']}")
        print(f"  Precision@{args.k}:     {metrics['precision@k']:.4f}")
        print(f"  Recall@{args.k}:        {metrics['recall@k']:.4f}")
        print(f"  Micro-F1@{args.k}:      {metrics['micro_f1@k']:.4f}")
        return

    print("\n[NOTE] Blind competition test set: Ground-truth qrels are held privately by the Codabench server.")
    print("Local evaluation requires ground-truth qrels (--qrels_file <path>).")
    print("To inspect official competition submission scores, see: results/official_submissions.csv")


if __name__ == "__main__":
    main()
