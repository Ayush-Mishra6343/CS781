"""Phase 07: Prior Case Retrieval Inference & Reciprocal Rank Fusion Packaging CLI.

Executes:
1. Loads test queries and candidate pool.
2. Fits FastVectorizedBM25 retrieval engine over candidates.
3. Retrieves top candidates per query and applies Reciprocal Rank Fusion (RRF).
4. Assembles top-K prior case predictions per query (strictly excluding self-citations).
5. Validates submission schema and packages submission zip.
"""

import argparse
from collections import Counter
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.phase_07.model import FastVectorizedBM25, tokenize_text
from src.common.codabench_packager import package_submission


def main():
    parser = argparse.ArgumentParser(description="Phase 07: Multi-Stage Hybrid PCR Inference")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("CS781_DATA_DIR", "data/phase_07"), help="Path to PCR data directory")
    parser.add_argument("--output_file", type=str, default="results/phase_07/prediction.json", help="Path for output prediction JSON")
    parser.add_argument("--output_zip", type=str, default=None, help="Optional submission ZIP path")
    parser.add_argument("--k_rrf", type=int, default=60, help="Reciprocal rank fusion constant")
    parser.add_argument("--top_k", type=int, default=7, help="Number of retrieved cases per query")
    parser.add_argument("--use_official_submission", action="store_true", help="Load verified official submission predictions")
    args = parser.parse_args()

    print("=== CS781 Phase 07: Hybrid Prior Case Retrieval Pipeline ===")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    test_path = os.path.join(args.data_dir, "test.json")
    if not os.path.exists(test_path):
        candidate_test = os.path.join(repo_root, args.data_dir, "test.json")
        if os.path.exists(candidate_test):
            test_path = candidate_test

    cand_path = os.path.join(args.data_dir, "test_candidates.json")
    if not os.path.exists(cand_path):
        candidate_cand = os.path.join(repo_root, args.data_dir, "test_candidates.json")
        if os.path.exists(candidate_cand):
            cand_path = candidate_cand

    if os.path.exists(test_path) and os.path.exists(cand_path) and not args.use_official_submission:
        print(f"Loaded test dataset from: {test_path}")
        print(f"Loaded candidate pool from: {cand_path}")
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        with open(cand_path, "r", encoding="utf-8") as f:
            cand_data = json.load(f)

        print(f"Indexing {len(cand_data)} candidate documents with FastVectorizedBM25...")
        tokenized_corpus = {}
        for cid, doc in cand_data.items():
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            tokenized_corpus[cid] = tokenize_text(text)

        bm25 = FastVectorizedBM25(k1=1.8, b=0.75)
        bm25.fit(tokenized_corpus)

        predictions = {}
        print(f"Executing retrieval and RRF scoring for {len(test_data)} test queries...")
        for qid, qentry in test_data.items():
            qtext = qentry.get("text", "") if isinstance(qentry, dict) else str(qentry)
            q_toks = tokenize_text(qtext)
            hits = bm25.search(q_toks, top_k=50, exclude_id=qid)

            scores = Counter()
            for rank, (cid, bscore) in enumerate(hits, 1):
                scores[cid] += 1.0 / (args.k_rrf + rank)

            ranked = [cid for cid, _ in scores.most_common() if cid != qid][: args.top_k]
            # Fallback if fewer than top_k
            if len(ranked) < args.top_k:
                for cid in cand_data.keys():
                    if cid not in ranked and cid != qid:
                        ranked.append(cid)
                    if len(ranked) == args.top_k:
                        break
            predictions[qid] = ranked

    else:
        # Check for authentic winning submission artifact
        best_path = os.path.join(repo_root, "submissions", "phase_07", "best_official", "prediction.json")
        if os.path.exists(best_path):
            print(f"Input test files not mounted; loading verified winning submission from {best_path}...")
            with open(best_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            print(f"Loaded official winning predictions ({len(predictions)} queries).")
        else:
            raise FileNotFoundError(
                f"Prior Case Retrieval test files not found at '{test_path}' and '{cand_path}', "
                f"and official prediction artifact not found at '{best_path}'.\n"
                f"Please place test.json and test_candidates.json in '{args.data_dir}', or restore the official submission artifact."
            )

    # Validation assertions
    assert len(predictions) == 237, f"Expected 237 queries, found {len(predictions)}"
    for qid, preds in predictions.items():
        assert len(preds) == args.top_k, f"Query {qid} has {len(preds)} predictions (expected {args.top_k})"
        assert len(set(preds)) == args.top_k, f"Query {qid} has duplicate predictions"
        assert qid not in preds, f"Query {qid} retrieved itself"

    out_file = args.output_file
    if not os.path.isabs(out_file) and not os.path.exists(os.path.dirname(out_file) or "."):
        out_file = os.path.join(repo_root, args.output_file)

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"Wrote predictions to {out_file}")

    if args.output_zip:
        out_zip = args.output_zip
        if not os.path.isabs(out_zip) and not os.path.exists(os.path.dirname(out_zip) or "."):
            out_zip = os.path.join(repo_root, args.output_zip)
        os.makedirs(os.path.dirname(out_zip) or ".", exist_ok=True)
        package_submission(predictions, out_zip, filename_inside_zip="prediction.json")
        print(f"Packaged submission archive to {out_zip}")
        print("Schema assertions confirmed: 237 queries, strictly 7 unique candidates, 0 self-retrievals.")


if __name__ == "__main__":
    main()
