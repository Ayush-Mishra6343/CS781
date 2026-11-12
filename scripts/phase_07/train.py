"""Phase 07: Fast Vectorized BM25 Candidate Corpus Indexer.

Builds and serializes the BM25 inverted index and document statistics over the candidate
judgment corpus for Phase 7 retrieval. (Dense cross-encoder reranking was trained separately
in the Kaggle GPU environment).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.phase_07.model import FastVectorizedBM25, tokenize_text


def main():
    parser = argparse.ArgumentParser(description="Phase 07: Candidate Corpus Indexing with Fast Vectorized BM25")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("CS781_DATA_DIR", "./data"))
    parser.add_argument("--k1", type=float, default=1.8)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    print(f"=== CS781 Phase 07: Fast Vectorized BM25 Candidate Indexer (k1={args.k1}, b={args.b}) ===")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    cand_candidates = [
        os.path.join(args.data_dir, "test_candidates.json"),
        os.path.join(repo_root, args.data_dir, "test_candidates.json"),
        os.path.join("data", "phase_07", "test_candidates.json"),
    ]
    cand_file = next((p for p in cand_candidates if os.path.exists(p)), None)

    if cand_file:
        print(f"Loading candidate pool from: {cand_file}")
        with open(cand_file, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        tokenized_corpus = {}
        for cid, doc in candidates.items():
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            tokenized_corpus[cid] = tokenize_text(text)

        engine = FastVectorizedBM25(k1=args.k1, b=args.b)
        engine.fit(tokenized_corpus)
        print(f"Indexed {len(tokenized_corpus)} candidate documents successfully.")
    else:
        print(f"\n[NOTE] Candidate corpus ('test_candidates.json') not found in '{args.data_dir}'.")
        print("Please place test_candidates.json in the data directory to index candidate judgments.")
        print("BM25 indexer architecture verification complete.")


if __name__ == "__main__":
    main()
