import os
import sys
import json
import time
import math
import csv
from collections import Counter
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

BASE_DIR = r"D:\Phase7_PCR_050_PLUS"
OLD_BASE = r"D:\Phase7_PCR"
PROJ_ROOT = r"C:\Users\LENOVO\.gemini\antigravity\scratch\CS781_Capstone"
DATA_DIR = os.path.join(PROJ_ROOT, "Phase_7_PCR", "data")

sys.path.append(os.path.join(BASE_DIR, "src"))
from e0_audit_and_reproduce import evaluate_retrieval
from fast_vectorized_bm25 import FastVectorizedBM25
from lexical_fielded_engine import tokenize_text

def run_lambdamart_pipeline():
    print("=" * 70)
    print("PHASE D+: HIGH-IMPACT LAMBDAMART (NDCG@7) STRUCTURAL RERANKER v2")
    print("=" * 70)

    # 1. Load data
    t0 = time.time()
    with open(os.path.join(DATA_DIR, "Train", "train.json"), "r", encoding="utf-8") as f:
        train_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Train", "train_candidates.json"), "r", encoding="utf-8") as f:
        train_c = json.load(f)
    with open(os.path.join(DATA_DIR, "Dev", "dev.json"), "r", encoding="utf-8") as f:
        dev_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Dev", "dev_candidates.json"), "r", encoding="utf-8") as f:
        dev_c = json.load(f)

    # Load 4-chunk cache
    cache_path = os.path.join(BASE_DIR, "cache", "tokenized_4chunks.pt")
    import torch
    chunk_cache = torch.load(cache_path, weights_only=False)

    train_q_chunks = chunk_cache["Train_queries"]
    train_c_chunks = chunk_cache["Train_candidates"]
    dev_q_chunks = chunk_cache["Dev_queries"]
    dev_c_chunks = chunk_cache["Dev_candidates"]

    # 2. Fit Fast Vectorized BM25 on Train and Dev
    print("Fitting FastVectorizedBM25 on Train candidates...")
    train_c_tokens = {cid: tokenize_text(" ".join(c["text"]), max_tokens=2500) for cid, c in train_c.items()}
    bm25_train = FastVectorizedBM25(k1=1.5, b=0.75)
    bm25_train.fit(train_c_tokens)

    print("Fitting FastVectorizedBM25 on Dev candidates...")
    dev_c_tokens = {cid: tokenize_text(" ".join(c["text"]), max_tokens=2500) for cid, c in dev_c.items()}
    bm25_dev = FastVectorizedBM25(k1=1.5, b=0.75)
    bm25_dev.fit(dev_c_tokens)

    # Search top 60 candidates for Train and Dev
    print("Searching top 60 BM25 for Train queries...")
    t_s = time.time()
    train_bm25_preds = {}
    for qid, q in train_q.items():
        res = bm25_train.search(tokenize_text(" ".join(q["text"]), max_tokens=2500), top_k=60, exclude_id=qid)
        train_bm25_preds[qid] = res
    print(f"Train search complete in {time.time()-t_s:.2f}s")

    print("Searching top 60 BM25 for Dev queries...")
    t_s = time.time()
    dev_bm25_preds = {}
    for qid, q in dev_q.items():
        res = bm25_dev.search(tokenize_text(" ".join(q["text"]), max_tokens=2500), top_k=60, exclude_id=qid)
        dev_bm25_preds[qid] = res
    print(f"Dev search complete in {time.time()-t_s:.2f}s")

    # 3. Rich Structural Feature Extraction
    def extract_rich_features(q_info, c_info, bm25_score, rank, q_toks, c_toks):
        # 1. Lexical BM25 & Rank
        f_bm25 = float(bm25_score)
        f_bm25_log = math.log1p(max(bm25_score, 0.0))
        f_rr = 1.0 / (10.0 + rank)
        f_rank = float(rank)

        # 2. Exact Citation Match (High Precision)
        q_cits = set(q_info.get("citations", []))
        c_cits = set(c_info.get("citations", []))
        cit_inter = len(q_cits & c_cits)
        has_cit = 1.0 if cit_inter > 0 else 0.0

        # 3. Exact Case Name Match (High Precision)
        q_cnames = set(q_info.get("case_names", []))
        c_cnames = set(c_info.get("case_names", []))
        cname_inter = len(q_cnames & c_cnames)
        has_cname = 1.0 if cname_inter > 0 else 0.0

        # 4. Statute Overlap (Auxiliary)
        q_stats = set(q_info.get("statutes", []))
        c_stats = set(c_info.get("statutes", []))
        stat_inter = len(q_stats & c_stats)
        stat_jacc = float(stat_inter) / (len(q_stats | c_stats) + 1e-6)

        # 5. Legal Event Overlap
        q_events = set(q_info.get("events", []))
        c_events = set(c_info.get("events", []))
        event_inter = len(q_events & c_events)
        event_jacc = float(event_inter) / (len(q_events | c_events) + 1e-6)

        # 6. Lexical Jaccard Overlap
        q_set = set(q_toks)
        c_set = set(c_toks)
        token_inter = len(q_set & c_set)
        token_jacc = float(token_inter) / (len(q_set | c_set) + 1e-6)

        # 7. Multi-Field / Rhetorical Zone Alignment using tokenized 4-chunk arrays
        # Chunk 0: Facts, Chunk 1: Issues, Chunk 2: Reasoning, Chunk 3: Disposition
        q_ids = q_info.get("input_ids")
        c_ids = c_info.get("input_ids")

        if q_ids is not None and c_ids is not None:
            # Non-pad tokens (exclude 0)
            q_facts = set(t for t in q_ids[0].tolist() if t != 0)
            q_issues = set(t for t in q_ids[1].tolist() if t != 0)
            q_reas = set(t for t in q_ids[2].tolist() if t != 0)
            q_disp = set(t for t in q_ids[3].tolist() if t != 0)

            c_facts = set(t for t in c_ids[0].tolist() if t != 0)
            c_issues = set(t for t in c_ids[1].tolist() if t != 0)
            c_reas = set(t for t in c_ids[2].tolist() if t != 0)
            c_disp = set(t for t in c_ids[3].tolist() if t != 0)

            facts_to_reas = len(q_facts & c_reas) / (len(q_facts | c_reas) + 1e-6)
            issues_to_reas = len(q_issues & c_reas) / (len(q_issues | c_reas) + 1e-6)
            reas_to_reas = len(q_reas & c_reas) / (len(q_reas | c_reas) + 1e-6)
            disp_to_disp = len(q_disp & c_disp) / (len(q_disp | c_disp) + 1e-6)
        else:
            facts_to_reas = 0.0
            issues_to_reas = 0.0
            reas_to_reas = 0.0
            disp_to_disp = 0.0

        # Cross interactions
        cit_x_bm25 = cit_inter * f_bm25_log
        cname_x_bm25 = cname_inter * f_bm25_log

        return [
            f_bm25, f_bm25_log, f_rr, f_rank,
            float(cit_inter), has_cit,
            float(cname_inter), has_cname,
            float(stat_inter), stat_jacc,
            float(event_inter), event_jacc,
            token_jacc,
            facts_to_reas, issues_to_reas, reas_to_reas, disp_to_disp,
            cit_x_bm25, cname_x_bm25
        ]

    # Pre-tokenize queries
    print("Tokenizing queries for feature extraction...")
    train_q_toks = {qid: tokenize_text(" ".join(q["text"]), max_tokens=1000) for qid, q in train_q.items()}
    dev_q_toks = {qid: tokenize_text(" ".join(q["text"]), max_tokens=1000) for qid, q in dev_q.items()}

    # 4. Build Train dataset grouped by query
    print("Building Train dataset for LambdaMART...")
    X_train = []
    y_train = []
    groups_train = []

    for qid, q in train_q.items():
        gold_set = set(q.get("relevant_candidates", []))
        candidates = train_bm25_preds.get(qid, [])[:50]
        if not candidates or len(gold_set) == 0:
            continue

        q_info = train_q_chunks.get(qid, {})
        q_tok = train_q_toks[qid]
        g_size = 0

        for rank, (cid, score) in enumerate(candidates, 1):
            c_info = train_c_chunks.get(cid, {})
            c_tok = train_c_tokens.get(cid, [])
            label = 1 if cid in gold_set else 0
            feat = extract_rich_features(q_info, c_info, score, rank, q_tok, c_tok)
            X_train.append(feat)
            y_train.append(label)
            g_size += 1

        if g_size > 0:
            groups_train.append(g_size)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"X_train: {X_train.shape}, Groups: {len(groups_train)}, Positives: {int(np.sum(y_train))}")

    # 5. Train LightGBM LambdaMART Ranker with tuned parameters
    print("\nTraining LGBMRanker with tuned parameters...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[7],
        n_estimators=120,
        learning_rate=0.03,
        num_leaves=25,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )
    ranker.fit(X_train, y_train, group=groups_train)

    # 6. Feature Importances
    feat_names = [
        "BM25_Raw", "BM25_Log", "Reciprocal_Rank", "BM25_Rank",
        "Citation_Count", "Has_Citation",
        "CaseName_Count", "Has_CaseName",
        "Statute_Count", "Statute_Jaccard",
        "Event_Count", "Event_Jaccard",
        "Token_Jaccard",
        "Facts_to_Reasoning", "Issues_to_Reasoning", "Reas_to_Reasoning", "Disp_to_Disp",
        "Cit_x_BM25", "CName_x_BM25"
    ]
    imps = ranker.feature_importances_
    for idx in np.argsort(-imps):
        print(f"  {feat_names[idx]:25s}: {imps[idx]:4d}")

    # 7. Evaluate on Dev Set (118 queries)
    print("\nEvaluating LambdaMART on Full Dev Set (118 queries)...")
    dev_preds = {}
    for qid in dev_q.keys():
        q_info = dev_q_chunks.get(qid, {})
        q_tok = dev_q_toks[qid]
        candidates = dev_bm25_preds.get(qid, [])[:60]

        X_dev = []
        cids = []
        for rank, (cid, score) in enumerate(candidates, 1):
            c_info = dev_c_chunks.get(cid, {})
            c_tok = dev_c_tokens.get(cid, [])
            feat = extract_rich_features(q_info, c_info, score, rank, q_tok, c_tok)
            X_dev.append(feat)
            cids.append(cid)

        if X_dev:
            scores = ranker.predict(np.array(X_dev, dtype=np.float32))
            cand_scores = list(zip(cids, scores))
            cand_scores.sort(key=lambda x: x[1], reverse=True)
            dev_preds[qid] = [c for c, _ in cand_scores]
        else:
            dev_preds[qid] = []

    m = evaluate_retrieval(dev_preds, dev_q, k=7)
    print("\n" + "=" * 70)
    print("RESULTS: LAMBDAMART NDCG@7 RERANKER v2 ON DEV")
    print("=" * 70)
    print(f"  Micro-F1@7 = {m['Micro_F1@7']:.4f}")
    print(f"  Micro-P@7  = {m['Micro_P@7']:.4f}")
    print(f"  Micro-R@7  = {m['Micro_R@7']:.4f}")
    print(f"  Total TP   = {m['Total_TP']} / 826 slots")
    print(f"  Recall@50  = {m['Recall@50']:.4f}")
    print(f"  MRR        = {m['MRR']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    run_lambdamart_pipeline()
