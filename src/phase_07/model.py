"""Phase 07: Prior Case Retrieval (PCR) Model & Evaluation Suite.

Provides:
- FastVectorizedBM25: Highly vectorized sublinear TF-IDF / Okapi BM25 engine
- tokenize_text: Domain-aware legal text tokenizer with bigram support
- extract_all_citations: Multi-pattern Indian Legal Citation extractor
- evaluate_retrieval: Micro-averaged retrieval evaluator calculating Micro-P@K, Micro-R@K, Micro-F1@K, MRR
"""

import re
import math
from collections import Counter
import numpy as np

from .fast_vectorized_bm25 import FastVectorizedBM25

# Multi-pattern Indian Legal Citation Extractor
RE_PATTERNS = [
    re.compile(r"\b((?:19|20)\d{2})\s*(?:SC|All|Bom|Cal|Del|Mad|Guj|Pat|MP|Ker|P&H)\s*(\d{1,5})\b", re.IGNORECASE),
    re.compile(r"\b(?:AIR\s*)?((?:19|20)\d{2})\s*(?:AIR\s*)?(?:SC|All|Bom|Cal|Del|Mad)?\s*(\d{1,5})\b", re.IGNORECASE),
    re.compile(r"(?:\(\s*)?((?:19|20)\d{2})(?:\s*\))?\s*(?:\(\s*(\d{1,2})\s*\))?\s*(?:SCC|SCR|SCALE|JT)\s*(?:\(\s*(\d{1,2})\s*\))?\s*(\d{1,5})\b", re.IGNORECASE)
]

def extract_all_citations(text: str) -> set:
    """Extract Indian court citations from judgment text."""
    cits = set()
    for m in RE_PATTERNS[0].finditer(text):
        cits.add(f"{m.group(1)}_{m.group(2)}")
    for m in RE_PATTERNS[1].finditer(text):
        if int(m.group(2)) > 0:
            cits.add(f"{m.group(1)}_{m.group(2)}")
    for m in RE_PATTERNS[2].finditer(text):
        year = m.group(1)
        page = m.group(4)
        vol = m.group(2) or m.group(3) or ""
        if int(page) > 0:
            cits.add(f"{year}_{page}")
            if vol:
                cits.add(f"{year}_v{vol}_{page}")
    return cits

def tokenize_text(text: str, use_bigrams: bool = True, max_tokens: int = 1000) -> list:
    """Tokenize legal text with optional bigrams and truncation."""
    words = [w for w in text.lower().replace("\n", " ").split() if len(w) > 1]
    if max_tokens and len(words) > max_tokens:
        words = words[:max_tokens // 2] + words[-max_tokens // 2:]
    tokens = list(words)
    if use_bigrams and len(words) > 1:
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        tokens.extend(bigrams)
    return tokens

def evaluate_retrieval(preds_dict: dict, gold_dict: dict, k: int = 7) -> dict:
    """
    Standard micro-averaged retrieval evaluator on query set.
    Computes Micro-P@K, Micro-R@K, Micro-F1@K, MRR, and Recall at various cutoffs.
    """
    total_tp = 0
    total_preds = 0
    total_golds = 0
    num_queries = len(gold_dict)
    mrr_sum = 0.0
    rec_at_k = {10: 0, 20: 0, 50: 0, 100: 0, 200: 0}
    per_query_metrics = {}

    for qid, qdata in gold_dict.items():
        if isinstance(qdata, dict):
            golds = set(qdata.get("relevant_candidates", []))
        elif isinstance(qdata, (list, set)):
            golds = set(qdata)
        else:
            golds = set()

        total_golds += len(golds)
        raw_preds = preds_dict.get(qid, [])
        preds = [c for c in raw_preds if c != qid]  # enforce self-mask
        top_k = preds[:k]

        tp = len(set(top_k) & golds)
        total_tp += tp
        total_preds += len(top_k)

        # MRR
        rr = 0.0
        for rank, cid in enumerate(preds, 1):
            if cid in golds:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        # Recall at cutoffs
        for cut in rec_at_k:
            cut_hits = len(set(preds[:cut]) & golds)
            rec_at_k[cut] += cut_hits / max(len(golds), 1)

        pq_p = tp / max(len(top_k), 1)
        pq_r = tp / max(len(golds), 1)
        pq_f1 = (2 * pq_p * pq_r) / max(pq_p + pq_r, 1e-12)
        per_query_metrics[qid] = {
            "TP": tp,
            "Preds": len(top_k),
            "Golds": len(golds),
            "P@K": pq_p,
            "R@K": pq_r,
            "F1@K": pq_f1,
            "MRR": rr
        }

    micro_p = total_tp / max(total_preds, 1)
    micro_r = total_tp / max(total_golds, 1)
    micro_f1 = (2 * micro_p * micro_r) / max(micro_p + micro_r, 1e-12)
    mrr = mrr_sum / max(num_queries, 1)

    res = {
        f"Micro_P@{k}": round(micro_p, 4),
        f"Micro_R@{k}": round(micro_r, 4),
        f"Micro_F1@{k}": round(micro_f1, 4),
        "MRR": round(mrr, 4),
        "Total_TP": total_tp,
        "Total_Preds": total_preds,
        "Total_Golds": total_golds,
        "Num_Queries": num_queries,
        "Per_Query": per_query_metrics
    }
    for cut in rec_at_k:
        res[f"Recall@{cut}"] = round(rec_at_k[cut] / max(num_queries, 1), 4)

    return res

__all__ = [
    "FastVectorizedBM25",
    "tokenize_text",
    "extract_all_citations",
    "evaluate_retrieval"
]
