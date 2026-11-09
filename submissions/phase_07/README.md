# Phase 7: Prior Case Retrieval (PCR): Submissions Archive

- **Official Winning Score:** **0.3739** (Submission ID: `1039640`)
- **Baseline Score:** **0.1175** (Submission ID: `1039667`)
- **Improvement Delta:** **+0.2564 (+218.21%)**
- **Evaluation Metric:** Micro F1@7 across 237 Official Test Queries (Evaluating Top-7 Retrieved Documents)

## 1. Task Description

Retrieval of relevant prior precedents from a massive Indian Supreme Court and High Court corpus given a new target case query, evaluated under strict top-7 cutoff constraints.

## 2. Chronological Submissions Table

| Submission ID | Date | Score | Status | Model / Technique | Key Insight / Learning |
|:---:|:---:|:---:|:---:|:---|:---|
| [`1039646`](runs/1039646/) | 2026-11-07 09:15 | **0.1742** | Superseded | TF-IDF N-Gram Retrieval Baseline with Cosine Similarity | Stronger than vanilla BM25 due to legal n-gram phrase matching (0.1742 vs 0.1175). |
| [`1039654`](runs/1039654/) | 2026-11-07 11:30 | **0.1955** | Superseded | Bi-Encoder Dense Retriever (Legal-RoBERTa embeddings, Cosine Similarity) | Captured semantic topic similarity between case facts (0.1955), substantially outperforming BM25 (0.1175). |
| [`1039658`](runs/1039658/) | 2026-11-07 12:45 | **0.1305** | Superseded | Bi-Encoder Dense Retriever trained with BM25 Hard Negatives | Trained to discriminate confusing legal documents retrieved by BM25. Over-penalized relevant documents sharing incidental terminology. |
| [`1039667`](runs/1039667/) | 2026-11-07 13:10 | **0.1175** | Superseded | Okapi BM25 Pure Lexical Baseline | Standard information retrieval baseline. Establishes the floor for lexical matching in legal case law. |
| [`1039640`](runs/1039640/) | 2026-11-07 13:52 | **0.3739** | Selected (Winning) 🏆 (Winning) | Multi-Stage Lexical BM25 + Dense Cross-Encoder Neural Reranker with Reciprocal Rank Fusion & Score Normalization | Official winning model for Phase 7. Tripled retrieval performance over pure lexical baselines (0.1175 -> 0.3739). BM25 provided high-recall candidate generation (Top-50), and Cross-Encoder fine-tuned with reciprocal rank fusion precisely re-ordered top candidates to maximize Micro F1@7. |

## 3. Architectural Evolution

```
Phase 7 Architectural Evolution:
1. Okapi BM25 Lexical Baseline (1039667: 0.1175)
   - Severe vocabulary mismatch between precedent descriptions and query cases
         │
         ▼
2. BM25 with Hard Negatives Bi-Encoder (1039658: 0.1305)
   - Over-penalized subtle relevant documents sharing incidental terminology
         │
         ▼
3. TF-IDF N-Gram Baseline (1039646: 0.1742)
   - Captured fixed statutory n-gram legal formulas better than BM25
         │
         ▼
4. Bi-Encoder Semantic Dense Retriever (1039654: 0.1955)
   - Mean-pooled Legal-RoBERTa embeddings captured semantic legal facts
         │
         ▼
5. Multi-Stage Lexical BM25 + Dense Cross-Encoder Neural Reranker (1039640: 0.3739) [WINNING]
   - High-recall lexical BM25 candidate generation (Top-50 candidates)
   - Cross-Encoder deep token-level query-document interaction
   - Reciprocal Rank Fusion (k=60) + Min-Max Score Normalization
   - Delivered massive 218% relative improvement to official winning 0.3739 score
```

## 4. Hyperparameter Tuning Narrative

Phase 7 required retrieving relevant precedent cases under a strict cutoff of exactly 7 predictions per query. Bi-encoders evaluate query and document representations independently, making them blind to intricate statutory cross-references. Our winning pipeline implemented a multi-stage hybrid architecture: Stage 1 leveraged fast vectorized BM25 (`fast_vectorized_bm25.py`) to retrieve the top 50 high-recall candidates; Stage 2 scored candidate pairs using a neural Cross-Encoder that performs all-to-all token attention between query and candidate text. Reciprocal Rank Fusion (k=60) combined lexical and dense scores into the final calibrated Top-7 ranking.

## 5. Failure Analysis & What Didn't Work

Pure BM25 (1039667: 0.1175) suffered from legal vocabulary divergence—judges describing the exact same legal principle (e.g. 'mens rea in abetment of suicide') using entirely distinct vocabulary. Pure dense bi-encoders (1039654: 0.1955) lacked cross-attention and failed to distinguish cases citing Section 302 IPC (Murder) from Section 304 IPC (Culpable Homicide). Only joint multi-stage cross-attention hybrid reranking successfully resolved both dilemmas.

