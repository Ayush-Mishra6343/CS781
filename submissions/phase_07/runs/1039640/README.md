# Submission 1039640 (Phase 7: Prior Case Retrieval (PCR))

- **Submission ID:** `1039640`
- **Phase:** 7
- **Submission Date:** 2026-11-07 13:52
- **Portal Score:** **0.3739**
- **Evaluation Metric:** Micro F1@7
- **Status:** Selected (Winning)

## 1. Technical Approach & Architecture
Multi-Stage Lexical BM25 + Dense Cross-Encoder Neural Reranker with Reciprocal Rank Fusion & Score Normalization

## 2. Hyperparameters
```json
{
  "first_stage": "BM25 (k1=1.5, b=0.75, top_k=50)",
  "second_stage": "Cross-Encoder (Legal-RoBERTa)",
  "rrf_k": 60,
  "score_normalization": "min_max",
  "cut_off": 7
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3739`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Official winning model for Phase 7. Tripled retrieval performance over pure lexical baselines (0.1175 -> 0.3739). BM25 provided high-recall candidate generation (Top-50), and Cross-Encoder fine-tuned with reciprocal rank fusion precisely re-ordered top candidates to maximize Micro F1@7.

## 5. Failure Analysis / Critical Findings
Pure dense retrieval suffered from out-of-vocabulary statute section citations; pure BM25 missed conceptual legal analogies. The hybrid RRF combination resolved both failure modes.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_07_prior_case_retrieval/CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb`](notebooks/phase_07_prior_case_retrieval/CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb)
