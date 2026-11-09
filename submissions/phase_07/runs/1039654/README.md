# Submission 1039654 (Phase 7: Prior Case Retrieval (PCR))

- **Submission ID:** `1039654`
- **Phase:** 7
- **Submission Date:** 2026-11-07 11:30
- **Portal Score:** **0.1955**
- **Evaluation Metric:** Micro F1@7
- **Status:** Superseded

## 1. Technical Approach & Architecture
Bi-Encoder Dense Retriever (Legal-RoBERTa embeddings, Cosine Similarity)

## 2. Hyperparameters
```json
{
  "backbone": "legal-roberta-base",
  "pooling": "mean",
  "similarity": "cosine",
  "top_k": 7
}
```

## 3. Results & Evaluation
- **Official Score:** `0.1955`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Captured semantic topic similarity between case facts (0.1955), substantially outperforming BM25 (0.1175).

## 5. Failure Analysis / Critical Findings
Bi-encoders evaluate query and document representations independently without cross-attention token interaction, missing subtle statutory distinctions.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_07_prior_case_retrieval/historical_experiments/CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb`](notebooks/phase_07_prior_case_retrieval/historical_experiments/CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb)
