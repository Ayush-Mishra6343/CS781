# Submission 1039658 (Phase 7: Prior Case Retrieval (PCR))

- **Submission ID:** `1039658`
- **Phase:** 7
- **Submission Date:** 2026-11-07 12:45
- **Portal Score:** **0.1305**
- **Evaluation Metric:** Micro F1@7
- **Status:** Superseded

## 1. Technical Approach & Architecture
Bi-Encoder Dense Retriever trained with BM25 Hard Negatives

## 2. Hyperparameters
```json
{
  "backbone": "legal-roberta-base",
  "loss": "MultipleNegativesRankingLoss",
  "hard_negatives": 5
}
```

## 3. Results & Evaluation
- **Official Score:** `0.1305`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Trained to discriminate confusing legal documents retrieved by BM25. Over-penalized relevant documents sharing incidental terminology.

## 5. Failure Analysis / Critical Findings
Hard negative mining selected false negatives (relevant cases with differing surface phrasing), depressing test score.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_07_prior_case_retrieval/historical_experiments/CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb`](notebooks/phase_07_prior_case_retrieval/historical_experiments/CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb)
