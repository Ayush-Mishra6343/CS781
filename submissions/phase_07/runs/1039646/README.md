# Submission 1039646 (Phase 7: Prior Case Retrieval (PCR))

- **Submission ID:** `1039646`
- **Phase:** 7
- **Submission Date:** 2026-11-07 09:15
- **Portal Score:** **0.1742**
- **Evaluation Metric:** Micro F1@7
- **Status:** Superseded

## 1. Technical Approach & Architecture
TF-IDF N-Gram Retrieval Baseline with Cosine Similarity

## 2. Hyperparameters
```json
{
  "ngram_range": [
    1,
    2
  ],
  "max_features": 40000,
  "norm": "l2"
}
```

## 3. Results & Evaluation
- **Official Score:** `0.1742`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Stronger than vanilla BM25 due to legal n-gram phrase matching (0.1742 vs 0.1175).

## 5. Failure Analysis / Critical Findings
High dimensional sparse vectors lacked semantic generalization for synonymous legal concepts.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_07_prior_case_retrieval/historical_experiments/CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb`](notebooks/phase_07_prior_case_retrieval/historical_experiments/CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb)
