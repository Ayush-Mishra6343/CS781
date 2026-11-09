# Submission 1039667 (Phase 7: Prior Case Retrieval (PCR))

- **Submission ID:** `1039667`
- **Phase:** 7
- **Submission Date:** 2026-11-07 13:10
- **Portal Score:** **0.1175**
- **Evaluation Metric:** Micro F1@7
- **Status:** Superseded

## 1. Technical Approach & Architecture
Okapi BM25 Pure Lexical Baseline

## 2. Hyperparameters
```json
{
  "algorithm": "Okapi BM25",
  "k1": 1.5,
  "b": 0.75,
  "top_k": 7
}
```

## 3. Results & Evaluation
- **Official Score:** `0.1175`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Standard information retrieval baseline. Establishes the floor for lexical matching in legal case law.

## 5. Failure Analysis / Critical Findings
Legal vocabulary mismatch problem: judges formulate identical legal doctrines using completely different words across jurisdictions.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_07_prior_case_retrieval/historical_experiments/CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb`](notebooks/phase_07_prior_case_retrieval/historical_experiments/CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb)
