# Submission 1011978 (Phase 5: Legal Statute Identification (LSI))

- **Submission ID:** `1011978`
- **Phase:** 5
- **Submission Date:** 2026-10-27 16:30
- **Portal Score:** **0.2244**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
BM25 Lexical Statute Retrieval Baseline

## 2. Hyperparameters
```json
{
  "retriever": "Okapi BM25",
  "k1": 1.5,
  "b": 0.75,
  "statute_definitions_indexed": 100
}
```

## 3. Results & Evaluation
- **Official Score:** `0.2244`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Lexical baseline matching statutory text against case descriptions. Fast and transparent, but failed when judgments discussed factual circumstances without naming the formal section number.

## 5. Failure Analysis / Critical Findings
Complete lexical mismatch when statutes were cited colloquially (e.g., 'cheating' instead of 'Section 420 IPC').

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_05_statute_identification/historical_experiments/V4_KAGGLE_HYBRID_LSI.ipynb`](notebooks/phase_05_statute_identification/historical_experiments/V4_KAGGLE_HYBRID_LSI.ipynb)
