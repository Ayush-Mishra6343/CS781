# Submission 1011971 (Phase 5: Legal Statute Identification (LSI))

- **Submission ID:** `1011971`
- **Phase:** 5
- **Submission Date:** 2026-10-27 14:10
- **Portal Score:** **0.2356**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
TF-IDF N-Gram Matching Baseline (Word 1-3 grams + Char 3-5 grams)

## 2. Hyperparameters
```json
{
  "max_features": 50000,
  "ngram_range": [
    1,
    3
  ],
  "sublinear_tf": true
}
```

## 3. Results & Evaluation
- **Official Score:** `0.2356`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Sublinear TF-IDF baseline. Handled exact statutory formula n-grams reasonably well, outperforming raw BM25.

## 5. Failure Analysis / Critical Findings
Could not generalize to latent semantic concepts or novel case formulations.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_05_statute_identification/historical_experiments/V4_KAGGLE_HYBRID_LSI.ipynb`](notebooks/phase_05_statute_identification/historical_experiments/V4_KAGGLE_HYBRID_LSI.ipynb)
