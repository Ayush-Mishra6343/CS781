# Submission 1013809 (Phase 5: Legal Statute Identification (LSI))

- **Submission ID:** `1013809`
- **Phase:** 5
- **Submission Date:** 2026-10-28 09:40
- **Portal Score:** **0.2293**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
DeBERTa-v3 multi-label baseline with max-sequence truncation

## 2. Hyperparameters
```json
{
  "backbone": "deberta-v3-base",
  "max_len": 512,
  "lr": 2e-05,
  "batch_size": 16
}
```

## 3. Results & Evaluation
- **Official Score:** `0.2293`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
DeBERTa relative position embeddings helped within the first 512 tokens, but whole-document context was missing.

## 5. Failure Analysis / Critical Findings
Lack of hierarchical aggregation limited legal statute retrieval across long judgments.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_05_statute_identification/historical_experiments/V4_KAGGLE_HYBRID_LSI.ipynb`](notebooks/phase_05_statute_identification/historical_experiments/V4_KAGGLE_HYBRID_LSI.ipynb)
