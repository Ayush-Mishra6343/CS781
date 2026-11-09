# Submission 1014073 (Phase 5: Legal Statute Identification (LSI))

- **Submission ID:** `1014073`
- **Phase:** 5
- **Submission Date:** 2026-10-28 14:15
- **Portal Score:** **0.3211**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Legal-BERT multi-label classifier with dynamic positive loss weighting

## 2. Hyperparameters
```json
{
  "backbone": "legal-bert",
  "pos_weight": "dynamic",
  "lr": 3e-05,
  "epochs": 5
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3211`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Strong legal BERT baseline (0.3211), but limited to 512 tokens which truncated references to earlier procedural sections.

## 5. Failure Analysis / Critical Findings
512 token truncation cut off citations located in concluding paragraphs of judicial orders.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_05_statute_identification/historical_experiments/V4_KAGGLE_HYBRID_LSI.ipynb`](notebooks/phase_05_statute_identification/historical_experiments/V4_KAGGLE_HYBRID_LSI.ipynb)
