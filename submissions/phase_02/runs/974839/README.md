# Submission 974839 (Phase 2: Legal Named Entity Recognition (L-NER))

- **Submission ID:** `974839`
- **Phase:** 2
- **Submission Date:** 2026-10-10 16:12
- **Portal Score:** **0.3142**
- **Evaluation Metric:** Strict Micro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
RoBERTa-Legal with standard token-level argmax boundary mapping

## 2. Hyperparameters
```json
{
  "backbone": "roberta-base",
  "lr": 3e-05,
  "batch_size": 16,
  "epochs": 6,
  "threshold": 0.5,
  "boundary_calibrated": false
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3142`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Demonstrated that model representations were strong (dev F1 0.5140), but uncalibrated span conversion hurt portal score.

## 5. Failure Analysis / Critical Findings
Argmax prediction without margin thresholding caused entity truncation.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb`](notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb)
