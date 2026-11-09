# Submission 976720 (Phase 2: Legal Named Entity Recognition (L-NER))

- **Submission ID:** `976720`
- **Phase:** 2
- **Submission Date:** 2026-10-10 23:50
- **Portal Score:** **0.3092**
- **Evaluation Metric:** Strict Micro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Legal-RoBERTa 5-fold cross-validation ensemble with soft probability voting

## 2. Hyperparameters
```json
{
  "backbone": "roberta-base",
  "folds": 5,
  "lr": 3e-05,
  "ensemble_method": "soft_voting"
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3092`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Robust ensemble that reduced prediction variance, but raw boundary decoding was suboptimal compared to calibrated 974876.

## 5. Failure Analysis / Critical Findings
Soft voting across 5 folds smeared entity boundaries at beginning and end tokens.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb`](notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb)
