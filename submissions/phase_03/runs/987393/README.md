# Submission 987393 (Phase 3: Rhetorical Role (RR) Classification & PEFT Suite)

- **Submission ID:** `987393`
- **Phase:** 3
- **Submission Date:** 2026-10-14 09:15
- **Portal Score:** **0.4311**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Legal-RoBERTa fine-tuned with multi-label BCEWithLogitsLoss and threshold tuning

## 2. Hyperparameters
```json
{
  "backbone": "legal-roberta-base",
  "lr": 2e-05,
  "loss": "BCE",
  "optimal_threshold": 0.38
}
```

## 3. Results & Evaluation
- **Official Score:** `0.4311`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Early strong result (0.4311). Showed that per-class decision thresholds significantly improve Macro-F1.

## 5. Failure Analysis / Critical Findings
Tuning per-class thresholds on small validation split led to mild over-tuning.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb`](notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb)
