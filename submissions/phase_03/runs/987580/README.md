# Submission 987580 (Phase 3: Rhetorical Role (RR) Classification & PEFT Suite)

- **Submission ID:** `987580`
- **Phase:** 3
- **Submission Date:** 2026-10-14 13:40
- **Portal Score:** **0.3941**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Standard Full Fine-Tuning Legal-RoBERTa baseline

## 2. Hyperparameters
```json
{
  "backbone": "legal-roberta-base",
  "lr": 2e-05,
  "batch_size": 16,
  "epochs": 4
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3941`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Full fine-tuning reference baseline. Suffered from gradient instability on minority class labels.

## 5. Failure Analysis / Critical Findings
Severe overfitting on high-frequency classes ('Facts') at the expense of rare classes ('Ratio').

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb`](notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb)
