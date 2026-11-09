# Submission 992293 (Phase 3: Rhetorical Role (RR) Classification & PEFT Suite)

- **Submission ID:** `992293`
- **Phase:** 3
- **Submission Date:** 2026-10-15 22:30
- **Portal Score:** **0.3336**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
LoRA with Asymmetric Loss for extreme class imbalance

## 2. Hyperparameters
```json
{
  "backbone": "legal-roberta-base",
  "lora_r": 8,
  "gamma_neg": 4.0,
  "gamma_pos": 1.0,
  "lr": 0.0002
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3336`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Asymmetric loss aggressively penalized negative predictions, resulting in high recall but depressed precision on dominant classes.

## 5. Failure Analysis / Critical Findings
Gamma hyperparameter tuning overly penalized negative examples, distorting probability calibration.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb`](notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb)
