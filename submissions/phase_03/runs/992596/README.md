# Submission 992596 (Phase 3: Rhetorical Role (RR) Classification & PEFT Suite)

- **Submission ID:** `992596`
- **Phase:** 3
- **Submission Date:** 2026-10-16 14:10
- **Portal Score:** **0.4058**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Weighted ensemble of LoRA and LoRMA without BitFit

## 2. Hyperparameters
```json
{
  "backbone": "legal-roberta-base",
  "lora_r": 8,
  "lorma_r": 8,
  "weights": [
    0.55,
    0.45
  ]
}
```

## 3. Results & Evaluation
- **Official Score:** `0.4058`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
High-performing two-model ensemble (0.4058). Addition of BitFit in 992721 added orthogonal bias shifts that lifted Macro-F1 to 0.4337.

## 5. Failure Analysis / Critical Findings
Lacked the fine-grained decision boundary adjustments provided by bias-only tuning.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_03_rhetorical_roles/CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb`](notebooks/phase_03_rhetorical_roles/CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb)
