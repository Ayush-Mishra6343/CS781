# Submission 992567 (Phase 3: Rhetorical Role (RR) Classification & PEFT Suite)

- **Submission ID:** `992567`
- **Phase:** 3
- **Submission Date:** 2026-10-16 12:45
- **Portal Score:** **0.4311**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Standalone LoRA (r=8, alpha=16) with BCEWithLogitsLoss

## 2. Hyperparameters
```json
{
  "backbone": "legal-roberta-base",
  "lora_r": 8,
  "lora_alpha": 16,
  "lr": 0.0002,
  "loss": "BCEWithLogitsLoss"
}
```

## 3. Results & Evaluation
- **Official Score:** `0.4311`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Best individual PEFT checkpoint (0.4311). Proved that parameter-efficient fine-tuning outperformed full fine-tuning by preventing catastrophic forgetting on small legal splits.

## 5. Failure Analysis / Critical Findings
Marginally under-predicted overlapping secondary rhetorical roles compared to the full ensemble.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_03_rhetorical_roles/CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb`](notebooks/phase_03_rhetorical_roles/CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb)
