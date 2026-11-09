# Submission 992721 (Phase 3: Rhetorical Role (RR) Classification & PEFT Suite)

- **Submission ID:** `992721`
- **Phase:** 3
- **Submission Date:** 2026-10-16 15:39
- **Portal Score:** **0.4337**
- **Evaluation Metric:** Macro F1
- **Status:** Selected (Winning)

## 1. Technical Approach & Architecture
LoRA (r=8) + Multiplicative LoRMA (r=8) + BitFit Multi-Method PEFT Ensemble

## 2. Hyperparameters
```json
{
  "backbone": "legal-roberta-base",
  "lora_r": 8,
  "lora_alpha": 16,
  "lorma_r": 8,
  "bitfit": true,
  "ensemble": "weighted_logits",
  "weights": [
    0.45,
    0.4,
    0.15
  ]
}
```

## 3. Results & Evaluation
- **Official Score:** `0.4337`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Official winning model for Phase 3. Combined diverse PEFT mechanisms (additive rank update via LoRA, multiplicative scaling via LoRMA, bias tuning via BitFit) into a complementary ensemble.

## 5. Failure Analysis / Critical Findings
Single PEFT methods suffered from individual inductive biases on minority rhetorical roles (e.g., 'Statute' and 'Ruling by Present Court').

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_03_rhetorical_roles/CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb`](notebooks/phase_03_rhetorical_roles/CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb)
