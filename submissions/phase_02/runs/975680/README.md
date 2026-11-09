# Submission 975680 (Phase 2: Legal Named Entity Recognition (L-NER))

- **Submission ID:** `975680`
- **Phase:** 2
- **Submission Date:** 2026-10-10 22:10
- **Portal Score:** **0.2931**
- **Evaluation Metric:** Strict Micro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
LoRMA parameter-efficient adapter on RoBERTa-Legal

## 2. Hyperparameters
```json
{
  "backbone": "roberta-base",
  "peft_type": "LoRMA",
  "rank": 8,
  "lr": 0.0001,
  "trainable_params_pct": 1.2
}
```

## 3. Results & Evaluation
- **Official Score:** `0.2931`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
First test of LoRMA in sequence labeling; trained in under 8 minutes, achieving 91% of full fine-tuning performance with 1.2% parameters.

## 5. Failure Analysis / Critical Findings
LoRMA rank 8 was slightly under-parameterized for token-level multi-class discrimination.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb`](notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb)
