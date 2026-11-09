# Submission 975538 (Phase 2: Legal Named Entity Recognition (L-NER))

- **Submission ID:** `975538`
- **Phase:** 2
- **Submission Date:** 2026-10-10 21:05
- **Portal Score:** **0.3131**
- **Evaluation Metric:** Strict Micro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
RoBERTa-Legal with Linear-Chain Conditional Random Field (CRF) layer

## 2. Hyperparameters
```json
{
  "backbone": "roberta-base",
  "crf": true,
  "lr_backbone": 2e-05,
  "lr_crf": 0.001,
  "batch_size": 16
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3131`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
CRF enforced valid BIO transition constraints (eliminating I- after O-), producing cleaner entity structures.

## 5. Failure Analysis / Critical Findings
Viterbi decoding overhead slowed down batch inference during portal submission generation.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb`](notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb)
