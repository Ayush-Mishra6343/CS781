# Submission 974378 (Phase 2: Legal Named Entity Recognition (L-NER))

- **Submission ID:** `974378`
- **Phase:** 2
- **Submission Date:** 2026-10-10 14:02
- **Portal Score:** **0.1874**
- **Evaluation Metric:** Strict Micro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Pilot RoBERTa token classifier without character span reconstruction

## 2. Hyperparameters
```json
{
  "backbone": "roberta-base",
  "lr": 3e-05,
  "batch_size": 16,
  "epochs": 2
}
```

## 3. Results & Evaluation
- **Official Score:** `0.1874`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
First end-to-end pipeline sanity test. Verified data loaders and evaluation format.

## 5. Failure Analysis / Critical Findings
Under-trained (2 epochs); character offset mapping was not yet implemented.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb`](notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb)
