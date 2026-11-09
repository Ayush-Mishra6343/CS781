# Submission 976925 (Phase 2: Legal Named Entity Recognition (L-NER))

- **Submission ID:** `976925`
- **Phase:** 2
- **Submission Date:** 2026-10-11 02:15
- **Portal Score:** **0.2367**
- **Evaluation Metric:** Strict Micro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
InLegalBERT uncalibrated token classifier

## 2. Hyperparameters
```json
{
  "backbone": "law-ai/InLegalBERT",
  "lr": 2e-05,
  "batch_size": 16,
  "epochs": 5
}
```

## 3. Results & Evaluation
- **Official Score:** `0.2367`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Domain-adapted vocabulary had smaller BPE fertility on legal terms, but struggled with general English syntax in headnotes.

## 5. Failure Analysis / Critical Findings
Domain pretraining alone could not compensate for subtoken boundary alignment errors.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb`](notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb)
