# Submission 974860 (Phase 2: Legal Named Entity Recognition (L-NER))

- **Submission ID:** `974860`
- **Phase:** 2
- **Submission Date:** 2026-10-10 18:45
- **Portal Score:** **0.3200**
- **Evaluation Metric:** Strict Micro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
RoBERTa-Legal with character boundary alignment at higher threshold (0.45)

## 2. Hyperparameters
```json
{
  "backbone": "roberta-base",
  "lr": 3e-05,
  "batch_size": 16,
  "epochs": 6,
  "threshold": 0.45,
  "boundary_calibrated": true
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3200`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Very close to peak (0.3200), slightly higher precision but lower recall on composite entities.

## 5. Failure Analysis / Critical Findings
Higher confidence threshold clipped borderline court designations.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_02_ner/CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb`](notebooks/phase_02_ner/CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb)
