# Submission 1012531 (Phase 5: Legal Statute Identification (LSI))

- **Submission ID:** `1012531`
- **Phase:** 5
- **Submission Date:** 2026-10-27 22:15
- **Portal Score:** **0.3097**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Standalone SwiGLU-LoRMA Single-Model Baseline (V3 LSI)

## 2. Hyperparameters
```json
{
  "chunk_size": 256,
  "adapter": "LoRMA",
  "rank": 8,
  "lr": 0.0002,
  "step_zero": false
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3097`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Early single-model milestone (0.3097). Highlighted the power of SwiGLU-LoRMA, but lacked step-zero stabilization and threshold tuning of winning 1016342.

## 5. Failure Analysis / Critical Findings
Without step-zero identity initialization, early training loss oscillated violently across rare sections.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_05_statute_identification/historical_experiments/CS781_Phase5_LSI_V1_SwiGLU_LoRMA.ipynb`](notebooks/phase_05_statute_identification/historical_experiments/CS781_Phase5_LSI_V1_SwiGLU_LoRMA.ipynb)
