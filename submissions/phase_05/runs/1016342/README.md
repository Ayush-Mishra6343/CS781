# Submission 1016342 (Phase 5: Legal Statute Identification (LSI))

- **Submission ID:** `1016342`
- **Phase:** 5
- **Submission Date:** 2026-10-28 20:53
- **Portal Score:** **0.3700**
- **Evaluation Metric:** Macro F1
- **Status:** Selected (Winning)

## 1. Technical Approach & Architecture
Multi-Label Hierarchical SwiGLU-LoRMA with Step-Zero Identity Verification & Hybrid LSI

## 2. Hyperparameters
```json
{
  "num_sections": 100,
  "chunk_size": 256,
  "hidden_dim": 768,
  "adapter": "LoRMA",
  "step_zero_identity": true,
  "loss": "AsymmetricMultilabel",
  "threshold_tuning": true
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3700`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Official winning model for Phase 5. Successfully classified 100 Indian statutory penal codes from lengthy court records. Step-zero identity initialization prevented initial gradient explosion, reaching 0.3700 Macro F1.

## 5. Failure Analysis / Critical Findings
Standard multi-label sigmoid classifiers without section threshold calibration suffered from severe recall drops on long-tail penal sections.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_05_statute_identification/CS781_Phase5_LSI_Hierarchical_LoRMA.ipynb`](notebooks/phase_05_statute_identification/CS781_Phase5_LSI_Hierarchical_LoRMA.ipynb)
