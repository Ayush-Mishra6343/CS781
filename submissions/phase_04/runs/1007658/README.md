# Submission 1007658 (Phase 4: Court Judgment Prediction and Explanation (CJPE))

- **Submission ID:** `1007658`
- **Phase:** 4
- **Submission Date:** 2026-10-25 19:15
- **Portal Score:** **0.5377**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Hierarchical Transformer with standard Cross-Entropy Loss

## 2. Hyperparameters
```json
{
  "chunk_size": 256,
  "hidden_dim": 768,
  "loss": "CrossEntropy",
  "lr": 0.0001
}
```

## 3. Results & Evaluation
- **Official Score:** `0.5377`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Standard cross-entropy baseline for hierarchical architecture.

## 5. Failure Analysis / Critical Findings
Struggled with class imbalance where majority rulings dominated loss gradients.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_04_judgment_prediction/historical_experiments/CS781_Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb`](notebooks/phase_04_judgment_prediction/historical_experiments/CS781_Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb)
