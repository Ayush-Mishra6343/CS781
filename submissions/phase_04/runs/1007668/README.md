# Submission 1007668 (Phase 4: Court Judgment Prediction and Explanation (CJPE))

- **Submission ID:** `1007668`
- **Phase:** 4
- **Submission Date:** 2026-10-25 21:40
- **Portal Score:** **0.5464**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Hierarchical Transformer with Focal Loss (gamma=2.0)

## 2. Hyperparameters
```json
{
  "chunk_size": 256,
  "hidden_dim": 768,
  "loss": "FocalLoss",
  "gamma": 2.0,
  "lr": 0.0001
}
```

## 3. Results & Evaluation
- **Official Score:** `0.5464`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Focal loss helped with the extreme judgment class distribution (Affirmed vs Reversed), but tended to over-predict reversals.

## 5. Failure Analysis / Critical Findings
Hard negative downweighting caused false positive predictions on subtle dismissal rulings.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_04_judgment_prediction/historical_experiments/CS781_Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb`](notebooks/phase_04_judgment_prediction/historical_experiments/CS781_Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb)
