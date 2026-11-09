# Submission 1008160 (Phase 4: Court Judgment Prediction and Explanation (CJPE))

- **Submission ID:** `1008160`
- **Phase:** 4
- **Submission Date:** 2026-10-26 12:30
- **Portal Score:** **0.5693**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
SwiGLU-LoRMA Standalone Model at Epoch 6

## 2. Hyperparameters
```json
{
  "chunk_size": 256,
  "hidden_dim": 768,
  "activation": "SwiGLU",
  "adapter": "LoRMA",
  "lr": 0.0002,
  "epochs": 6
}
```

## 3. Results & Evaluation
- **Official Score:** `0.5693`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Solid standalone baseline (0.5693). Adding the second stage LoRMA-Plus tuning and rank fusion boosted score to 0.6886.

## 5. Failure Analysis / Critical Findings
Single model without multi-view rank aggregation had higher variance on contentious appeal cases.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_04_judgment_prediction/historical_experiments/swiglu_local_global_experiment_phase_4.ipynb`](notebooks/phase_04_judgment_prediction/historical_experiments/swiglu_local_global_experiment_phase_4.ipynb)
