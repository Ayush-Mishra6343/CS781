# Submission 1009439 (Phase 4: Court Judgment Prediction and Explanation (CJPE))

- **Submission ID:** `1009439`
- **Phase:** 4
- **Submission Date:** 2026-10-26 18:53
- **Portal Score:** **0.6886**
- **Evaluation Metric:** Macro F1
- **Status:** Selected (Winning)

## 1. Technical Approach & Architecture
Hierarchical Local-Global SwiGLU-LoRMA Document Classifier (Rank Super-Ensemble)

## 2. Hyperparameters
```json
{
  "chunk_size": 256,
  "num_chunks": 16,
  "hidden_dim": 768,
  "global_layers": 4,
  "adapter": "LoRMA-Plus",
  "activation": "SwiGLU",
  "lr": 0.00015,
  "ensemble_method": "rank_super_ensemble"
}
```

## 3. Results & Evaluation
- **Official Score:** `0.6886`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Official winning model for Phase 4. Solved long legal judgment classification (often >10,000 words) using hierarchical chunk-and-aggregate with SwiGLU gating and LoRMA parameter efficiency.

## 5. Failure Analysis / Critical Findings
Flat truncation at 512 tokens threw away court reasoning, resulting in ~14% lower Macro-F1.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_04_judgment_prediction/CS781_Phase4_CJPE_SwiGLU_LoRMA.ipynb`](notebooks/phase_04_judgment_prediction/CS781_Phase4_CJPE_SwiGLU_LoRMA.ipynb)
