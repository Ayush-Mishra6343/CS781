# Submission 1007647 (Phase 4: Court Judgment Prediction and Explanation (CJPE))

- **Submission ID:** `1007647`
- **Phase:** 4
- **Submission Date:** 2026-10-25 16:50
- **Portal Score:** **0.5735**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Soft Prompt Tuning (20 virtual tokens) + LoRMA Adapter with Asymmetric Loss

## 2. Hyperparameters
```json
{
  "virtual_tokens": 20,
  "adapter": "LoRMA",
  "rank": 8,
  "loss": "Asymmetric",
  "lr": 0.0003
}
```

## 3. Results & Evaluation
- **Official Score:** `0.5735`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Prompt tuning guided model attention toward jurisdictional header cues, achieving a strong 0.5735.

## 5. Failure Analysis / Critical Findings
Virtual token embeddings were sensitive to initialization and prompt length.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_04_judgment_prediction/historical_experiments/CS781_Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb`](notebooks/phase_04_judgment_prediction/historical_experiments/CS781_Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb)
