# Submission 1006364 (Phase 4: Court Judgment Prediction and Explanation (CJPE))

- **Submission ID:** `1006364`
- **Phase:** 4
- **Submission Date:** 2026-10-24 18:22
- **Portal Score:** **0.5520**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Chunked Legal-RoBERTa baseline with mean-pooling over 8 chunks

## 2. Hyperparameters
```json
{
  "backbone": "legal-roberta-base",
  "chunks": 8,
  "chunk_len": 512,
  "aggregation": "mean_pool",
  "lr": 3e-05
}
```

## 3. Results & Evaluation
- **Official Score:** `0.5520`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Initial hierarchical prototype. Simple mean-pooling diluted crucial ratio decidendi statements across uninformative factual paragraphs.

## 5. Failure Analysis / Critical Findings
Mean-pooling across chunks failed to weight decisive legal holding sentences over narrative background.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_04_judgment_prediction/historical_experiments/CS781_Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb`](notebooks/phase_04_judgment_prediction/historical_experiments/CS781_Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb)
