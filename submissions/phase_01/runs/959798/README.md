# Submission 959798 (Phase 1: Legal Language Model Pre-Training)

- **Submission ID:** `959798`
- **Phase:** 1
- **Submission Date:** 2026-10-04 13:24
- **Portal Score:** **0.8308**
- **Evaluation Metric:** Token Prediction Accuracy
- **Status:** Selected (Winning)

## 1. Technical Approach & Architecture
HybridLegalLM (94.76M params, GQA 12/4, Sliding Window Local 256 + Global [0,4,8,11])

## 2. Hyperparameters
```json
{
  "layers": 12,
  "hidden_dim": 768,
  "q_heads": 12,
  "kv_heads": 4,
  "vocab_size": 16000,
  "lr": 0.0003,
  "weight_decay": 0.1,
  "batch_size": 32,
  "max_seq_len": 1024,
  "local_window": 256,
  "global_layers": [
    0,
    4,
    8,
    11
  ]
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8308`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Official winning model for Phase 1. Achieved highest token continuation accuracy (0.8308) while maintaining lowest perplexity (21.589) via hybrid local/global sliding window attention.

## 5. Failure Analysis / Critical Findings
Pure global attention runs suffered quadratic memory scaling and lower throughput on 1024-token contexts.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_01_pretraining/CS781_Phase1_HybridLegalLM_Pretraining.ipynb`](notebooks/phase_01_pretraining/CS781_Phase1_HybridLegalLM_Pretraining.ipynb)
