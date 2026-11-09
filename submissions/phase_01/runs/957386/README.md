# Submission 957386 (Phase 1: Legal Language Model Pre-Training)

- **Submission ID:** `957386`
- **Phase:** 1
- **Submission Date:** 2026-10-03 16:47
- **Portal Score:** **0.8255**
- **Evaluation Metric:** Token Prediction Accuracy
- **Status:** Superseded

## 1. Technical Approach & Architecture
Standard Decoder LM with full Multi-Head Attention (12 heads, full dense context)

## 2. Hyperparameters
```json
{
  "layers": 12,
  "hidden_dim": 768,
  "heads": 12,
  "vocab_size": 16000,
  "lr": 0.0003,
  "weight_decay": 0.05,
  "batch_size": 16,
  "max_seq_len": 1024
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8255`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Strong dense attention baseline (0.8255), but slower inference and slightly worse long-range continuation than the GQA hybrid architecture.

## 5. Failure Analysis / Critical Findings
Quadratic attention memory limited effective batch size to 16, resulting in slower training convergence.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_01_pretraining/historical_experiments/CS781_Phase1_Pretraining_V1_Decoder.ipynb`](notebooks/phase_01_pretraining/historical_experiments/CS781_Phase1_Pretraining_V1_Decoder.ipynb)
