# Submission 954523 (Phase 1: Legal Language Model Pre-Training)

- **Submission ID:** `954523`
- **Phase:** 1
- **Submission Date:** 2026-10-02 21:10
- **Portal Score:** **0.8305**
- **Evaluation Metric:** Token Prediction Accuracy
- **Status:** Superseded

## 1. Technical Approach & Architecture
RoFormer-style Rotary Position Embedding (RoPE) Legal Decoder (12 layers, 768 hidden)

## 2. Hyperparameters
```json
{
  "layers": 12,
  "hidden_dim": 768,
  "heads": 12,
  "vocab_size": 16000,
  "lr": 0.0003,
  "rotary_dim": 64,
  "batch_size": 32,
  "max_seq_len": 1024
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8305`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Excellent score (0.8305), barely behind 959798. RoPE demonstrated outstanding length extrapolation, but GQA in 959798 reduced KV cache memory by 3x.

## 5. Failure Analysis / Critical Findings
KV cache size during generation caused memory spikes when serving batch inference.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_01_pretraining/historical_experiments/CS781_Phase1_Pretraining_V1_Decoder.ipynb`](notebooks/phase_01_pretraining/historical_experiments/CS781_Phase1_Pretraining_V1_Decoder.ipynb)
