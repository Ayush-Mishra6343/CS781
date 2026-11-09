# Submission 956150 (Phase 1: Legal Language Model Pre-Training)

- **Submission ID:** `956150`
- **Phase:** 1
- **Submission Date:** 2026-10-03 04:36
- **Portal Score:** **0.8212**
- **Evaluation Metric:** Token Prediction Accuracy
- **Status:** Superseded

## 1. Technical Approach & Architecture
Decoder LM with custom Indian Legal BPE 16k Tokenizer and Cosine Annealing

## 2. Hyperparameters
```json
{
  "layers": 10,
  "hidden_dim": 640,
  "heads": 10,
  "vocab_size": 16000,
  "lr": 0.0005,
  "weight_decay": 0.01,
  "batch_size": 32,
  "max_seq_len": 512
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8212`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Validated custom 16k legal vocabulary against HuggingFace base tokenizers. Good speed but 512 context truncated long statutory citations.

## 5. Failure Analysis / Critical Findings
512-token maximum sequence length proved insufficient for multi-paragraph Indian Supreme Court judgments.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_01_pretraining/historical_experiments/CS781_Phase1_A1_BPE_Tokenizer.ipynb`](notebooks/phase_01_pretraining/historical_experiments/CS781_Phase1_A1_BPE_Tokenizer.ipynb)
