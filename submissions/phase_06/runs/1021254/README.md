# Submission 1021254 (Phase 6: Legal Judgment Summarization)

- **Submission ID:** `1021254`
- **Phase:** 6
- **Submission Date:** 2026-11-02 19:10
- **Portal Score:** **0.8356**
- **Evaluation Metric:** Composite BERTScore / ROUGE-L F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Omni-Ensemble MBR (V1-Deep16, LoRA, LoRMA, Bottleneck Adapters)

## 2. Hyperparameters
```json
{
  "ensemble_size": 4,
  "mbr_candidates": 16,
  "mbr_metric": "bertscore"
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8356`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Highly diverse 4-model candidate pool for MBR selection. Delivered 0.8356 score.

## 5. Failure Analysis / Critical Findings
Generating candidates across 4 distinct models increased inference latency by 3.8x.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_06_summarization/historical_experiments/PHASE6_CHAMPIONSHIP_ANALYSIS.ipynb`](notebooks/phase_06_summarization/historical_experiments/PHASE6_CHAMPIONSHIP_ANALYSIS.ipynb)
