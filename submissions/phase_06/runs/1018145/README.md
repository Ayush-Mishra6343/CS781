# Submission 1018145 (Phase 6: Legal Judgment Summarization)

- **Submission ID:** `1018145`
- **Phase:** 6
- **Submission Date:** 2026-11-01 18:30
- **Portal Score:** **0.8067**
- **Evaluation Metric:** Composite BERTScore / ROUGE-L F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Longformer-Encoder-Decoder (LED) baseline (4096 input tokens)

## 2. Hyperparameters
```json
{
  "backbone": "allenai/led-base-16384",
  "max_input_len": 4096,
  "max_summary_len": 512,
  "lr": 5e-05
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8067`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Handled full case transcripts without truncation (0.8067). Great factual coverage.

## 5. Failure Analysis / Critical Findings
Abstractive generation suffered from abstract hallucinations without MBR risk minimization.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_06_summarization/historical_experiments/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb`](notebooks/phase_06_summarization/historical_experiments/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb)
