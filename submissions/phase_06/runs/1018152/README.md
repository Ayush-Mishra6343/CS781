# Submission 1018152 (Phase 6: Legal Judgment Summarization)

- **Submission ID:** `1018152`
- **Phase:** 6
- **Submission Date:** 2026-11-01 22:05
- **Portal Score:** **0.7590**
- **Evaluation Metric:** Composite BERTScore / ROUGE-L F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
DeBERTa-v3 sequence-to-sequence summarizer baseline

## 2. Hyperparameters
```json
{
  "backbone": "microsoft/deberta-v3-base",
  "max_input_len": 1024,
  "max_summary_len": 256,
  "lr": 3e-05
}
```

## 3. Results & Evaluation
- **Official Score:** `0.7590`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Encoder-decoder formulation using DeBERTa embeddings. Reasonable summary fluency but missed mid-document legal holdings.

## 5. Failure Analysis / Critical Findings
Input length limit of 1024 tokens caused severe document truncation.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_06_summarization/historical_experiments/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb`](notebooks/phase_06_summarization/historical_experiments/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb)
