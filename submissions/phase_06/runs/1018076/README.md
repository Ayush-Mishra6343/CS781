# Submission 1018076 (Phase 6: Legal Judgment Summarization)

- **Submission ID:** `1018076`
- **Phase:** 6
- **Submission Date:** 2026-11-01 12:40
- **Portal Score:** **0.8316**
- **Evaluation Metric:** Composite BERTScore / ROUGE-L F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
RoBERTa-base sequence-to-sequence summarizer with beam search

## 2. Hyperparameters
```json
{
  "backbone": "roberta-base",
  "beam_size": 4,
  "length_penalty": 0.8
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8316`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Parallel run testing general domain checkpoint vs legal-adapted. Produced identical portal score (0.8316) but slightly less accurate statutory terms.

## 5. Failure Analysis / Critical Findings
Occasionally substituted general English synonyms for precise legal terms of art.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_06_summarization/historical_experiments/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb`](notebooks/phase_06_summarization/historical_experiments/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb)
