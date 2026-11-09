# Submission 1018121 (Phase 6: Legal Judgment Summarization)

- **Submission ID:** `1018121`
- **Phase:** 6
- **Submission Date:** 2026-11-01 15:15
- **Portal Score:** **0.8316**
- **Evaluation Metric:** Composite BERTScore / ROUGE-L F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Legal-RoBERTa generative summarizer with beam search (beam_size=4)

## 2. Hyperparameters
```json
{
  "backbone": "legal-roberta-base",
  "beam_size": 4,
  "length_penalty": 1.0,
  "no_repeat_ngram_size": 3
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8316`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Strong standard beam search result (0.8316). Clean summary structures.

## 5. Failure Analysis / Critical Findings
Beam search tended to favor generic safe phrases rather than distinctive legal citations.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_06_summarization/historical_experiments/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb`](notebooks/phase_06_summarization/historical_experiments/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb)
