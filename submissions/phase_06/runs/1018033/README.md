# Submission 1018033 (Phase 6: Legal Judgment Summarization)

- **Submission ID:** `1018033`
- **Phase:** 6
- **Submission Date:** 2026-11-01 09:20
- **Portal Score:** **0.8243**
- **Evaluation Metric:** Composite BERTScore / ROUGE-L F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
InLegalBERT sequence-to-sequence summarizer baseline

## 2. Hyperparameters
```json
{
  "backbone": "law-ai/InLegalBERT",
  "beam_size": 4,
  "lr": 3e-05
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8243`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Good domain terminology, but standard greedy/beam decoding lagged behind distilled student + MBR consensus.

## 5. Failure Analysis / Critical Findings
Lacked the output candidate diversification that enabled MBR consensus decoding.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_06_summarization/historical_experiments/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb`](notebooks/phase_06_summarization/historical_experiments/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb)
