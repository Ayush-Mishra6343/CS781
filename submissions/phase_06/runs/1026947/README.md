# Submission 1026947 (Phase 6: Legal Judgment Summarization)

- **Submission ID:** `1026947`
- **Phase:** 6
- **Submission Date:** 2026-11-03 11:45
- **Portal Score:** **0.8362**
- **Evaluation Metric:** Composite BERTScore / ROUGE-L F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
TIES-Merged Champion Model with MBR Consensus (sample size 16)

## 2. Hyperparameters
```json
{
  "merge_method": "TIES",
  "models_merged": [
    "Deep16-KD",
    "LoRA-Wide",
    "LoRMA-V4"
  ],
  "density": 0.2,
  "mbr_candidates": 16
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8362`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Remarkable 0.8362 score (0.0001 from peak). Trimming, Electing sign, and Scaling (TIES) successfully merged parameter weights without interference.

## 5. Failure Analysis / Critical Findings
Marginally lower coherence on complex constitutional judgment summaries compared to pure KD Deep-16.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_06_summarization/historical_experiments/PHASE6_CHAMPIONSHIP_ANALYSIS.ipynb`](notebooks/phase_06_summarization/historical_experiments/PHASE6_CHAMPIONSHIP_ANALYSIS.ipynb)
