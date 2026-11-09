# Submission 1029981 (Phase 6: Legal Judgment Summarization)

- **Submission ID:** `1029981`
- **Phase:** 6
- **Submission Date:** 2026-11-03 16:29
- **Portal Score:** **0.8363**
- **Evaluation Metric:** Composite BERTScore / ROUGE-L F1
- **Status:** Selected (Winning)

## 1. Technical Approach & Architecture
Deep-16 Knowledge Distillation Student (16 layers, 512 hidden) + Minimum Bayes Risk (MBR) Consensus Decoding

## 2. Hyperparameters
```json
{
  "student_layers": 16,
  "hidden_dim": 512,
  "teacher": "Legal-RoBERTa-Large",
  "distill_temp": 2.0,
  "alpha_distill": 0.5,
  "mbr_candidates": 16,
  "mbr_metric": "bertscore_f1"
}
```

## 3. Results & Evaluation
- **Official Score:** `0.8363`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Official winning model for Phase 6. Compressed a massive legal summarizer into a 16-layer student via distillation, then used MBR decoding across 16 stochastic candidate summaries to pick the maximum expected utility summary, reaching 0.8363.

## 5. Failure Analysis / Critical Findings
Standard greedy decoding produced repetitive legal boilerplate; MBR consensus decoding eliminated halluncinations and repetitive phrasing.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_06_summarization/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb`](notebooks/phase_06_summarization/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb)
