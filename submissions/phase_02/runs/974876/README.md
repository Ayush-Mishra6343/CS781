# Submission 974876 (Phase 2: Legal Named Entity Recognition (L-NER))

- **Submission ID:** `974876`
- **Phase:** 2
- **Submission Date:** 2026-10-10 20:18
- **Portal Score:** **0.3223**
- **Evaluation Metric:** Strict Micro F1
- **Status:** Selected (Winning)

## 1. Technical Approach & Architecture
RoBERTa-Legal Token Classifier + Calibrated Strict Character Boundary Alignment

## 2. Hyperparameters
```json
{
  "backbone": "roberta-base",
  "lr": 3e-05,
  "batch_size": 16,
  "epochs": 6,
  "threshold": 0.42,
  "boundary_calibrated": true,
  "label_count": 14
}
```

## 3. Results & Evaluation
- **Official Score:** `0.3223`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Official winning model for Phase 2. Solved severe subtoken-to-character offset misalignment in legal statute spans (petitioner, respondent, judge).

## 5. Failure Analysis / Critical Findings
Raw subword span detokenization without boundary offset matching lost up to 40% of micro-F1 due to punctuation whitespace shifts.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_02_ner/CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb`](notebooks/phase_02_ner/CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb)
