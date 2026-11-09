# Submission 975506 (Phase 2: Legal Named Entity Recognition (L-NER))

- **Submission ID:** `975506`
- **Phase:** 2
- **Submission Date:** 2026-10-10 19:30
- **Portal Score:** **0.1386**
- **Evaluation Metric:** Strict Micro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
BERT-base-uncased baseline with default WordPiece tokenization

## 2. Hyperparameters
```json
{
  "backbone": "bert-base-uncased",
  "lr": 5e-05,
  "batch_size": 32,
  "epochs": 3
}
```

## 3. Results & Evaluation
- **Official Score:** `0.1386`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Initial naive benchmark. Severely impacted by WordPiece out-of-vocabulary splits on Indian legal citations and Latin maxims.

## 5. Failure Analysis / Critical Findings
Severe fragmentation of legal vocabulary (e.g. 'suo motu', 'certiorari', 'IPC').

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb`](notebooks/phase_02_ner/historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb)
