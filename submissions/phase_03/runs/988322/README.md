# Submission 988322 (Phase 3: Rhetorical Role (RR) Classification & PEFT Suite)

- **Submission ID:** `988322`
- **Phase:** 3
- **Submission Date:** 2026-10-14 20:05
- **Portal Score:** **0.4278**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Hierarchical Sentence BiLSTM on top of frozen Legal-RoBERTa sentence embeddings

## 2. Hyperparameters
```json
{
  "backbone": "frozen legal-roberta",
  "bilstm_hidden": 256,
  "bilstm_layers": 2,
  "lr": 0.001,
  "dropout": 0.3
}
```

## 3. Results & Evaluation
- **Official Score:** `0.4278`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Explicit document-level discourse modeling (0.4278). Showed that sequential sentence progression (Facts -> Issues -> Arguments -> Ratio) carries strong temporal priors.

## 5. Failure Analysis / Critical Findings
Frozen sentence embeddings could not adapt token-level representations to legal domain nuances.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb`](notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb)
