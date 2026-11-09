# Submission 990317 (Phase 3: Rhetorical Role (RR) Classification & PEFT Suite)

- **Submission ID:** `990317`
- **Phase:** 3
- **Submission Date:** 2026-10-15 11:20
- **Portal Score:** **0.4292**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Longformer-Legal with Focal Loss (gamma=2.0)

## 2. Hyperparameters
```json
{
  "backbone": "allenai/longformer-base-4096",
  "max_len": 2048,
  "gamma": 2.0,
  "lr": 3e-05
}
```

## 3. Results & Evaluation
- **Official Score:** `0.4292`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Outstanding individual long-context model (0.4292). Handled long legal preambles and factual narratives effectively.

## 5. Failure Analysis / Critical Findings
Slightly less sample-efficient than the PEFT RoBERTa ensemble on short, punchy legal holdings.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb`](notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb)
