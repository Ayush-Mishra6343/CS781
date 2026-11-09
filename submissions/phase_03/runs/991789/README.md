# Submission 991789 (Phase 3: Rhetorical Role (RR) Classification & PEFT Suite)

- **Submission ID:** `991789`
- **Phase:** 3
- **Submission Date:** 2026-10-15 18:12
- **Portal Score:** **0.4144**
- **Evaluation Metric:** Macro F1
- **Status:** Superseded

## 1. Technical Approach & Architecture
Longformer-Legal with asymmetric loss and context window 2048

## 2. Hyperparameters
```json
{
  "backbone": "allenai/longformer-base-4096",
  "max_len": 2048,
  "lr": 3e-05,
  "batch_size": 4
}
```

## 3. Results & Evaluation
- **Official Score:** `0.4144`
- **Evaluation Harness:** Codabench Standard Execution Environment

## 4. Why Retained / Why Superseded
Strong multi-sentence context handling (0.4144), confirming that rhetorical roles heavily depend on surrounding document discourse.

## 5. Failure Analysis / Critical Findings
High memory footprint restricted batch size to 4, limiting gradient stability.

## 6. Notebook & Execution Provenance
Generated from: [`notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb`](notebooks/phase_03_rhetorical_roles/historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb)
