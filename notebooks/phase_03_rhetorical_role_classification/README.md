# Phase 3: Rhetorical Role (RR) Classification & PEFT Suite

**Task Description:** Multi-method PEFT comparison (LoRA, LoRMA, BitFit) and ensemble for sentence-level rhetorical role classification.

- **Official Benchmark Score:** `Macro-F1: 0.4337 (Multi-Method Ensemble)`
- **Primary Notebook:** [`00_BEST_OFFICIAL_IMPLEMENTATION.ipynb`](00_BEST_OFFICIAL_IMPLEMENTATION.ipynb) (Identical to [`CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb`](CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb))
- **Historical Experiments:** [`historical_experiments/`](historical_experiments/) (24 notebooks preserved)

---

## Kaggle Execution & Environment Setup

This notebook is fully self-contained and ready to execute on Kaggle or any Jupyter/CUDA environment.

1. **Hardware Accelerator:** `GPU T4 (16GB VRAM)`
2. **Estimated Runtime:** `~20 minutes (parameter-efficient adapter training)`
3. **Dataset Attachment:**
   - On Kaggle: Click `+ Add Input` -> Search for `Rhetorical Role Judgment Sentences (7 discourse roles)` or upload the data archive from `data/`.
   - Locally: Place raw JSON datasets in `./data` or set environment variable `CS781_DATA_DIR`.
4. **Dependency Installation:** The first code cell automatically configures environment dependencies.
5. **Execution Order:** Run cells sequentially from Section 1 through Section 12.

---

## Standardized 12-Section Pipeline Architecture

| Section | Purpose | Key Operations |
|:---:|:---|:---|
| **1** | Environment & Hardware Diagnostics | CUDA detection, GPU memory allocation, dependency verification |
| **2** | Reproducibility & Seed Configuration | Deterministic seed lock (`torch.manual_seed(42)`, cuDNN deterministic) |
| **3** | Dataset Loading & Verification | Automated file discovery, schema validation, split verification |
| **4** | Tokenization & Preprocessing | Domain BPE / HuggingFace subword alignment, tensor encoding |
| **5** | Model Architecture & Parameter Count | Instantiate winning model, count trainable vs frozen parameters |
| **6** | Optimization Dynamics | AdamW optimizer, linear/cosine schedule, gradient clipping |
| **7** | Training & Validation Checkpoints | Loss trajectory logging, best validation checkpoint preservation |
| **8** | Formal Validation & Metric Evaluation | Compute exact official competition metric on validation split |
| **9** | Error Analysis & Diagnostics | Per-class breakdown, confusion matrices, failure analysis |
| **10** | Codabench Submission Pipeline | Generate test predictions formatted to exact competition schema |
| **11** | Submission Package Verification | Create submission ZIP archive, verify integrity and SHA-256 hash |
| **12** | Provenance & Execution Summary | Log runtime statistics, environment hashes, and audit record |

---

## Historical Experiments Preserved in this Phase

- [`CS781_Assignment3_work_so_far (1).ipynb`](historical_experiments/CS781_Assignment3_work_so_far (1).ipynb)
- [`CS781_Assignment3_work_so_far.ipynb`](historical_experiments/CS781_Assignment3_work_so_far.ipynb)
- [`CS781_Assignment4_actual_run_results.ipynb`](historical_experiments/CS781_Assignment4_actual_run_results.ipynb)
- [`CS781_Assignment_3_FinalTrainReady_NoPrefilledConclusions.ipynb`](historical_experiments/CS781_Assignment_3_FinalTrainReady_NoPrefilledConclusions.ipynb)
- [`CS781_Assignment_3_Q1_Completed.ipynb`](historical_experiments/CS781_Assignment_3_Q1_Completed.ipynb)
- [`CS781_Assignment_3_Q2_Completed.ipynb`](historical_experiments/CS781_Assignment_3_Q2_Completed.ipynb)
- [`CS781_Assignment_3_Q3_Completed.ipynb`](historical_experiments/CS781_Assignment_3_Q3_Completed.ipynb)
- [`CS781_Assignment_3_Q4_Completed.ipynb`](historical_experiments/CS781_Assignment_3_Q4_Completed.ipynb)
- [`CS781_Assignment_3_Q5_Completed.ipynb`](historical_experiments/CS781_Assignment_3_Q5_Completed.ipynb)
- [`CS781_Assignment_3_Questions (1).ipynb`](historical_experiments/CS781_Assignment_3_Questions (1).ipynb)
- [`CS781_Assignment_4_Kaggle.ipynb`](historical_experiments/CS781_Assignment_4_Kaggle.ipynb)
- [`Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb`](historical_experiments/Phase3_RR_V1_Balanced_Context_v3_Codabench.ipynb)
- [`Phase3_RR_V1_Baseline.ipynb`](historical_experiments/Phase3_RR_V1_Baseline.ipynb)
- [`Phase3_RR_V1_PEFT_LoRA_LoRMA_BitFit.ipynb`](historical_experiments/Phase3_RR_V1_PEFT_LoRA_LoRMA_BitFit.ipynb)
- [`Phase3_RR_V1_Safe_Retrain_v5.ipynb`](historical_experiments/Phase3_RR_V1_Safe_Retrain_v5.ipynb)
- [`Phase3_RR_V1_Sequence_CV_v4.ipynb`](historical_experiments/Phase3_RR_V1_Sequence_CV_v4.ipynb)
- [`Phase3_RR_V1_Sequence_CV_v4_fixed.ipynb`](historical_experiments/Phase3_RR_V1_Sequence_CV_v4_fixed.ipynb)
- [`Phase3_RR_V1_Three_Model_Ensemble_v8.ipynb`](historical_experiments/Phase3_RR_V1_Three_Model_Ensemble_v8.ipynb)
- [`Phase3_RR_V2_PEFT_LoRA_LoRMA_BitFit_KD.ipynb`](historical_experiments/Phase3_RR_V2_PEFT_LoRA_LoRMA_BitFit_KD.ipynb)
- [`Phase3_RR_V3_LoRA_LoRMA_KD_Kaggle.ipynb`](historical_experiments/Phase3_RR_V3_LoRA_LoRMA_KD_Kaggle.ipynb)
- [`assignment3_instruction_aligned.ipynb`](historical_experiments/assignment3_instruction_aligned.ipynb)
- [`notebook.ipynb`](historical_experiments/notebook.ipynb)
- [`phase_3_v3.ipynb`](historical_experiments/phase_3_v3.ipynb)
- [`phase_3_v4.ipynb`](historical_experiments/phase_3_v4.ipynb)
