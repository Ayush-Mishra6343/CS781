# Phase 4: Court Judgment Prediction and Explanation (CJPE)

**Task Description:** Hierarchical Local-Global SwiGLU-LoRMA Document Classifier over 16 chunks with rank super-ensemble.

- **Official Benchmark Score:** `Macro-F1: 0.6886 (Official Winning Score)`
- **Primary Notebook:** [`00_BEST_OFFICIAL_IMPLEMENTATION.ipynb`](00_BEST_OFFICIAL_IMPLEMENTATION.ipynb) (Identical to [`CS781_Phase4_CJPE_SwiGLU_LoRMA.ipynb`](CS781_Phase4_CJPE_SwiGLU_LoRMA.ipynb))
- **Historical Experiments:** [`historical_experiments/`](historical_experiments/) (15 notebooks preserved)

---

## Kaggle Execution & Environment Setup

This notebook is fully self-contained and ready to execute on Kaggle or any Jupyter/CUDA environment.

1. **Hardware Accelerator:** `GPU T4 (16GB VRAM) or P100`
2. **Estimated Runtime:** `~35 minutes (chunk extraction + hierarchical attention)`
3. **Dataset Attachment:**
   - On Kaggle: Click `+ Add Input` -> Search for `CJPE Indian High Court Judgment Outcomes` or upload the data archive from `data/`.
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

- [`CS781_Assignment_4_Questions_updated.ipynb`](historical_experiments/CS781_Assignment_4_Questions_updated.ipynb)
- [`CS781_Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb`](historical_experiments/CS781_Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb)
- [`Phase4_CJPE_V1_LoRA_Stage1.ipynb`](historical_experiments/Phase4_CJPE_V1_LoRA_Stage1.ipynb)
- [`Phase4_CJPE_V1_SoftPrompt_Adapter_Executed.ipynb`](historical_experiments/Phase4_CJPE_V1_SoftPrompt_Adapter_Executed.ipynb)
- [`Phase4_CJPE_V1_SoftPrompt_Adapter_Experiments.ipynb`](historical_experiments/Phase4_CJPE_V1_SoftPrompt_Adapter_Experiments.ipynb)
- [`Phase4_CJPE_V2_Improved.ipynb`](historical_experiments/Phase4_CJPE_V2_Improved.ipynb)
- [`Phase4_CJPE_V2_Improved_Local.ipynb`](historical_experiments/Phase4_CJPE_V2_Improved_Local.ipynb)
- [`Phase4_CJPE_V2_Improved_Local_Executed.ipynb`](historical_experiments/Phase4_CJPE_V2_Improved_Local_Executed.ipynb)
- [`Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb`](historical_experiments/Phase4_CJPE_V2_LoRMA_Plus_Stage2.ipynb)
- [`Phase4_CJPE_V3_BitFit_Stage3.ipynb`](historical_experiments/Phase4_CJPE_V3_BitFit_Stage3.ipynb)
- [`Phase4_CJPE_V4_LoRA_Stage4.ipynb`](historical_experiments/Phase4_CJPE_V4_LoRA_Stage4.ipynb)
- [`Phase4_CJPE_V5_FromScratch_KD.ipynb`](historical_experiments/Phase4_CJPE_V5_FromScratch_KD.ipynb)
- [`V1_EXECUTED.ipynb`](historical_experiments/V1_EXECUTED.ipynb)
- [`V1_PRE_RUN.ipynb`](historical_experiments/V1_PRE_RUN.ipynb)
- [`swiglu_local_global_experiment_phase_$.ipynb`](historical_experiments/swiglu_local_global_experiment_phase_$.ipynb)
