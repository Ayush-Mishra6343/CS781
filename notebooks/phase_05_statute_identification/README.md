# Phase 5: Legal Statute Identification (LSI)

**Task Description:** Multi-label classification of 100 Indian Penal Code sections with step-zero identity initialization.

- **Official Benchmark Score:** `Macro-F1: 0.3700 across 100 penal sections`
- **Primary Notebook:** [`00_BEST_OFFICIAL_IMPLEMENTATION.ipynb`](00_BEST_OFFICIAL_IMPLEMENTATION.ipynb) (Identical to [`CS781_Phase5_LSI_Hierarchical_LoRMA.ipynb`](CS781_Phase5_LSI_Hierarchical_LoRMA.ipynb))
- **Historical Experiments:** [`historical_experiments/`](historical_experiments/) (7 notebooks preserved)

---

## Kaggle Execution & Environment Setup

This notebook is fully self-contained and ready to execute on Kaggle or any Jupyter/CUDA environment.

1. **Hardware Accelerator:** `GPU T4 (16GB VRAM)`
2. **Estimated Runtime:** `~30 minutes (multi-label hierarchical training)`
3. **Dataset Attachment:**
   - On Kaggle: Click `+ Add Input` -> Search for `Legal Statute Case Summaries & IPC Ground Truth` or upload the data archive from `data/`.
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

- [`CS781_Assignment_5_Questions.ipynb`](historical_experiments/CS781_Assignment_5_Questions.ipynb)
- [`V1_LORMA_KD_FROM_V3.ipynb`](historical_experiments/V1_LORMA_KD_FROM_V3.ipynb)
- [`V2_KAGGLE_PARTIAL_FT.ipynb`](historical_experiments/V2_KAGGLE_PARTIAL_FT.ipynb)
- [`V3_ADALORA_APEX_R32.ipynb`](historical_experiments/V3_ADALORA_APEX_R32.ipynb)
- [`V3_KAGGLE_ADALORA.ipynb`](historical_experiments/V3_KAGGLE_ADALORA.ipynb)
- [`V4_KAGGLE_HYBRID_LSI.ipynb`](historical_experiments/V4_KAGGLE_HYBRID_LSI.ipynb)
- [`V4_SUPER_APEX_LSI.ipynb`](historical_experiments/V4_SUPER_APEX_LSI.ipynb)
