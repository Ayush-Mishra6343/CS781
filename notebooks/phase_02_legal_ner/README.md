# Phase 2: Legal Named Entity Recognition (L-NER)

**Task Description:** Extracting 14 legal entity types with calibrated exact character boundary alignment.

- **Official Benchmark Score:** `Strict Micro-F1: 0.3223 (Calibrated Boundary F1: 0.7226)`
- **Primary Notebook:** [`00_BEST_OFFICIAL_IMPLEMENTATION.ipynb`](00_BEST_OFFICIAL_IMPLEMENTATION.ipynb) (Identical to [`CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb`](CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb))
- **Historical Experiments:** [`historical_experiments/`](historical_experiments/) (22 notebooks preserved)

---

## Kaggle Execution & Environment Setup

This notebook is fully self-contained and ready to execute on Kaggle or any Jupyter/CUDA environment.

1. **Hardware Accelerator:** `GPU T4 (16GB VRAM)`
2. **Estimated Runtime:** `~25 minutes (6 epochs fine-tuning and calibration)`
3. **Dataset Attachment:**
   - On Kaggle: Click `+ Add Input` -> Search for `Legal NER Judgment Corpus (14 domain entity classes)` or upload the data archive from `data/`.
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

- [`Legal_NER_Phase2_CharacterBoundary_6Epoch_Calibrated_FIXED.ipynb`](historical_experiments/Legal_NER_Phase2_CharacterBoundary_6Epoch_Calibrated_FIXED.ipynb)
- [`Legal_NER_Phase2_CharacterBoundary_Antigravity_Edited.ipynb`](historical_experiments/Legal_NER_Phase2_CharacterBoundary_Antigravity_Edited.ipynb)
- [`Legal_NER_Phase2_Character_Boundary_Experiment.ipynb`](historical_experiments/Legal_NER_Phase2_Character_Boundary_Experiment.ipynb)
- [`Legal_NER_Phase2_Uploaded_Hybrid_LORMA_KD_DEVICE_FIXED.ipynb`](historical_experiments/Legal_NER_Phase2_Uploaded_Hybrid_LORMA_KD_DEVICE_FIXED.ipynb)
- [`Legal_NER_Phase2_Uploaded_Hybrid_LORMA_KD_Single_Submission.ipynb`](historical_experiments/Legal_NER_Phase2_Uploaded_Hybrid_LORMA_KD_Single_Submission.ipynb)
- [`Legal_NER_Phase2_V1_0p323_Rescue.ipynb`](historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue.ipynb)
- [`Legal_NER_Phase2_V1_0p323_Rescue_Antigravity_Reviewed.ipynb`](historical_experiments/Legal_NER_Phase2_V1_0p323_Rescue_Antigravity_Reviewed.ipynb)
- [`Legal_NER_Phase2_V1_5Fold_OOF_Ensemble.ipynb`](historical_experiments/Legal_NER_Phase2_V1_5Fold_OOF_Ensemble.ipynb)
- [`Legal_NER_Phase2_V1_BestBranch_ValidationSelected.ipynb`](historical_experiments/Legal_NER_Phase2_V1_BestBranch_ValidationSelected.ipynb)
- [`Legal_NER_Phase2_V1_Decoder_Repair_Cells.ipynb`](historical_experiments/Legal_NER_Phase2_V1_Decoder_Repair_Cells.ipynb)
- [`Legal_NER_Phase2_V1_Decoder_Repair_Reviewed.ipynb`](historical_experiments/Legal_NER_Phase2_V1_Decoder_Repair_Reviewed.ipynb)
- [`Legal_NER_Phase2_V1_Recovery_Control_Single_Submission.ipynb`](historical_experiments/Legal_NER_Phase2_V1_Recovery_Control_Single_Submission.ipynb)
- [`Legal_NER_Phase2_V1_Recovery_Control_Single_Submission_Antigravity_Reviewed.ipynb`](historical_experiments/Legal_NER_Phase2_V1_Recovery_Control_Single_Submission_Antigravity_Reviewed.ipynb)
- [`Legal_NER_Phase2_V2_LORMA_CharacterBoundary.ipynb`](historical_experiments/Legal_NER_Phase2_V2_LORMA_CharacterBoundary.ipynb)
- [`Legal_NER_Phase2_V2_LORMA_CharacterBoundary_Antigravity_Reviewed.ipynb`](historical_experiments/Legal_NER_Phase2_V2_LORMA_CharacterBoundary_Antigravity_Reviewed.ipynb)
- [`V1_Phase2_LNER_dev_v3.ipynb`](historical_experiments/V1_Phase2_LNER_dev_v3.ipynb)
- [`V1_Phase2_LNER_integrated.ipynb`](historical_experiments/V1_Phase2_LNER_integrated.ipynb)
- [`V1_Phase2_LNER_integrated_final.ipynb`](historical_experiments/V1_Phase2_LNER_integrated_final.ipynb)
- [`V1_Phase2_LNER_preparation.ipynb`](historical_experiments/V1_Phase2_LNER_preparation.ipynb)
- [`V1_Phase2_LNER_ready.ipynb`](historical_experiments/V1_Phase2_LNER_ready.ipynb)
- [`V1_Phase2_LNER_submission_ready.ipynb`](historical_experiments/V1_Phase2_LNER_submission_ready.ipynb)
- [`V1_Phase2_LNER_submission_ready_fixed.ipynb`](historical_experiments/V1_Phase2_LNER_submission_ready_fixed.ipynb)
