# Kaggle Compatibility Audit

**Project:** In-House Legal Intelligence System (CS781 Capstone Project)  
**Student:** Ayush Mishra (Roll Number: 250253)  
**Audit Standard:** Part 8 Specification (Kaggle-First Execution Assessment)

---

## 1. Kaggle Execution Compatibility Assessment

All seven research notebooks have been evaluated against standard Kaggle execution constraints (Kaggle GPU P100 / 2x T4 environments):

| Phase | Primary Accelerator | Expected VRAM | Internet Required | Attached Input Datasets | Expected Runtime | Kaggle Status |
|:---:|:---:|:---:|:---:|:---|:---:|:---|
| **Phase 1** | GPU (NVIDIA P100 / T4) | ~11.5 GB | Yes (HF download) or Local dataset | `cs781-legal-pretrain-corpus`, `bpe-25k` | ~3.5 hours (100M tokens) | **Kaggle-Verified (Full Run Preserved)** |
| **Phase 2** | GPU (NVIDIA T4 / P100) | ~8.0 GB | Yes (Hub download) | `cs781-lner-data` | ~45 minutes (6 epochs) | **Kaggle-Verified (Full Run Preserved)** |
| **Phase 3** | GPU (NVIDIA T4 / P100) | ~6.5 GB | Optional (offline mode supported) | `cs781-peft-artifacts` | ~30 minutes (all modules) | **Kaggle-Verified (Full Run Preserved)** |
| **Phase 4** | GPU (NVIDIA T4 / P100) | ~9.0 GB | Optional | `cs781-cjpe-data`, `swiglu-checkpoints` | ~40 minutes (6 epochs) | **Kaggle-Verified (Full Run Preserved)** |
| **Phase 5** | GPU (NVIDIA T4 / P100) | ~10.0 GB | Optional | `cs781-lsi-statutes` | ~50 minutes | **Kaggle-Verified (Full Run Preserved)** |
| **Phase 6** | GPU (NVIDIA T4 / P100) | ~12.0 GB | Optional | `cs781-summarization-corpus` | ~1.2 hours (MBR decoding) | **Kaggle-Verified (Full Run Preserved)** |
| **Phase 7** | GPU (NVIDIA T4 / P100) | ~7.0 GB | Optional | `cs781-pcr-queries`, `pcr-documents` | ~25 minutes (237 queries) | **Kaggle-Verified (Full Run Preserved)** |

---

## 2. Kaggle-First Execution Checklist per Notebook

Each notebook directory in `in-house-legal-intelligence/notebooks/` includes an execution guide covering:
1. **Notebook Environment:** Standard Kaggle GPU session (Python 3.10+).
2. **Dataset Attachment:** Step-by-step instructions for attaching input data under `/kaggle/input/`.
3. **Hardware Selection:** GPU P100 or 2x T4 (Persistence enabled).
4. **Internet Setting:** Enabled for initial dependency verification or pre-cached for offline execution.
5. **Output Delivery:** Automatic serialization of predictions to `/kaggle/working/` ready for download and Codabench benchmark submission.
