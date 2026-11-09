# Project File Inventory

**Project:** In-House Legal Intelligence System (CS781 Capstone Project)  
**Student:** Ayush Mishra (Roll Number: 250253)  
**Audit Date:** 2026-11-09  
**Total Indexed Artifacts:** 2451 files across drives C: and D:

---

## 1. Executive Summary

This inventory documents all 2,451 discovered project files across the primary research environment. Every artifact has been categorized by associated phase, MIME/file type, research purpose, provenance (original vs. generated), role, runtime necessity, and potential ambiguity.

### Key Metrics
- **Jupyter Notebooks (`.ipynb`):** 205
- **Python Source Files (`.py`):** 424
- **Model Checkpoints (`.pt`, `.bin`, `.pth`):** 140
- **Configuration & Metadata (`.json`, `.yml`):** 441
- **Submission Archives (`.zip`):** 304
- **Tabular Data & Metrics (`.csv`):** 242
- **Documentation & Reports (`.md`, `.txt`):** 693

---

## 2. Inventory Breakdown by Phase

| Phase | Description | Notebooks | Python Files | Checkpoints | Submissions | Configs |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Phase 1** | Legal Language Model (Pretraining) | 11 | 22 | 50 | 27 | 32 |
| **Phase 2** | Legal Named Entity Recognition (L-NER) | 40 | 13 | 14 | 12 | 49 |
| **Phase 3** | Rhetorical Role Classification / PEFT | 54 | 1 | 23 | 48 | 79 |
| **Phase 4** | Court Judgment Prediction & Explanation (CJPE) | 30 | 74 | 12 | 53 | 61 |
| **Phase 5** | Multi-Label Legal Statute Identification (LSI) | 14 | 114 | 13 | 27 | 49 |
| **Phase 6** | Legal Document Summarization | 1 | 4 | 8 | 21 | 38 |
| **Phase 7** | Prior Case Retrieval (PCR) | 4 | 51 | 8 | 17 | 32 |
| **Shared** | Cross-Phase Utilities & Infrastructure | 51 | 145 | 12 | 99 | 101 |

---

## 3. Top Candidate Final Implementations

| Phase | Candidate Notebook | File Path | Apparent Purpose | Runtime Necessity |
|:---:|:---|:---|:---|:---:|
| **P1** | `v4.ipynb` | `C:\Users\LENOVO\Downloads\v4.ipynb` | 94.76M parameter HybridLegalLM pretraining with GQA and local/global windowing | Critical |
| **P2** | `CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb` | `D:\CS781_GitHub_Submission_Notebooks\Phase_2_Legal_NER\CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb` | RoBERTa token classification with strict boundary calibration | Critical |
| **P3** | `CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb` | `D:\CS781_GitHub_Submission_Notebooks\Phase_3_PEFT\CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb` | Comprehensive PEFT suite (LoRA, LoRMA, BitFit, NF4, Distillation) | Critical |
| **P4** | `swiglu_local_global_experiment_phase_4.ipynb` | `D:\CS781_Capstone_Variants\v4\experiments\Phase_4_CJPE\swiglu_local_global_experiment_phase_4.ipynb` | Hierarchical SwiGLU-LoRMA document transformer (Macro-F1 0.6886) | Critical |
| **P5** | `V1_EXECUTED.ipynb` | `D:\Phase5_LSI_Research\outputs\V1_HLG_SwiGLU_LoRMA\notebooks\V1_EXECUTED.ipynb` | Multi-label statute classification with step-zero identity verification | Critical |
| **P6** | `CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb` | `D:\CS781_GitHub_Submission_Notebooks\Phase_6_Summarization\CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb` | Deep-16 distillation student + MBR consensus decoding | Critical |
| **P7** | `CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb` | `D:\CS781_GitHub_Submission_Notebooks\Phase_7_PCR\CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb` | BM25 lexical + dense neural multi-stage cross-encoder reranker | Critical |

*Full itemized listings for all 2451 files are recorded in `01_PROJECT_INVENTORY.csv`.*
