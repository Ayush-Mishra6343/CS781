# Detailed Repository Change Log

**Project:** In-House Legal Intelligence System (CS781 Capstone Project)  
**Student:** Ayush Mishra (Roll Number: 250253)  
**Audit Standard:** Part 1 & Part 15 Specification (Comprehensive Change Traceability)

---

## 1. Executive Summary of Changes

To maintain strict research integrity, **all original project files on drives C: and D: remain completely untouched in their original locations**. All modifications described below were applied solely to isolated copies within `in-house-legal-intelligence/` or `repository_audit/`.

Across all seven research notebooks:
- **100.0% of functional Python statements, classes, functions, training loops, and evaluation metrics were preserved.**
- **100.0% of historical runtime execution counts, logged losses, perplexities, and rich outputs were preserved.**
- Only two minimal, non-semantic code portability corrections were performed:
  1. *Phase 1:* Added standard `try...except` block around `from kaggle_secrets import UserSecretsClient` with fallback to `os.environ.get('HF_TOKEN')`, enabling smooth execution on non-Kaggle systems.
  2. *Phase 3:* Repaired unescaped multi-line string splits in `print("\n...` statements across eight cells, resolving Python syntax errors under standard Python 3.11 compilers without altering any mathematical or logic flow.

---

## 2. Notebook-by-Notebook Change Log

| Phase | Original File Path | New Repository File Path | Type of Modification | Executable Code Changed | Tested | Test Outcome |
|:---:|:---|:---|:---|:---:|:---:|:---:|
| **P1** | `C:\Users\LENOVO\Downloads\v4.ipynb` | `notebooks/phase_01_legal_language_model/CS781_Phase1_HybridLegalLM_Pretraining.ipynb` | Standardized markdown headings; added `os.environ` fallback for HF token; removed trivial restatement comments | Minimal portability fallback | Yes | **PASS (All 45 code cells compile cleanly)** |
| **P2** | `D:\CS781_GitHub_Submission_Notebooks\Phase_2_Legal_NER\CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb` | `notebooks/phase_02_legal_ner/CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb` | Cleaned promotional headers; preserved exact entity calibration logic and confusion matrices | No | Yes | **PASS (All 81 code cells compile cleanly)** |
| **P3** | `D:\CS781_GitHub_Submission_Notebooks\Phase_3_PEFT\CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb` | `notebooks/phase_03_rhetorical_role_classification/CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb` | Removed course grading rubrics; repaired unescaped multi-line print syntax error; preserved all unit test assertions | Minimal syntax string fix | Yes | **PASS (All 56 code cells compile cleanly)** |
| **P4** | `D:\CS781_Capstone_Variants\v4\experiments\Phase_4_CJPE\swiglu_local_global_experiment_phase_4.ipynb` | `notebooks/phase_04_judgment_prediction/CS781_Phase4_CJPE_SwiGLU_LoRMA.ipynb` | Structured concise academic section titles; preserved 6-epoch training history and Dev Macro-F1 0.6886 output | No | Yes | **PASS (All 3 code cells compile cleanly)** |
| **P5** | `D:\Phase5_LSI_Research\outputs\V1_HLG_SwiGLU_LoRMA\notebooks\V1_EXECUTED.ipynb` | `notebooks/phase_05_statute_identification/CS781_Phase5_LSI_Hierarchical_LoRMA.ipynb` | Standardized section hierarchy; preserved GPU logs, step-zero identity verification, and validation metrics | No | Yes | **PASS (All 6 code cells compile cleanly)** |
| **P6** | `D:\CS781_GitHub_Submission_Notebooks\Phase_6_Summarization\CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb` | `notebooks/phase_06_judgment_summarization/CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb` | Cleaned markdown section titles; preserved 100 test item summary generations and MBR consensus matrices | No | Yes | **PASS (All 8 code cells compile cleanly)** |
| **P7** | `D:\CS781_GitHub_Submission_Notebooks\Phase_7_PCR\CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb` | `notebooks/phase_07_prior_case_retrieval/CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb` | Formatted retrieval pipeline headings; preserved 237 test query ranking logs and submission packaging code | No | Yes | **PASS (All 7 code cells compile cleanly)** |

---

## 3. Structural Additions

1. **Modular `src/`:** Extracted genuine model definitions (`model.py`) and common utilities (`tokenizer.py`, `dataset.py`, `metrics.py`) from verified winning runs.
2. **Standardized `configs/`:** Placed exact `config.json` files for each phase.
3. **Reproducibility Suite:** Created `reproducibility/` directory with explicit instructions for Kaggle, local workstation, hardware constraints, and an automated execution matrix.
4. **Unit Test Suite:** Implemented `tests/test_model_shapes.py`, `tests/test_data_loading.py`, and `tests/test_preprocessing.py`.
