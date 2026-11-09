# Kaggle Execution and Portability Walkthrough

This guide provides step-by-step instructions for reproducing any of the seven capstone phases on Kaggle's free GPU infrastructure.

---

## 1. Hardware & Environment Recommendations

| Phase | Task | Recommended Accelerator | RAM | Estimated Runtime |
|:---:|:---|:---:|:---:|:---:|
| **Phase 1** | Legal LM Pre-Training | GPU T4 x2 or P100 | 16 GB | ~45 min (demo slice) |
| **Phase 2** | Legal NER | GPU T4 | 16 GB | ~25 min |
| **Phase 3** | Rhetorical Roles | GPU T4 | 16 GB | ~20 min |
| **Phase 4** | Judgment Prediction | GPU T4 or P100 | 16 GB | ~35 min |
| **Phase 5** | Statute Identification | GPU T4 | 16 GB | ~30 min |
| **Phase 6** | Summarization | GPU T4 | 16 GB | ~25 min |
| **Phase 7** | Prior Case Retrieval | GPU T4 or CPU | 16 GB | ~15 min |

---

## 2. Step-by-Step Kaggle Setup

### Step 1: Create a New Kaggle Notebook
1. Go to kaggle.com/code and click **New Notebook**.
2. Under **Notebook Options** (right side pane):
   - **Accelerator:** Select `GPU T4` (or `GPU P100`).
   - **Internet:** Toggle to `On` (required for downloading HuggingFace tokenizers).
   - **Persistence:** Set to `Files only` (recommended).

### Step 2: Upload or Link the Phase Notebook
- In Kaggle, click **File -> Import Notebook**.
- Select the `00_BEST_OFFICIAL_IMPLEMENTATION.ipynb` from the corresponding phase directory in this repository (e.g. `notebooks/phase_01_legal_language_model/00_BEST_OFFICIAL_IMPLEMENTATION.ipynb`).

### Step 3: Attach the Phase Dataset
1. In the right pane, click **+ Add Input**.
2. Search for the CS781 dataset corresponding to your phase or upload the raw JSON files from `data/`.
3. The intelligent file discovery function (`find_file()`) in Section 3 of every primary notebook automatically checks `/kaggle/input/` and locates your data files regardless of Kaggle dataset naming.

### Step 4: Run the 12 Standardized Sections
Execute cells sequentially from top to bottom:
- **Section 1 & 2:** Verifies GPU availability, installs any missing packages, and locks random seeds (`42`).
- **Section 3 & 4:** Loads and validates data splits, tokenizes text.
- **Section 5 to 7:** Builds the exact winning model architecture, configures optimizer, runs training.
- **Section 8 & 9:** Computes official validation metrics and per-class diagnostics.
- **Section 10 to 12:** Generates test predictions, packages the official Codabench submission ZIP, and outputs SHA-256 provenance hashes.
