# Phase 1: In-House Legal Language Model Pre-Training

**Task Description:** Pre-training autoregressive Legal LLM from scratch with custom BPE 16k vocabulary, GQA, and local/global sliding window attention.

- **Official Benchmark Score:** `Token Prediction Accuracy: 0.8308 (Perplexity: 21.589)`
- **Primary Notebook:** [`00_BEST_OFFICIAL_IMPLEMENTATION.ipynb`](00_BEST_OFFICIAL_IMPLEMENTATION.ipynb) (Identical to [`CS781_Phase1_HybridLegalLM_Pretraining.ipynb`](CS781_Phase1_HybridLegalLM_Pretraining.ipynb))
- **Historical Experiments:** [`historical_experiments/`](historical_experiments/) (7 notebooks preserved)

---

## Kaggle Execution & Environment Setup

This notebook is fully self-contained and ready to execute on Kaggle or any Jupyter/CUDA environment.

1. **Hardware Accelerator:** `GPU T4 x2 or P100 (16GB VRAM recommended)`
2. **Estimated Runtime:** `~45 minutes (pretraining demonstration slice) / 8 hours full corpus`
3. **Dataset Attachment:**
   - On Kaggle: Click `+ Add Input` -> Search for `Indian Legal Corpus (Supreme Court & High Court judgments)` or upload the data archive from `data/`.
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

- [`CS781_Assignment3_Final_Run.ipynb`](historical_experiments/CS781_Assignment3_Final_Run.ipynb)
- [`CS781_Assignment_3_FinalTrainReady.ipynb`](historical_experiments/CS781_Assignment_3_FinalTrainReady.ipynb)
- [`CS781_Phase1_A1_BPE_Tokenizer.ipynb`](historical_experiments/CS781_Phase1_A1_BPE_Tokenizer.ipynb)
- [`CS781_Phase1_Pretraining_V1_Decoder.ipynb`](historical_experiments/CS781_Phase1_Pretraining_V1_Decoder.ipynb)
- [`v1.ipynb`](historical_experiments/v1.ipynb)
- [`v2.ipynb`](historical_experiments/v2.ipynb)
- [`v4.ipynb`](historical_experiments/v4.ipynb)
