# Phase 7: Prior Case Retrieval (PCR)

**Task Description:** Multi-stage lexical BM25 + dense neural cross-encoder reranking with reciprocal rank fusion.

- **Official Benchmark Score:** `Micro-F1@7: 0.3739 on 237 official test queries`
- **Primary Notebook:** [`00_BEST_OFFICIAL_IMPLEMENTATION.ipynb`](00_BEST_OFFICIAL_IMPLEMENTATION.ipynb) (Identical to [`CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb`](CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb))
- **Historical Experiments:** [`historical_experiments/`](historical_experiments/) (1 notebooks preserved)

---

## Kaggle Execution & Environment Setup

This notebook is fully self-contained and ready to execute on Kaggle or any Jupyter/CUDA environment.

1. **Hardware Accelerator:** `GPU T4 or CPU (fast vectorized BM25 + lightweight reranker)`
2. **Estimated Runtime:** `~15 minutes (BM25 search + RRF scoring on 237 test queries)`
3. **Dataset Attachment:**
   - On Kaggle: Click `+ Add Input` -> Search for `Prior Case Retrieval Query & Precedent Candidate Corpus` or upload the data archive from `data/`.
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

- [`CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb`](historical_experiments/CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb)
