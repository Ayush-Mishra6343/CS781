# Best Official Implementation: Phase 7: Prior Case Retrieval (PCR)

- **Codabench Submission ID:** `1039640`
- **Official Benchmark Score:** `0.3739` (Micro F1@7 across 237 Official Test Queries (Evaluating Top-7 Retrieved Documents))
- **Primary Architecture:** Multi-Stage Lexical BM25 + Dense Cross-Encoder Neural Reranker with Reciprocal Rank Fusion & Score Normalization
- **Submission Date:** 2026-11-07 13:52

### Verification and Execution
This directory contains the exact, unmodified standalone code, configuration, and prediction artifacts that achieved the winning official score on the Codabench competition portal.

Files in this directory:
- `CS781_Phase7_PCR_Appendix_Audit_and_Analysis.ipynb`
- `CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb`
- `config.json`
- `deploy_ultimate_portal_solution.py`
- `fast_vectorized_bm25.py`
- `prediction.json`
- `submission.zip`
- `submission_E6_Hybrid_Reranker.zip`
- `train_lambdamart_reranker.py`
