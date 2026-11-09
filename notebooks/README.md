# Seven Kaggle-Ready Research Notebooks

This directory contains the primary research notebooks corresponding to each phase of the In-House Legal Intelligence System.

---

## Directory Index

| Phase Directory | Notebook File | Focus Architecture & Milestone |
|:---|:---|:---|
| `phase_01_legal_language_model/` | `CS781_Phase1_HybridLegalLM_Pretraining.ipynb` | HybridLocalGlobal Transformer (94.76M params, GQA 12/4, Local 256/Global sliding window, BPE 25k) |
| `phase_02_legal_ner/` | `CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb` | RoBERTa-Legal + Strict Character Boundary Alignment & Span Threshold Calibration |
| `phase_03_rhetorical_role_classification/` | `CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb` | PEFT Benchmarks: LoRA, LoRMA, BitFit, NF4 Quantization, Knowledge Distillation |
| `phase_04_judgment_prediction/` | `CS781_Phase4_CJPE_SwiGLU_LoRMA.ipynb` | Hierarchical Local-Global SwiGLU-LoRMA Document Classifier (Codabench Score: 0.6886) |
| `phase_05_statute_identification/` | `CS781_Phase5_LSI_Hierarchical_LoRMA.ipynb` | Multi-Label Legal Statute Identification with Step-Zero Identity Verification (Score: 0.3700) |
| `phase_06_judgment_summarization/` | `CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb` | Deep-16 Distilled Student + Minimum Bayes Risk (MBR) Consensus Decoding (Score: 0.8363) |
| `phase_07_prior_case_retrieval/` | `CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb` | Multi-Stage Lexical BM25 + Dense Cross-Encoder Neural Reranker (Score: 0.3739) |

---

## Key Principles
1. **Preserved Authentic Outputs:** Every notebook retains its real historical execution outputs, loss logs, and evaluation metrics.
2. **Kaggle Optimized:** Configured for seamless execution on Kaggle GPU P100 / T4 runtimes.
3. **No Hardcoded Secrets:** Abstracted to standard environment variables.
