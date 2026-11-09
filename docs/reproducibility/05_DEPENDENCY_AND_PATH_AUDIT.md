# Dependency and File Path Audit

**Project:** In-House Legal Intelligence System (CS781 Capstone Project)  
**Student:** Ayush Mishra (Roll Number: 250253)  
**Audit Standard:** Part 7B & 7C Specification (Path Portability & External Resource Requirements)

---

## 1. External Path Analysis & Portability Remediation

Historical research notebooks originally referenced local filesystem paths on developer workstations or default Kaggle scratch directories. To ensure seamless reproducibility across local Linux/Windows and Kaggle environments, all paths in the public GitHub-ready notebooks have been reviewed and standardized using a uniform fallback architecture:

```python
# Portable Path Configuration Pattern
import os

BASE_DATA_DIR = os.environ.get("CS781_DATA_DIR", "/kaggle/input/cs781-legal-dataset")
BASE_CHECKPOINT_DIR = os.environ.get("CS781_CHECKPOINT_DIR", "/kaggle/input/cs781-checkpoints")
BASE_OUTPUT_DIR = os.environ.get("CS781_OUTPUT_DIR", "/kaggle/working/output")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
```

---

## 2. Phase-by-Phase External Resource Audit

| Phase | External Dataset Required | External Tokenizer / Config | Required Checkpoints | Output Path Standardized |
|:---:|:---|:---|:---|:---|
| **P1** | Indian Legal Judgments Pretraining Corpus (~105M tokens raw text) | Byte-Level BPE 25k Vocabulary (`bpe_25k.json`) | None (Trained from scratch) | `/kaggle/working/cs781_phase1_hybrid/` |
| **P2** | L-NER Annotated Legal Corpus (Character spans) | RoBERTa Tokenizer + Custom Span Map | RoBERTa-Legal Pretrained Base | `/kaggle/working/cs781_phase2_lner/` |
| **P3** | Multi-Task Legal/IMDb Benchmark Partition | Hugging Face Hub / Restored A-3 Tokenizer | Restored A-3 Q5 Checkpoint | `/kaggle/working/cs781_phase3_peft/` |
| **P4** | CJPE Judgment Texts & Binary Labels (2,471 train / 618 val) | HuggingFace Hub Tokenizer | SwiGLU-LoRMA Base Checkpoint | `/kaggle/working/cs781_phase4_cjpe/` |
| **P5** | Legal Statute Identification Corpus (5,120 cases, 100 classes) | Pretrained Tokenizer | Identity-Initialized LoRMA Weights | `/kaggle/working/cs781_phase5_lsi/` |
| **P6** | Indian Supreme Court Judgment Summarization Corpus | Pretrained Encoder-Decoder Tokenizer | Teacher Summary Model / Distil-16 Student | `/kaggle/working/cs781_phase6_summarization/` |
| **P7** | Prior Case Retrieval Corpus (Case queries + Document candidates) | BM25 Inverted Index + Dense Embeddings | Cross-Encoder PCR Checkpoint | `/kaggle/working/cs781_phase7_pcr/` |

---

## 3. Dependency Compatibility Matrix

| Package | Minimum Tested Version | Recommended Version | Scope of Usage |
|:---|:---:|:---:|:---|
| `python` | `3.10` | `3.11.x` | Global runtime |
| `torch` | `2.1.0` | `2.2.0+cu121` | Core neural network computation, GQA attention, SwiGLU |
| `transformers` | `4.36.0` | `4.38.2` | Base models, tokenizers, learning rate schedulers |
| `datasets` | `2.16.0` | `2.18.0` | Streaming dataset ingestion and caching |
| `evaluate` | `0.4.1` | `0.4.1` | ROUGE-L, F1, exact match metrics |
| `scikit-learn` | `1.3.0` | `1.4.0` | Micro/Macro-F1, confusion matrices, classification reports |
| `numpy` | `1.24.0` | `1.26.4` | Memory-mapped token arrays, numerical arrays |
| `scipy` | `1.11.0` | `1.12.0` | Distance metrics, ranking metrics (MRR) |
| `rank_bm25` | `0.2.2` | `0.2.2` | Lexical BM25 candidate scoring (Phase 7) |
| `rouge_score` | `0.1.2` | `0.1.2` | Abstractive summarization evaluation (Phase 6) |
| `bert_score` | `0.3.13` | `0.3.13` | Semantic evaluation of generated summaries (Phase 6) |
