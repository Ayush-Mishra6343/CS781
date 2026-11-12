# Model Checkpoints Manifest

**Project:** Legal NLP for Indian Court Judgments (CS781 Capstone Project, IIT Kanpur)

---

## Checkpoint Manifest & Storage Overview

Pretrained model checkpoints (ranging from 80MB to 670MB each) are archived in external storage to maintain repository health and comply with GitHub file size limits. 

The accompanying [`CHECKPOINT_MANIFEST.csv`](CHECKPOINT_MANIFEST.csv) catalogs all 129 model checkpoints and tokenizer assets with:
- **`checkpoint_name`**: Model filename
- **`phase`**: Project phase (1 to 7)
- **`file_size_mb`**: Verified file size in megabytes
- **`storage_status`**: Indicates whether the file is `Included in Repository` (e.g. `bpe_en_25k.json`) or `External Archive (Local Workstation)`
- **`archive_location`**: Sanitized archival path reference
- **`sha256_checksum`**: Authentically calculated 64-character SHA-256 cryptographic hash

---

## Core Model Summary

| Phase | Model Identifier | Parameter Count | Architecture Summary |
|:---:|:---|:---:|:---|
| **P1** | `HybridLegalLM` | 94,764,288 | 12 layers, 768 dim, GQA (12 Q / 4 KV heads), sliding local 256 + global layers |
| **P2** | `RoBERTa-Legal-NER` | 124,655,629 | RoBERTa backbone with 14-class BIO classification head and span boundary calibration |
| **P3** | `PEFT-LoRMA-Student` | 81,912,576 | DistilGPT-2 Student with Multiplicative LoRMA adapters ($r=8$) |
| **P4** | `SwiGLU-LoRMA-CJPE` | 42,118,146 | Hierarchical chunk SwiGLU encoders with global document transformer |
| **P5** | `Hierarchical-LSI` | 44,228,836 | 100-class multi-label statutory classifier with step-zero identity initialization |
| **P6** | `Deep16-KD-Summarizer` | 86,412,288 | 16-layer causal transformer student model with MBR consensus decoding |
| **P7** | `BM25-RRF-PCR` | N/A (Lexical) | Vectorized BM25 indexing engine with reciprocal rank fusion |
