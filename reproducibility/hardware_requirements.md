# Hardware Requirements & Compute Budget

**Project:** In-House Legal Intelligence System (CS781 Capstone Project)

---

## Hardware Specifications

| Component | Minimum Specification | Recommended Specification |
|:---|:---|:---|
| **CPU** | 8 cores, x86_64 | 16+ cores (Intel Core i7/i9 or AMD Ryzen 7/9) |
| **RAM** | 16 GB | 32 GB or 64 GB DDR4/DDR5 |
| **GPU** | NVIDIA GPU with $\ge 8$ GB VRAM | NVIDIA RTX 3090 / 4090 / A100 or Kaggle Tesla P100 |
| **CUDA** | CUDA 11.8+ | CUDA 12.1+ |
| **Storage** | 50 GB free SSD space | 150 GB NVMe SSD |

---

## Compute Budget per Phase

- **Phase 1 (Pretraining):** ~3.5 hours on NVIDIA P100 for 100M tokens ($B=16, T=384$, FP16 mixed precision).
- **Phase 2 (L-NER):** ~45 minutes on NVIDIA T4 for 6 epochs of token classification.
- **Phase 3 (PEFT):** ~30 minutes for LoRA/LoRMA adapter training on IMDb benchmark.
- **Phase 4 (CJPE):** ~40 minutes on NVIDIA T4 for 6 epochs of hierarchical document modeling.
- **Phase 5 (LSI):** ~50 minutes for multi-label statute classification across 100 penal sections.
- **Phase 6 (Summarization):** ~1.2 hours for student generation and MBR consensus matrix calculation.
- **Phase 7 (PCR):** ~25 minutes for lexical BM25 candidate scoring and neural cross-encoder reranking.
