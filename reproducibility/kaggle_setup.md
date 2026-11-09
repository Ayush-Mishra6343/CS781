# Kaggle Setup Guide

This guide explains how to configure Kaggle for reproducing the research notebooks.

---

## 1. Kaggle Environment Overview
Kaggle Notebooks provide:
- Free access to NVIDIA Tesla P100 (16 GB VRAM) or Dual T4 GPUs (16 GB VRAM each).
- 30 GB disk space under `/kaggle/working/`.
- Pre-installed PyTorch, Transformers, and scientific libraries.

---

## 2. Recommended Settings per Phase

| Phase | Recommended Accelerator | Internet Access | Persistence | Attached Input Datasets |
|:---:|:---:|:---:|:---:|:---|
| **P1** | GPU P100 | On | Files only | `cs781-legal-pretrain-corpus`, `bpe-25k` |
| **P2** | GPU T4 x2 | On | Files only | `cs781-lner-data` |
| **P3** | GPU T4 x2 | Optional | Files only | `cs781-peft-artifacts` |
| **P4** | GPU T4 x2 | Optional | Files only | `cs781-cjpe-data`, `swiglu-checkpoints` |
| **P5** | GPU T4 x2 | Optional | Files only | `cs781-lsi-statutes` |
| **P6** | GPU P100 | Optional | Files only | `cs781-summarization-corpus` |
| **P7** | GPU T4 x2 | Optional | Files only | `cs781-pcr-queries`, `pcr-documents` |
