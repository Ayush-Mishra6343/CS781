# Dataset Setup Guide

This document describes how to obtain and configure dataset splits for each phase.

---

## Accessing Datasets
All raw legal judgments derive from Indian judicial databases. For academic compliance and competition rules, raw training corpora are accessed via HuggingFace Hub or Kaggle Dataset attachments:

```bash
# Example: Ingesting dataset using HuggingFace
from datasets import load_dataset
dataset = load_dataset("ayush-mishra/cs781-legal-corpus", split="train")
```

For offline execution, place downloaded JSON splits under your local data directory and set:
```bash
export CS781_DATA_DIR="/path/to/data"
```
