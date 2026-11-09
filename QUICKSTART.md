# Quickstart Guide

This guide describes how to set up the environment, run unit tests, and execute research notebooks both locally and on Kaggle.

---

## Option A: Kaggle Execution (Recommended)

1. Open Kaggle and click **New Notebook**.
2. Under **Notebook Options** (right sidebar):
   - Set **Accelerator** to `GPU P100` or `GPU T4 x2`.
   - Set **Internet** to `On` (required for initial library installation).
3. Click **File -> Import Notebook** and select any notebook from `notebooks/`.
4. Run all cells sequentially. Output predictions will automatically be exported to `/kaggle/working/` ready for download.

---

## Option B: Local Workstation Setup

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone <repository_url>
cd in-house-legal-intelligence

# Create virtual environment
conda create -n cs781 python=3.11 -y
conda activate cs781

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Verification Tests
```bash
# Verify model architectures and tensor shapes
python -m unittest tests/test_model_shapes.py

# Verify tokenization and preprocessing pipelines
python -m unittest tests/test_preprocessing.py

# Verify dataset loaders
python -m unittest tests/test_data_loading.py
```

### 3. Run Standalone Training / Inference Scripts
```bash
# Example: Phase 1 Language Model Pretraining
python scripts/phase_01/train.py --config configs/phase_01/config.json

# Example: Phase 7 Prior Case Reranking
python scripts/phase_07/infer.py --config configs/phase_07/config.json
```
