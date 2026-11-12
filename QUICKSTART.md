# Quickstart Guide

This guide describes how to set up the environment, run unit tests, and execute research notebooks and scripts.

---

## 1. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/Ayush-Mishra6343/CS781.git
cd CS781

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## 2. Running Unit Tests

```bash
# Run all unit tests
python -m unittest discover tests/
```

---

## 3. Running Notebooks on Kaggle

The primary experiment notebooks in `notebooks/` were designed to run on Kaggle GPU environments:
1. Open Kaggle and create a new notebook.
2. In the right sidebar, select a **GPU P100** or **GPU T4 x2** accelerator.
3. Import the relevant notebook from `notebooks/`.
4. Run cells sequentially to reproduce feature extraction, model evaluation, and submission packaging.

---

## 4. Standalone CLI Entrypoints

The scripts under `scripts/` provide entrypoints to initialize models, inspect architectures, and run evaluation:
```bash
# Model initialization and pretraining loop
python scripts/phase_01/train.py --config configs/phase_01/config.json

# Prior case retrieval candidate indexing
python scripts/phase_07/train.py --data_dir data/phase_07

# Prior case retrieval evaluation (requires local ground-truth qrels file)
python scripts/phase_07/evaluate.py --prediction_file submissions/phase_07/best_official/prediction.json
```
