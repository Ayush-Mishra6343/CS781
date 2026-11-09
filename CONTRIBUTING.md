# Contributing to In-House Legal Intelligence System

Thank you for your interest in contributing to the **In-House Legal Intelligence System** codebase. This repository contains the complete capstone implementation developed at the Indian Institute of Technology Kanpur (CS781 Capstone Project).

---

## 1. Academic Context & Governance

- **Author:** Ayush Mishra (Roll Number: 250253)
- **Institution:** Department of Computer Science & Engineering, IIT Kanpur
- **Project Scope:** End-to-end Legal NLP across 7 specialized competition phases.
- **Submitted Academic Report:** The formal 58-page final capstone report has been officially evaluated and submitted. The report is permanent and immutable. All code updates must maintain exact fidelity with reported benchmark methodologies.

---

## 2. Repository Architecture & Layout

Contributions must align with the standardized repository layout:
```
├── configs/            # JSON configurations per phase
├── docs/               # Architecture, phase guides, reproducibility documentation
├── notebooks/          # Standardized 12-section Kaggle-runnable Jupyter Notebooks
├── report/             # Submitted final academic report (FROZEN)
├── results/            # Official benchmark result logs and predictions
├── scripts/            # Functional CLI entrypoints (train.py, infer.py, evaluate.py)
├── src/                # Modular Python packages (phase_01 through phase_07, common)
├── submissions/        # Complete 47-run submission archive with manifests
└── tests/              # Automated unit test suite
```

---

## 3. Development Guidelines

### A. Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Ayush-Mishra6343/CS781.git
   cd CS781
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### B. Notebook Standards
All primary notebooks in `notebooks/` must adhere to the 12-section standardized template:
1. Section 1: Environment Setup & Hardware Diagnostics
2. Section 2: Reproducibility & Seed Configuration
3. Section 3: Dataset Loading & Verification
4. Section 4: Domain Tokenization & Preprocessing
5. Section 5: Model Architecture & Parameter Count
6. Section 6: Optimizer, Scheduler & Training Dynamics
7. Section 7: Training Loop with Validation Checkpoints
8. Section 8: Formal Validation & Metric Evaluation
9. Section 9: Error Analysis & Per-Class Diagnostics
10. Section 10: Codabench / Submission Inference Pipeline
11. Section 11: Submission Package Verification
12. Section 12: Provenance & Execution Summary

- Notebooks must be fully self-contained.
- Do not commit large binary cell outputs (e.g., hundreds of megabytes of raw base64 or progress bars).
- Keep path references portable: use `os.environ.get("CS781_DATA_DIR", "./data")` rather than absolute local filesystem paths.

### C. Testing & Verification
Before submitting a pull request, ensure all tests pass:
```bash
python -m unittest discover tests/
```
Verify that all Jupyter notebooks parse as valid JSON:
```bash
python -c "import json, glob; [json.load(open(f, encoding='utf-8')) for f in glob.glob('notebooks/**/*.ipynb', recursive=True)]; print('All notebooks valid JSON!')"
```

### D. File Size & Checkpoint Hygiene
- GitHub has a strict 100 MB hard limit and 50 MB warning threshold.
- No binary weights (`.pt`, `.bin`, `.pth`) larger than 25 MB should be committed directly to Git.
- Large backbones are tracked via cryptographic SHA-256 hashes in `checkpoints/CHECKPOINT_MANIFEST.csv`.

---

## 4. Citation

If you use this codebase or system architecture in your research, please cite:
```bibtex
@misc{mishra2026legalintelligence,
  author       = {Ayush Mishra},
  title        = {In-House Legal Intelligence System: Domain Pretraining, Structured Extraction, Hierarchical Decision Modeling, and Hybrid Precedent Retrieval for Indian Jurisprudence},
  institution  = {Indian Institute of Technology Kanpur},
  year         = {2026},
  howpublished = {\url{https://github.com/Ayush-Mishra6343/CS781}}
}
```
