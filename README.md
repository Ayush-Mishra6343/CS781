# Legal NLP for Indian Court Judgments

### CS781 Capstone Project | IIT Kanpur
**Author:** Ayush Mishra  
**Course:** CS781: Deep Learning for Natural Language Processing  
**Institution:** Department of Computer Science and Engineering, IIT Kanpur  
**Report:** [`report/FINAL_REPORT.pdf`](report/FINAL_REPORT.pdf)

---

## A. Project Overview

This repository contains the coursework and experimental code developed for the CS781 capstone project at IIT Kanpur. The project explores natural language processing techniques applied to Indian legal texts, specifically judicial decisions from the Supreme Court of India and various High Courts. 

The project spans seven distinct legal NLP tasks: domain-specific language modelling, named entity recognition, rhetorical role classification, court judgment prediction, legal statute identification, judgment summarization, and prior case retrieval. Across these phases, experiments were conducted to evaluate different model architectures, parameter-efficient fine-tuning (PEFT) methods, hierarchical long-document modeling strategies, and retrieval formulations. 

The work documents both effective techniques and approaches that failed to improve upon baseline results. This codebase represents an academic research project developed across weekly competition deadlines rather than a commercial product or a fully integrated operational pipeline. Model weights exceeding GitHub size limits are archived externally, and primary experiments were developed to run on Kaggle GPU accelerators.

---

## B. Project Tasks

| Phase | Task | Main Focus |
|:---:|:---|:---|
| 1 | Legal Language Modelling | Domain-specific causal language modelling on Indian court judgments |
| 2 | Legal Named Entity Recognition | Strict character-span named entity extraction across 14 legal classes |
| 3 | Rhetorical Role Classification | Discourse classification of sentences into 7 legal rhetorical roles |
| 4 | Court Judgment Prediction | Binary outcome prediction and rationale extraction for long judgments |
| 5 | Legal Statute Identification | Multi-label prediction of applicable penal statutes from facts |
| 6 | Judgment Summarization | Abstractive legal document summarization with consensus selection |
| 7 | Prior Case Retrieval | Case-to-case precedent retrieval using lexical search and rank fusion |

---

## C. Results

The table below summarizes the selected official submission results verified on the Codabench competition platform across all seven phases:

| Phase | Task | Selected Submission ID | Official Score | Official Metric Name |
|:---:|:---|:---:|:---:|:---|
| 1 | Legal Language Modelling | 959798 | 0.8308 | Token Prediction Accuracy |
| 2 | Legal Named Entity Recognition | 974876 | 0.3223 | Strict Micro-F1 |
| 3 | Rhetorical Role Classification | 992721 | 0.4337 | Macro-F1 |
| 4 | Court Judgment Prediction | 1009439 | 0.6886 | Macro-F1 |
| 5 | Legal Statute Identification | 1016342 | 0.3700 | Macro-F1 |
| 6 | Judgment Summarization | 1029981 | 0.8363 | Composite BERTScore / ROUGE-L F1 |
| 7 | Prior Case Retrieval | 1039640 | 0.3739 | Micro-F1@7 |

*Note: Complete audit trails for all 47 competition submissions across phases are catalogued in [`submissions/submission_registry.csv`](submissions/submission_registry.csv) and [`results/official_submissions.csv`](results/official_submissions.csv). Local development scores and evaluation notes are documented in [`results/local_validation_results.csv`](results/local_validation_results.csv).*

---

## D. Methods Explored

The following methods and architectures were implemented and evaluated across the project:

- **Domain-Specific Language Modelling (Phase 1):** Pretrained a 94.76M parameter causal transformer (`HybridLegalLM`) from scratch on legal text using Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA, 12 query / 4 KV heads), and SwiGLU non-linearities with a custom 25,000 Byte-Level BPE tokenizer.
- **Legal Named Entity Recognition (Phase 2):** Fine-tuned RoBERTa on token-level BIO classification with character-span alignment calibration across 14 legal entity categories.
- **Parameter-Efficient Fine-Tuning (Phase 3):** Compared LoRA ($r=8$), Multiplicative LoRMA ($r=8$), and BitFit bias-tuning on legal discourse classification, evaluating adapter parameter efficiency and convergence speed.
- **Hierarchical Document Modelling (Phase 4):** Addressed 4,000+ word judgments using a two-tier architecture: 256-token local chunk encoders combined with a global document transformer to predict verdict outcomes.
- **Multi-Label Penal Code Classification (Phase 5):** Trained a hierarchical multi-label classifier over 100 Indian Penal Code (IPC) statutory provisions using Asymmetric Focal Loss and step-zero identity initialization.
- **Student Distillation & MBR Decoding (Phase 6):** Compressed a wide legal model into a 16-layer causal student model (`V1DeepLegalLM`), exploring Minimum Bayes Risk (MBR) consensus selection over candidate summaries using token-level similarity metrics.
- **Lexical Retrieval and Rank Fusion (Phase 7):** Implemented a vectorized BM25 retrieval engine with reciprocal rank fusion (RRF) and legal citation graph boosting to retrieve relevant prior judicial precedents.

---

## E. Repository Structure

```text
CS781/
│
├── README.md                           # Project overview, results, and reproduction
├── requirements.txt                    # Core Python dependencies
├── pyproject.toml                      # Package configuration and build metadata
│
├── report/
│   └── FINAL_REPORT.pdf                # Submitted academic capstone report (58 pages)
│
├── notebooks/                          # Primary and historical experiment notebooks
│   ├── phase_01_legal_language_model/
│   ├── phase_02_legal_ner/
│   ├── phase_03_rhetorical_role_classification/
│   ├── phase_04_judgment_prediction/
│   ├── phase_05_statute_identification/
│   ├── phase_06_judgment_summarization/
│   └── phase_07_prior_case_retrieval/
│
├── src/                                # Modular source library
│   ├── common/                         # Shared utilities, PEFT modules, metrics, packaging
│   └── phase_01/ ... phase_07/         # Phase-specific model definitions and data pipelines
│
├── configs/                            # Model and experiment hyperparameter JSON configs
│   └── phase_01/ ... phase_07/
│
├── scripts/                            # Standalone CLI entrypoints
│   └── phase_01/ ... phase_07/         # train.py, infer.py, evaluate.py
│
├── results/                            # Official submission and local validation records
│   ├── official_submissions.csv        # Selected Codabench submission scores
│   └── local_validation_results.csv    # Local development validation metrics
│
├── submissions/                        # Complete 47-run competition audit archive
│   ├── submission_registry.csv         # Full 47-run metadata registry
│   └── phase_01/ ... phase_07/         # Per-phase run manifests and best official submissions
│
├── checkpoints/                        # Model artifacts metadata
│   ├── CHECKPOINT_MANIFEST.csv         # Cryptographic SHA-256 catalog of model weights
│   └── bpe_en_25k.json                 # 25,000 Byte-Level BPE tokenizer vocabulary
│
├── docs/                               # Additional documentation
│   ├── methodology_and_limitations.md  # Detailed technical notes and experimental decisions
│   └── REPOSITORY_CLEANUP_SUMMARY.md   # Record of cleanup actions and bug corrections
│
└── tests/                              # Automated unit test suite
```

---

## F. Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Ayush-Mishra6343/CS781.git
cd CS781

# Create and activate a Python 3.11 virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Running Unit Tests
```bash
# Run the automated test suite
python -m unittest discover tests/
```

### 3. Running Notebooks (Recommended for GPU Training)
The primary notebooks in `notebooks/phase_XX/00_BEST_OFFICIAL_IMPLEMENTATION.ipynb` contain the end-to-end pipelines used during the project.
- **Kaggle Execution:** Import any notebook into Kaggle, enable a GPU accelerator (e.g., T4 or P100), and execute cells sequentially.
- **Data Paths:** Notebooks accept local paths via configuration variables or environment variables (`CS781_DATA_DIR`).

### 4. Running CLI Scripts
The standalone CLI scripts in `scripts/` provide entrypoints for model initialization, inference, and metric evaluation:
```bash
# Inspect model architecture and training setup (Phase 1)
python scripts/phase_01/train.py --config configs/phase_01/config.json

# Run evaluation on test predictions (requires local ground truth if available)
python scripts/phase_07/evaluate.py --prediction_file submissions/phase_07/best_official/prediction.json
```
*Note: Because competition test sets were evaluated privately on the Codabench platform, local evaluation scripts report that ground truth is private unless a local ground-truth file is supplied via `--ground_truth_file`.*

### 5. Inspecting Official Competition Submissions
Verified submission outputs and archive ZIPs for each phase are preserved under:
`submissions/phase_XX/best_official/`

---

## G. Limitations

- **Model Weights Storage:** Full pretrained and fine-tuned checkpoints (ranging from 80MB to 670MB each) exceed GitHub repository storage limits and are preserved in external local archives. The repository includes the complete model architectures, configurations, tokenizer files, and official test predictions.
- **Blind Evaluation:** The competition test sets for several phases were evaluated directly on the Codabench server without public ground-truth labels. Local evaluation scripts therefore cannot compute test metrics without ground-truth files.
- **Compute Constraints:** Standalone training scripts are designed as modular entrypoints; reproducing full pretraining or multi-epoch fine-tuning requires a dedicated CUDA-capable GPU (16GB+ VRAM recommended) and the full tokenized datasets.
- **Standalone CLI Scaffolds:** While Phases 1, 5, 6, and 7 include functional training and retrieval pipelines, several CLI training entrypoints (Phases 2, 3, 4) serve primarily as architecture and adapter initialization scripts; the full training runs were executed within the Kaggle notebook environment.

---

## H. Report

The complete academic report describing the methodology, design choices, ablation studies, and qualitative analyses across all seven phases is available at:

- **Canonical Path:** [`report/FINAL_REPORT.pdf`](report/FINAL_REPORT.pdf)

---

### License & Course Attribution
This repository represents student coursework submitted for CS781 at the Indian Institute of Technology Kanpur. Code and documentation are provided for academic research and evaluation purposes.
