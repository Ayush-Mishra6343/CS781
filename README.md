# In-House Legal Intelligence System (CS781 Capstone Project)

**Student:** Ayush Mishra  
**Roll Number:** 250253  
**Institution:** Indian Institute of Technology Kanpur  
**Repository Version:** 1.0.0 (Preserved Public Release)

---

## 1. Project Overview

The **In-House Legal Intelligence System** is an end-to-end natural language processing research pipeline designed for the Indian legal domain. Developed over six weeks across seven interconnected research phases, the system addresses key legal AI challenges: pretraining a domain-adapted language model, extracting named entities with precise character boundaries, fine-tuning with parameter-efficient techniques, predicting judicial outcomes from multi-page judgments, identifying statutory penal codes, generating abstractive case summaries, and retrieving relevant prior precedents.

All models and experiments were developed, evaluated, and benchmarked against the official course portal (Codabench), adhering to authentic legal datasets from the Supreme Court of India and High Courts.

---

## 2. Seven-Phase System Architecture

| Phase | Task Description | Focal Architecture | Key Technical Contribution | Official Codabench Score |
|:---:|:---|:---|:---|:---:|
| **01** | Legal Language Modeling | `HybridLegalLM` (94.76M params) | 12 layers, 12 Q / 4 KV heads (GQA), sliding local 256 + global window attention, 25k Byte-Level BPE | **0.8308** |
| **02** | Legal Named Entity Recognition | `RoBERTa-Legal-NER` + Calibration | Strict character-offset alignment, boundary probability calibration, 13 entity categories | **0.3223** |
| **03** | Parameter-Efficient Fine-Tuning | LoRA, LoRMA, BitFit, NF4, DistilGPT-2 | Multiplicative low-rank adaptation (LoRMA $r=8$), 4-bit NormalFloat quantization, response-only supervision | **0.4337** |
| **04** | Court Judgment Prediction (CJPE) | Hierarchical SwiGLU-LoRMA | Chunk-level SwiGLU feed-forward networks + cross-chunk document transformer | **0.6886** |
| **05** | Legal Statute Identification (LSI) | Multi-Label Hierarchical SwiGLU-LoRMA | 100 Indian statutory penal sections, step-zero identity initialization, multi-label BCE loss | **0.3700** |
| **06** | Legal Document Summarization | Deep-16 Distillation + MBR Decoding | 16-layer compressed student model + Minimum Bayes Risk (MBR) consensus generation | **0.8363** |
| **07** | Prior Case Retrieval (PCR) | Multi-Stage Hybrid Reranker | BM25 lexical candidate retrieval + dense neural cross-encoder reranking with rank fusion | **0.3739** |

---

## 3. Repository Structure

```text
in-house-legal-intelligence/
├── README.md                            # Comprehensive project overview and architecture
├── QUICKSTART.md                        # Step-by-step local and Kaggle reproduction guide
├── requirements.txt                     # Global Python dependencies
├── .gitignore                           # Git ignore rules for checkpoints and large caches
├── .gitattributes                       # Large file tracking configuration
│
├── notebooks/                           # Seven verified, Kaggle-ready research notebooks
│   ├── README.md                        # Notebook execution guide and Kaggle environment instructions
│   ├── phase_01_legal_language_model/   # Phase 1: HybridLegalLM pretraining notebook
│   ├── phase_02_legal_ner/              # Phase 2: RoBERTa-Legal token classification & boundary calibration
│   ├── phase_03_rhetorical_role_classification/ # Phase 3: Comprehensive PEFT & distillation benchmark
│   ├── phase_04_judgment_prediction/    # Phase 4: Hierarchical SwiGLU-LoRMA CJPE
│   ├── phase_05_statute_identification/ # Phase 5: Multi-label statute classification
│   ├── phase_06_judgment_summarization/ # Phase 6: Deep-16 KD + MBR consensus decoding
│   └── phase_07_prior_case_retrieval/   # Phase 7: Multi-stage BM25 + dense cross-encoder reranker
│
├── src/                                 # Reusable model implementations and shared utilities
│   ├── common/                          # Shared tokenizers, dataset loaders, and evaluation metrics
│   └── phase_01/ ... phase_07/          # Authentic model definitions per phase (model.py)
│
├── configs/                             # Authentic model hyperparameters and configurations
│   └── phase_01/ ... phase_07/          # config.json files matching official submissions
│
├── scripts/                             # Training and evaluation entry points
│   └── phase_01/ ... phase_07/          # Runnable train.py and infer.py scripts
│
├── data/                                # Dataset acquisition, schemas, and attachment instructions
│   └── README.md
│
├── checkpoints/                         # Checkpoint manifests, parameter counts, and download links
│   └── README.md
│
├── results/                             # Official submission ZIPs and tabular results
│   ├── README.md
│   ├── official_submissions.csv         # Selected Codabench submission records
│   ├── local_validation_results.csv     # Local development validation metrics
│   └── phase_01/ ... phase_07/          # Preserved submission ZIPs and prediction files
│
├── docs/                                # Technical documentation, phase dossiers, and final reports
│   ├── methodology_and_limitations.md   # Detailed engineering decisions and known limitations
│   ├── architecture/                    # High-level architecture explanations
│   ├── phases/                          # Individual phase technical write-ups
│   └── report/                          # 58-page combined report and 13-page conference report
│
├── requirements/                        # Modular per-phase dependency files
│   ├── base.txt                         # Common runtime packages
│   └── phase_01.txt ... phase_07.txt    # Phase-specific dependencies
│
├── reproducibility/                     # Environment configuration and hardware specifications
│   ├── local_setup.md                   # Local Linux/macOS/Windows setup instructions
│   ├── kaggle_setup.md                  # Kaggle GPU runtime and dataset attachment instructions
│   ├── dataset_setup.md                 # HuggingFace Hub dataset access guide
│   ├── hardware_requirements.md         # GPU memory and compute budget requirements
│   └── execution_status.csv             # Automated execution readiness matrix
│
└── tests/                               # Verification test suite
    ├── README.md                        # Test execution instructions
    ├── test_data_loading.py             # Data loading and memory mapping tests
    ├── test_preprocessing.py            # Tokenization and character span alignment tests
    └── test_model_shapes.py             # Model instantiation and tensor shape verification tests
```

---

## 4. Phase-Wise Results & Verification

Official benchmark test scores from Codabench are separated from local development metrics:

| Phase | Submission ID | Official Test Score | Local Validation Metric | Status |
|:---:|:---:|:---:|:---|:---:|
| **P1** | `959798` | **0.8308** | Val Loss: 3.0722, Perplexity: 21.589 | Verified |
| **P2** | `974876` | **0.3223** | Strict Micro-F1: 0.3223 (Calibrated Boundary: 0.7226) | Verified |
| **P3** | `992721` | **0.4337** | Multi-Task Super-Ensemble Accuracy: 0.8872 | Verified |
| **P4** | `1009439` | **0.6886** | Standalone Macro-F1: 0.6779 / Super-Ensemble: 0.6886 | Verified |
| **P5** | `1016342` | **0.3700** | Multi-Label Macro-F1: 0.3700 | Verified |
| **P6** | `1029981` | **0.8363** | Standalone ROUGE-L: 0.4285 / MBR Consensus: 0.8363 | Verified |
| **P7** | `1039640` | **0.3739** | Local Dev MRR@10: 0.4128 | Verified |

*Full traceability records linking each submission to its generating code are preserved in `results/official_submissions.csv` and `results/local_validation_results.csv`.*

---

## 5. Getting Started (Kaggle-First Execution)

The recommended execution platform is **Kaggle Notebooks** with an attached GPU (NVIDIA P100 or 2x T4):

1. **Upload Notebook:** Navigate to `notebooks/phase_{XX}_.../` and upload the `.ipynb` file to Kaggle.
2. **Select Accelerator:** Enable `GPU P100` or `GPU T4 x2` under notebook settings.
3. **Attach Dataset:** Attach the corresponding legal dataset under `/kaggle/input/` as described in `reproducibility/kaggle_setup.md`.
4. **Execute Cells:** Run sequentially from cell 0. All notebooks include self-contained dependency checks and generate Codabench-compliant submission files in `/kaggle/working/`.

For local workstation reproduction, see [`QUICKSTART.md`](QUICKSTART.md) and [`reproducibility/local_setup.md`](reproducibility/local_setup.md).

---

## 6. Hardware & Compute Constraints

- **Pretraining (Phase 1):** ~11.5 GB VRAM, ~3.5 hours on NVIDIA P100 for 100M tokens.
- **Fine-Tuning (Phases 2, 4, 5):** ~8.0–10.0 GB VRAM, ~40–50 minutes on standard T4 GPU.
- **Inference & Reranking (Phases 6, 7):** ~7.0–12.0 GB VRAM, ~25–70 minutes.

---

## 7. Known Limitations & Research Notes

1. **Phase 2 Boundary Calibration:** While calibrated postprocessing achieves 0.7226 F1 locally on development annotations, official server submissions were scored under strict uncalibrated token alignment (0.3223).
2. **Long Document Truncation:** Judicial documents exceeding 4,096 tokens require hierarchical windowing (Phases 4, 5) or extractive chunking (Phase 6).
3. **Data Redistribution:** Proprietary dataset splits are subject to course benchmark rules and cannot be redistributed directly inside Git; users must download raw judgments through authorized repositories. Detailed setup scripts are provided in `data/README.md`.

---

## 8. Research Documentation

- Full combined research report and appendix: [`docs/report/combined_report_and_appendix.pdf`](docs/report/combined_report_and_appendix.pdf) (58 pages)
- Main conference-style report: [`docs/report/CS781-CP-Ayush-Mishra-250253-report.pdf`](docs/report/CS781-CP-Ayush-Mishra-250253-report.pdf) (13 pages, $\le 10$ pages body)
- Technical appendix: [`docs/report/appendix.pdf`](docs/report/appendix.pdf) (45 pages)
