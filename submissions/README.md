# Complete Submissions Archive: 47 Portal Submissions Across 7 Phases

**Student:** Ayush Mishra (Roll Number: 250253)  
**Institution:** Indian Institute of Technology Kanpur  
**Project:** In-House Legal Intelligence System (CS781 Capstone Project)  
**Submission Platform:** Codabench Competition Portal  

---

## 1. Executive Summary & Official Benchmark Results

This directory contains the exhaustive, cryptographically verified record of all **47 official submissions** across all seven project phases.

| Phase | Phase Name | Submissions | Baseline Score | Winning Score | Improvement Delta | Official Evaluation Metric |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **Phase 1** | [Phase 1: Legal Language Model Pre-Training](phase_01/) | 4 runs | 0.8212 | **0.8308** | **+0.0096 (+1.17%)** | Token Prediction Accuracy (Hidden Competition Continuation Benchmark) |
| **Phase 2** | [Phase 2: Legal Named Entity Recognition (L-NER)](phase_02/) | 9 runs | 0.1386 | **0.3223** | **+0.1837 (+132.54%)** | Strict Micro F1 (Exact Character Span & Type Match) |
| **Phase 3** | [Phase 3: Rhetorical Role (RR) Classification & PEFT Suite](phase_03/) | 9 runs | 0.3941 | **0.4337** | **+0.0396 (+10.05%)** | Macro F1 across 7 Rhetorical Roles (Facts, Issue, Argument, Statute, Precedent, Ratio, Ruling) |
| **Phase 4** | [Phase 4: Court Judgment Prediction and Explanation (CJPE)](phase_04/) | 6 runs | 0.5520 | **0.6886** | **+0.1366 (+24.75%)** | Macro F1 (Binary Outcome Prediction & Explanation Grounding) |
| **Phase 5** | [Phase 5: Legal Statute Identification (LSI)](phase_05/) | 6 runs | 0.2244 | **0.3700** | **+0.1456 (+64.88%)** | Macro F1 across 100 Indian Statutory Penal Sections (IPC) |
| **Phase 6** | [Phase 6: Legal Judgment Summarization](phase_06/) | 8 runs | 0.7590 | **0.8363** | **+0.0773 (+10.18%)** | Composite Official Benchmark Score (ROUGE-L / BERTScore F1 / MBR Consensus Quality) |
| **Phase 7** | [Phase 7: Prior Case Retrieval (PCR)](phase_07/) | 5 runs | 0.1175 | **0.3739** | **+0.2564 (+218.21%)** | Micro F1@7 across 237 Official Test Queries (Evaluating Top-7 Retrieved Documents) |
| **TOTAL** | **Entire System (7 Phases)** | **47 runs** | — | — | — | **Exhaustive Multi-Task Legal AI** |

---

## 2. Directory Structure

Each phase directory (`phase_01` to `phase_07`) adheres to a standardized dual-tier structure:
```
submissions/phase_0X/
├── README.md               # Phase overview, table of all runs, architectural evolution, hyperparameter tuning
├── best_official/          # Full code, configs, notebooks, and exact prediction artifacts of winning run
└── runs/                   # Granular record of every single submission submitted to Codabench
    ├── <submission_id>/    # Run-specific directory
    │   ├── README.md       # Technical explanation, hypothesis, results, why selected/superseded
    │   ├── manifest.json   # Full cryptographic and portal metadata
    │   ├── config.json     # Hyperparameter configuration
    │   ├── results.json    # Exact portal score and metric breakdown
    │   ├── notebook_ref.md # Link to producing notebook and cell execution evidence
    │   └── artifacts/      # Submission ZIP / JSON package or SHA-256 pointer manifest
```

---

## 3. Cryptographic and Provenance Verification

Every submission artifact in this repository is cross-referenced against the master audit registers and Codabench execution logs:
- Master Experiment Register: `submissions/submission_registry.csv`
- Checkpoint Manifest: `checkpoints/CHECKPOINT_MANIFEST.csv`
- No placeholders: All scores, dates, and parameter values correspond directly to real project executions.
