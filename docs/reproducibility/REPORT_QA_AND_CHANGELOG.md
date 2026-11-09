# CS781 Capstone Project: Report QA and Editorial Changelog

**Project:** In-House Legal Intelligence System  
**Student:** Ayush Mishra (Roll Number: 250253)  
**Institution:** Department of Computer Science & Engineering, IIT Kanpur  
**Target Document:** `FINAL_REPORT.pdf` / `combined_report_and_appendix.pdf` (58 Pages)  
**Compilation Status:** Tectonic 0.17.0 (0 Errors, 0 Overfull `\hbox` Warnings)

---

## 1. Executive Summary of Revisions

This document records the editorial revisions, layout corrections, factual reconciliations, and academic writing polish applied to the master research report and comprehensive technical appendix for the CS781 Capstone Project.

The document preserves all genuine experimental findings, recovered architectures, authentic training curves, official Codabench submissions, and ablation registers across all seven project phases, while achieving strict compliance with course formatting rules and page limits.

---

## 2. Specific Formatting & Layout Corrections

The following specific page issues identified in earlier drafts were methodically resolved in the LaTeX sources:

| Original Issue Location | Problem Description | Root Cause | Editorial & LaTeX Fix Applied | Verification Status |
|:---|:---|:---|:---|:---:|
| **Page 11 (Section 9)** | LLM Usage Declaration text overflowing right margin ($x_1 > 540.0$ pt) | Unwrapped verbatim prompt strings and questionnaire response items in fixed tabular columns | Replaced manual tables with structured itemize environments using explicit `\sloppy` and constrained paragraph bounds; right boundary verified at $x_1 \le 540.0$ pt | **Verified (Bounds: [72.0, 73.0, 540.0, 752.0])** |
| **Page 23 (Phase 1)** | Phase 1 hyperparameter and architecture table extending beyond the page | Unconstrained multi-column width in standard `tabular` environment | Migrated to `tabularx{\textwidth}{lX}` with tuned inter-column padding (`\tabcolsep=4pt`) and concise parameter grouping | **Verified (Zero overflow, clean margin)** |
| **Page 27 (Phase 2)** | Phase 2 performance tables with clipped columns | Excessive number of entity metrics (12 categories) forced into a single wide horizontal row | Refactored into a structured two-column layout with grouped entity clusters (Macro-F1, Micro-F1, Strict vs. Relaxed Span F1) | **Verified (Zero column clipping)** |
| **Page 32 (Phase 3)** | Phase 3 class-wise rhetorical role results table clipped | 13 rhetorical role labels with precision, recall, and support overflowing horizontal text width | Standardized onto `tabularx` with wrapped header labels and explicit column width allocations ($L$, $C$, $R$) | **Verified (Zero horizontal truncation)** |
| **Page 54--55 (Appendix)** | Master engineering decision register cramped with awkward line wrapping | Too many decision columns packed into a single-page float | Implemented a clean multi-page `longtable` spanning pages 54 and 55 with repeating table headers (`\endhead`) and consistent column proportioning | **Verified (Continuous running table)** |
| **Page 58 (Appendix)** | Poor final-page break with excessive blank space and orphan heading | Subsections ending prematurely after Page 57, leaving 85% whitespace on Page 58 | Balanced narrative distribution across Section 9.3 (Cross-Phase Lineage & Capstone Synthesis); Page 58 now contains 1,877 characters of substantive conclusions | **Verified (Balanced page filling, no orphans)** |

---

## 3. Academic Prose Polish & Tone Refinement

In accordance with academic standards:
1. **Dramatic and Promotional Expressions Removed:**
   - Replaced terms like *"catastrophic failure"* with *"severe optimization divergence due to gradient instability"* or *"representation collapse under unweighted cross-entropy"*.
   - Replaced *"absolute theoretical ceiling"* with *"empirical upper bound under oracle candidate pooling"*.
   - Replaced *"definitive milestone"* with *"empirically verified baseline"*.
2. **Punctuation and Typography Cleanups:**
   - Removed unnecessary em dashes (`---`) and en dashes (`--`) in flowing prose, replacing them with clear subordinate clauses, semicolons, or standard academic sentence structures.
   - Standardized all mathematical notation (KaTeX / LaTeX math mode for all Greek variables, loss functions, and epsilon terms).
3. **Connected Technical Explanations:**
   - Every phase narrative now explains the linear chain of inquiry: hypothesis $\to$ observed training behavior $\to$ failure mode diagnosis $\to$ architectural adjustment $\to$ empirical evaluation.

---

## 4. Factual Alignments & Metric Reconciliation

All values reported in the main report and appendix were rigorously audited against primary artifacts on disk:

1. **Phase 1 (RMSNorm Epsilon):** Clarified the transition from FP32 prototype ($\epsilon = 10^{-5}$) to FP16 mixed precision ($\epsilon = 10^{-6}$) in `v4.ipynb` to prevent variance underflow.
2. **Phase 2 (Span F1):** Clarified the distinction between local development calibrated strict F1 ($0.7226$ at threshold $0.42$ with character boundary adjustment) and uncalibrated official Codabench server submission score ($0.3223$).
3. **Phase 3 (Multi-Task):** Clarified that local $0.8872$ represents IMDb validation accuracy during PEFT ablation, while official score $0.4337$ represents the multi-class legal challenge benchmark.
4. **Phase 4 (Sample Counts & Scores):**
   - Corrected dataset arithmetic in the discrepancy register: the authoritative competition benchmark evaluates on **994 balanced development judgments** (497 Class 0 and 497 Class 1 as confirmed in `development_metrics.json`) and **1,517 blind test judgments** (`candidate_manifest.json`), with 37,387 sliding-window tokens processed.
   - Clarified that the standalone SwiGLU model achieved $0.6779$ Macro-F1, while the 7-component super-ensemble achieved $0.6886$ on the official Codabench portal.
5. **Phase 5 (Statute Identification):** Verified that local Macro-F1 ($0.3700$) matches the official Codabench score ($0.3700$) exactly, reflecting reproducible step-zero state.
6. **Phase 6 (Summarization Metrics):** Clarified that local ROUGE-L is $0.4285$, whereas Codabench portal score is $0.8363$ (composite metric weighting semantic BERTScore, ROUGE consensus, and length regularity).
7. **Phase 7 (PCR Retrieval):** Clarified local dev MRR@10 ($0.4128$ on 50 dev queries) vs. official test MRR@10 ($0.3739$ on 237 test queries), and explained the oracle upper bounds ($0.784$ top-100 union vs. $0.542$ top-10 cutoff).

---

## 5. Course Rules & Page Ceiling Compliance

- **Sections 1--8 (Main Report):** Exactly **10 pages** (Pages 1--10), strictly satisfying the course limit of $\le 10$ pages.
- **Section 9 (LLM Usage Declaration):** Exactly **1 page** (Page 11), complete with declaration questionnaire, prompts, models, and outputs.
- **Student Personal Contribution Statement & References:** Exactly **2 pages** (Pages 12--13).
- **Master Technical Appendix:** Exactly **45 pages** (Pages 14--58), containing complete phase-by-phase implementation chapters, authentic loss curves, ablation tables, decision narratives, and engineering decision registers.
- **Total Combined Document:** Exactly **58 pages**.
