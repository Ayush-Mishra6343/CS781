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

The following specific page issues identified across editorial rounds were methodically resolved in the LaTeX sources:

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
   - Standardized all mathematical notation (LaTeX math mode for all Greek variables, loss functions, and epsilon terms).
3. **Connected Technical Explanations:**
   - Every phase narrative now explains the linear chain of inquiry: hypothesis $\to$ observed training behavior $\to$ failure mode diagnosis $\to$ architectural adjustment $\to$ empirical evaluation.

---

## 4. Factual Alignments & Metric Reconciliation

All values reported in the main report and appendix were rigorously audited against primary artifacts on disk:

1. **Phase 1 (Pretraining & Continuations):**
   - Official Codabench metric identified as Token Prediction Accuracy ($0.8308$ across 1,000 prompt continuations).
   - Local validation cross-entropy loss: $3.0722$; validation perplexity: $21.589$ across 5M tokens.
   - RMSNorm epsilon: FP16 mixed precision ($\epsilon = 10^{-6}$) vs. FP32 prototype ($\epsilon = 10^{-5}$).
2. **Phase 2 (Named Entity Recognition):**
   - Local validation strict span Macro-F1: $0.5140$ (relaxed span F1 $0.7226$ at calibrated $0.42$ threshold).
   - Official Codabench test submission score: $0.3223$ (exact official evaluation).
3. **Phase 3 (Legal Rhetorical Roles):**
   - Multi-class classification across 13 Indian Supreme Court rhetorical roles: dev Macro-F1 $0.4337$ (PEFT LoRA + BitFit).
   - Unrelated CS781 Assignment 4 IMDb sentiment exploration preserved in historical records but cleanly separated from legal capstone metrics.
4. **Phase 4 (Court Judgment Prediction & Dataset Counts):**
   - Resolved dataset judgment counts: strictly **994 balanced development judgments** (497 Class 0, 497 Class 1 as verified in `development_metrics.json`) and **1,517 blind test judgments**, eliminating the historical 1,988 duplicate count across main report and appendix.
   - Local validation: $0.5838$ (baseline) and $0.6779$ (standalone SwiGLU Stage 4 LoRA).
   - Official Codabench test submission score: $0.6886$ (7-component super-ensemble).
5. **Phase 5 (Statute Identification & Document Counts):**
   - Abstract updated: clearly identifies $0.3700$ as the official winning test Macro-F1 and $0.3097$ as the earlier single-model baseline.
   - Dataset counts clarified with primary evidence: **42,750 training documents**, **10,698 raw development documents** (yielding **10,181 evaluated judgments** in `dev_eval_detailed.json` after filtering 517 documents lacking target statute annotations), and **13,019 blind test documents** (matching the 13,019 predictions in `prediction.json`).
   - All historical wording calling the $0.3097$ baseline "competition-leading" corrected across narratives and decision registers.
   - Local development validation: $0.3700$ Macro-F1 on `dev_labels.json`.
   - Single-model baseline official submission (`RUN-00`, V1 calibrated checkpoint): $0.3097$.
   - Selected winning Dense-Sparse Super Apex Ensemble (Codabench ID 1016342): $0.3700$.
6. **Phase 6 (Judgment Summarization):**
   - Candidate pool size: $K=16$ diverse MBR candidates (scaled up from $K=8$ pilot trials).
   - Evaluation metric: RoBERTa-Large Layer 17 BERTScore consensus (dev BERTScore $0.8335$ vs. official test Codabench score $0.8363$).
7. **Phase 7 (Prior Case Retrieval & Oracle Values):**
   - Verified strict mathematical consistency of Oracle values across the entire document:
     * Max achievable true positives: $\sum_{q=1}^{118} \min(N_{\text{gold}}^{(q)}, 7) = 556$.
     * Total gold citation pairs: $\sum N_{\text{gold}}^{(q)} = 973$.
     * Total prediction slots: $118 \times 7 = 826$.
     * Unavoidable forced false positives ($N_{\text{gold}} < 7$ across 70 queries): $826 - 556 = 270$.
     * Unavoidable false negatives ($N_{\text{gold}} > 7$ across 39 queries): $973 - 556 = 417$.
     * Exact Oracle Micro-Precision@7: $556 / 826 = 0.6731$.
     * Exact Oracle Micro-Recall@7: $556 / 973 = 0.5714$.
     * Theoretical Oracle Micro-F1@7 ceiling: $\frac{2 \times 556}{826 + 973} = \frac{1112}{1799} = \mathbf{0.6181}$.
     * Champion E6 pipeline dev score: $0.4125$, capturing $\mathbf{66.73\%}$ of the theoretical ceiling ($\frac{371}{556}$). Official test score: $0.3739$.
8. **LLM Declaration & Student Contribution:**
   - Accurately discloses ChatGPT-4/4o (OpenAI) and Antigravity Assistant (Google Gemini 3.8) assistance per course policy.
   - All student entry placeholders (`[Student Entry]`) and unverified rank assumptions removed.

---

## 5. Course Rules & Page Ceiling Compliance

- **Sections 1--8 (Main Report):** Exactly **10 pages** (Pages 1--10), strictly satisfying the course limit of $\le 10$ pages.
- **Section 9 (LLM Usage Declaration):** Exactly **1 page** (Page 11), complete with declaration questionnaire, prompts, models, and outputs.
- **Student Personal Contribution Statement & References:** Exactly **2 pages** (Pages 12--13).
- **Master Technical Appendix:** Exactly **45 pages** (Pages 14--58), containing complete phase-by-phase implementation chapters, authentic loss curves, ablation tables, decision narratives, and engineering decision registers.
- **Total Combined Document:** Exactly **58 pages**.
