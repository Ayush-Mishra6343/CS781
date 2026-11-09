# Phase 5: Legal Statute Identification (LSI): Submissions Archive

- **Official Winning Score:** **0.3700** (Submission ID: `1016342`)
- **Baseline Score:** **0.2244** (Submission ID: `1011978`)
- **Improvement Delta:** **+0.1456 (+64.88%)**
- **Evaluation Metric:** Macro F1 across 100 Indian Statutory Penal Sections (IPC)

## 1. Task Description

Multi-label statute identification predicting which of 100 Indian Penal Code sections apply to a given factual legal case narrative.

## 2. Chronological Submissions Table

| Submission ID | Date | Score | Status | Model / Technique | Key Insight / Learning |
|:---:|:---:|:---:|:---:|:---|:---|
| [`1011971`](runs/1011971/) | 2026-10-27 14:10 | **0.2356** | Superseded | TF-IDF N-Gram Matching Baseline (Word 1-3 grams + Char 3-5 grams) | Sublinear TF-IDF baseline. Handled exact statutory formula n-grams reasonably well, outperforming raw BM25. |
| [`1011978`](runs/1011978/) | 2026-10-27 16:30 | **0.2244** | Superseded | BM25 Lexical Statute Retrieval Baseline | Lexical baseline matching statutory text against case descriptions. Fast and transparent, but failed when judgments discussed factual circumstances without naming the formal section number. |
| [`1012531`](runs/1012531/) | 2026-10-27 22:15 | **0.3097** | Superseded | Standalone SwiGLU-LoRMA Single-Model Baseline (V3 LSI) | Early single-model milestone (0.3097). Highlighted the power of SwiGLU-LoRMA, but lacked step-zero stabilization and threshold tuning of winning 1016342. |
| [`1013809`](runs/1013809/) | 2026-10-28 09:40 | **0.2293** | Superseded | DeBERTa-v3 multi-label baseline with max-sequence truncation | DeBERTa relative position embeddings helped within the first 512 tokens, but whole-document context was missing. |
| [`1014073`](runs/1014073/) | 2026-10-28 14:15 | **0.3211** | Superseded | Legal-BERT multi-label classifier with dynamic positive loss weighting | Strong legal BERT baseline (0.3211), but limited to 512 tokens which truncated references to earlier procedural sections. |
| [`1016342`](runs/1016342/) | 2026-10-28 20:53 | **0.3700** | Selected (Winning) 🏆 (Winning) | Multi-Label Hierarchical SwiGLU-LoRMA with Step-Zero Identity Verification & Hybrid LSI | Official winning model for Phase 5. Successfully classified 100 Indian statutory penal codes from lengthy court records. Step-zero identity initialization prevented initial gradient explosion, reaching 0.3700 Macro F1. |

## 3. Architectural Evolution

```
Phase 5 Architectural Evolution:
1. Okapi BM25 Lexical Retrieval (1011978: 0.2244)
   - Pure keyword overlap failed when facts described offenses without citing section numbers
         │
         ▼
2. TF-IDF N-Gram Matching (1011971: 0.2356)
   - Handled exact formulaic phrases, but failed on paraphrased factual narratives
         │
         ▼
3. DeBERTa-v3 Truncated Baseline (1013809: 0.2293)
   - 512-token limit cut off crucial evidentiary discussions
         │
         ▼
4. Standalone SwiGLU-LoRMA Baseline (1012531: 0.3097)
   - First neural hierarchical model; demonstrated strong capability across frequent statutes
         │
         ▼
5. Legal-BERT Multi-Label with Positive Weighting (1014073: 0.3211)
   - Dynamic positive loss weighting boosted recall on moderately rare sections
         │
         ▼
6. Multi-Label SwiGLU-LoRMA with Step-Zero Identity & Hybrid Reranker (1016342: 0.3700) [WINNING]
   - Step-zero identity initialization stabilized early gradient backprop across 100 sections
   - Calibrated per-section decision thresholds delivered official winning score of 0.3700
```

## 4. Hyperparameter Tuning Narrative

Phase 5 tackled multi-label prediction across 100 distinct statutory sections under severe label sparsity. In earlier runs (1012531: 0.3097), random initialization of the 100-way classification projection caused severe initial gradient spikes that disrupted pre-trained weights. Introducing step-zero identity initialization, asymmetric multi-label loss, and empirical validation threshold tuning lifted performance to the official winning 0.3700 score.

## 5. Failure Analysis & What Didn't Work

Lexical baselines (BM25: 0.2244, TF-IDF: 0.2356) completely collapsed on descriptive factual queries where common synonyms were used (e.g. 'unlawful assembly' or 'rioting with deadly weapons' without explicitly mentioning 'Section 147' or 'Section 148 IPC'). Conversely, standard neural cross-entropy classifiers without threshold calibration collapsed on long-tail sections, predicting only the top 10 most common sections.

