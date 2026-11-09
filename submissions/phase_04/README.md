# Phase 4: Court Judgment Prediction and Explanation (CJPE): Submissions Archive

- **Official Winning Score:** **0.6886** (Submission ID: `1009439`)
- **Baseline Score:** **0.5520** (Submission ID: `1006364`)
- **Improvement Delta:** **+0.1366 (+24.75%)**
- **Evaluation Metric:** Macro F1 (Binary Outcome Prediction & Explanation Grounding)

## 1. Task Description

Predicting judicial appeal outcomes (Allowed vs Dismissed) from multi-thousand-word Indian court records while extracting grounding textual explanations.

## 2. Chronological Submissions Table

| Submission ID | Date | Score | Status | Model / Technique | Key Insight / Learning |
|:---:|:---:|:---:|:---:|:---|:---|
| [`1006364`](runs/1006364/) | 2026-10-24 18:22 | **0.5520** | Superseded | Chunked Legal-RoBERTa baseline with mean-pooling over 8 chunks | Initial hierarchical prototype. Simple mean-pooling diluted crucial ratio decidendi statements across uninformative factual paragraphs. |
| [`1007647`](runs/1007647/) | 2026-10-25 16:50 | **0.5735** | Superseded | Soft Prompt Tuning (20 virtual tokens) + LoRMA Adapter with Asymmetric Loss | Prompt tuning guided model attention toward jurisdictional header cues, achieving a strong 0.5735. |
| [`1007658`](runs/1007658/) | 2026-10-25 19:15 | **0.5377** | Superseded | Hierarchical Transformer with standard Cross-Entropy Loss | Standard cross-entropy baseline for hierarchical architecture. |
| [`1007668`](runs/1007668/) | 2026-10-25 21:40 | **0.5464** | Superseded | Hierarchical Transformer with Focal Loss (gamma=2.0) | Focal loss helped with the extreme judgment class distribution (Affirmed vs Reversed), but tended to over-predict reversals. |
| [`1008160`](runs/1008160/) | 2026-10-26 12:30 | **0.5693** | Superseded | SwiGLU-LoRMA Standalone Model at Epoch 6 | Solid standalone baseline (0.5693). Adding the second stage LoRMA-Plus tuning and rank fusion boosted score to 0.6886. |
| [`1009439`](runs/1009439/) | 2026-10-26 18:53 | **0.6886** | Selected (Winning) 🏆 (Winning) | Hierarchical Local-Global SwiGLU-LoRMA Document Classifier (Rank Super-Ensemble) | Official winning model for Phase 4. Solved long legal judgment classification (often >10,000 words) using hierarchical chunk-and-aggregate with SwiGLU gating and LoRMA parameter efficiency. |

## 3. Architectural Evolution

```
Phase 4 Architectural Evolution:
1. Flat Chunked Legal-RoBERTa (1006364: 0.5520)
   - Naive mean-pooling over 8 chunks diluted the decisive legal holdings
         │
         ▼
2. Hierarchical Transformer + CE Loss (1007658: 0.5377)
   - Chunk-level self-attention, but majority class dominated gradients
         │
         ▼
3. Hierarchical Transformer + Focal Loss (1007668: 0.5464)
   - Focal loss improved minority class attention, but introduced false reversal predictions
         │
         ▼
4. Soft Prompt Tuning + LoRMA Adapter (1007647: 0.5735)
   - Virtual prefix tokens guided attention toward jurisdictional header cues
         │
         ▼
5. SwiGLU-LoRMA Standalone (1008160: 0.5693)
   - SwiGLU gating mechanism captured non-linear legal concept interactions
         │
         ▼
6. Hierarchical Local-Global SwiGLU-LoRMA Super-Ensemble (1009439: 0.6886) [WINNING]
   - 16 chunks of 256 tokens, 4 global aggregation layers, SwiGLU activation
   - Rank-based multi-view super-ensemble achieved peak score of 0.6886
```

## 4. Hyperparameter Tuning Narrative

Court judgments present extreme document lengths, often spanning 15 to 50 pages. Standard 512-token truncation completely eliminates the court's reasoning and ratio decidendi located at the document conclusion. The Hierarchical Local-Global architecture partitioned each document into 16 chunks of 256 tokens, computed local chunk representations using SwiGLU gating, and passed chunk representations through 4 global cross-attention layers. Ensembling across rank permutations delivered the winning 0.6886 Macro F1.

## 5. Failure Analysis & What Didn't Work

Initial flat chunk mean-pooling (1006364) treated all document paragraphs equally: procedural opening paragraphs (parties, court location, procedural history) overwhelmed the subtle legal doctrine discussed in a single operative paragraph near the end. Without global cross-chunk attention routing, the model was incapable of linking earlier factual allegations to the judge's final verdict.

