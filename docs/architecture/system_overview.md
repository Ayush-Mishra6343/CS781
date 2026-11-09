# In-House Legal Intelligence System: Technical Architecture Overview

**Author:** Ayush Mishra (Roll Number: 250253)  
**Institution:** Indian Institute of Technology Kanpur  
**Course:** CS781: Deep Learning for Natural Language Processing (Capstone Project)  

---

## 1. System Philosophy & Objectives

Legal text presents unique structural and linguistic challenges:
1. **Extreme Document Length:** Supreme Court and High Court judgments frequently exceed 10,000 to 50,000 words, far surpassing standard transformer context windows (512 tokens).
2. **Specialized Terminology:** Complex statutory citations, Latin maxims (*mens rea*, *suo motu*, *certiorari*), and jurisdictional shorthand.
3. **Severe Class Imbalance:** Statutory provisions and rhetorical arguments follow extreme power-law distributions.
4. **Discourse Hierarchy:** Legal judgments follow rigid discourse structures (Factual Background -> Procedural History -> Contentions -> Statutory Provisions -> Ratio Decidendi -> Operative Order).

The **In-House Legal Intelligence System** addresses these challenges via a modular, seven-phase end-to-end NLP architecture:

```
[Raw Legal Judgments & Statutes]
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Domain-Specific Pre-Training                  │
│ • Custom Indian Legal BPE Tokenizer (16,000 rules)      │
│ • HybridLegalLM: 12 Layers, 768 Dim, GQA (12/4)         │
│ • Sliding Window Local (256) + Global Attention [0,4,8] │
└──────────────────────────┬──────────────────────────────┘
                           │
         ┌─────────────────┴──────────────────┐
         ▼                                    ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ Phase 2: Structured Entity  │  │ Phase 3: Discourse Role     │
│ Extraction (L-NER)          │  │ Classification (RR)         │
│ • RoBERTa-Legal-NER         │  │ • Multi-Method PEFT Suite   │
│ • Calibrated Character      │  │ • LoRA (r=8) + LoRMA (r=8)  │
│   Boundary Alignment        │  │ • BitFit Bias Adaptation    │
│ • Strict Micro-F1: 0.3223   │  │ • Macro-F1: 0.4337          │
└──────────────┬──────────────┘  └──────────────┬──────────────┘
               │                                │
               └────────────────┬───────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Court Judgment Prediction & Explanation (CJPE) │
│ • Hierarchical Local-Global Chunking (16 x 256 tokens)   │
│ • SwiGLU Gating & LoRMA-Plus Adapter                    │
│ • Rank Super-Ensemble Prediction (Macro-F1: 0.6886)     │
└──────────────────────────┬──────────────────────────────┘
                           │
         ┌─────────────────┴──────────────────┐
         ▼                                    ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ Phase 5: Legal Statute      │  │ Phase 6: Judgment           │
│ Identification (LSI)        │  │ Summarization               │
│ • Multi-Label HLG-SwiGLU    │  │ • Student Distillation      │
│ • 100 IPC Penal Sections    │  │   (16-layer compressed)     │
│ • Step-Zero Identity Init   │  │ • Minimum Bayes Risk (MBR)  │
│ • Macro-F1: 0.3700          │  │ • Composite Score: 0.8363   │
└─────────────────────────────┘  └──────────────┬──────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 7: Prior Case Retrieval & Neural Reranking (PCR)  │
│ • Fast Vectorized Okapi BM25 First-Stage (Top-50 hits)  │
│ • Deep Cross-Encoder Neural Reranking                   │
│ • Reciprocal Rank Fusion (k=60) + Min-Max Normalization │
│ • Official Micro-F1@7: 0.3739 (+218% over BM25)         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Core Architectural Design Patterns

### Pattern A: Grouped-Query Attention (GQA) & Sliding Window Local/Global Attention
- **Problem:** Full dense self-attention scales quadratically in time and memory, making long sequence pretraining prohibitively expensive.
- **Solution:** Grouped-Query Attention partitions 12 query heads across 4 key/value heads, reducing KV cache footprint by 66%. Attention layers alternate between local sliding windows (256 tokens) for local syntactic coherence and global attention at layers [0, 4, 8, 11] to route cross-document legal principles.

### Pattern B: Multi-Method Parameter-Efficient Fine-Tuning (PEFT)
- **Problem:** Full fine-tuning of 100M+ parameter models on small, highly imbalanced legal tasks causes catastrophic forgetting and overfits dominant majority classes.
- **Solution:** We developed a multi-method PEFT ensemble:
  1. **LoRA (Low-Rank Adaptation):** Additive low-rank parameter updates.
  2. **LoRMA (Low-Rank Multiplicative Adaptation):** Multiplicative scale modulation.
  3. **BitFit:** Freezes all weight matrices and optimizes exclusively the bias terms.
  - Combining these complementary inductive biases yielded our winning Phase 3 score of 0.4337.

### Pattern C: Hierarchical Local-Global Chunking with SwiGLU Gating
- **Problem:** Legal judgments cannot fit in standard 512-token transformer windows. Truncation discards the verdict and ratio decidendi located at the end.
- **Solution:** Partition documents into 16 uniform chunks of 256 tokens. Each chunk is processed locally by a transformer backbone with SwiGLU non-linear gating. A second-stage global cross-attention layer aggregates all 16 chunk representations into a unified document-level vector.

### Pattern D: Minimum Bayes Risk (MBR) Consensus Decoding
- **Problem:** Standard autoregressive beam search and greedy decoding repeatedly generate repetitive, generic legal boilerplate.
- **Solution:** MBR decoding samples N=16 stochastic candidate summaries from the model using nucleus sampling (p=0.9, T=0.7) and computes an N x N pairwise BERTScore utility matrix. The candidate with the highest expected consensus utility is selected. This guarantees factual coverage, eliminates hallucinations, and achieved the winning 0.8363 score in Phase 6.

### Pattern E: Multi-Stage Hybrid Retrieval with Reciprocal Rank Fusion
- **Problem:** Pure lexical retrieval (BM25) fails when judges cite synonymous concepts; pure dense bi-encoders fail when specific statutory section citations must be matched exactly.
- **Solution:** A two-stage pipeline:
  - Stage 1: Vectorized Okapi BM25 retrieves the top 50 high-recall candidates in <15ms.
  - Stage 2: Deep Cross-Encoder scores full query-candidate token interactions.
  - Fusion: Reciprocal Rank Fusion (k=60) dynamically merges dense and lexical scores into calibrated Top-7 rankings, achieving 0.3739 Micro-F1@7 (a 218% relative improvement over BM25).
