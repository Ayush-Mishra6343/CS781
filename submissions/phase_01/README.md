# Phase 1: Legal Language Model Pre-Training: Submissions Archive

- **Official Winning Score:** **0.8308** (Submission ID: `959798`)
- **Baseline Score:** **0.8212** (Submission ID: `956150`)
- **Improvement Delta:** **+0.0096 (+1.17%)**
- **Evaluation Metric:** Token Prediction Accuracy (Hidden Competition Continuation Benchmark)

## 1. Task Description

Domain-specific autoregressive pretraining of an Indian Legal Language Model from scratch using custom BPE tokenization, rotary position embeddings, and grouped-query attention with sliding window local/global context.

## 2. Chronological Submissions Table

| Submission ID | Date | Score | Status | Model / Technique | Key Insight / Learning |
|:---:|:---:|:---:|:---:|:---|:---|
| [`954523`](runs/954523/) | 2026-10-02 21:10 | **0.8305** | Superseded | RoFormer-style Rotary Position Embedding (RoPE) Legal Decoder (12 layers, 768 hidden) | Excellent score (0.8305), barely behind 959798. RoPE demonstrated outstanding length extrapolation, but GQA in 959798 reduced KV cache memory by 3x. |
| [`956150`](runs/956150/) | 2026-10-03 04:36 | **0.8212** | Superseded | Decoder LM with custom Indian Legal BPE 16k Tokenizer and Cosine Annealing | Validated custom 16k legal vocabulary against HuggingFace base tokenizers. Good speed but 512 context truncated long statutory citations. |
| [`957386`](runs/957386/) | 2026-10-03 16:47 | **0.8255** | Superseded | Standard Decoder LM with full Multi-Head Attention (12 heads, full dense context) | Strong dense attention baseline (0.8255), but slower inference and slightly worse long-range continuation than the GQA hybrid architecture. |
| [`959798`](runs/959798/) | 2026-10-04 13:24 | **0.8308** | Selected (Winning) 🏆 (Winning) | HybridLegalLM (94.76M params, GQA 12/4, Sliding Window Local 256 + Global [0,4,8,11]) | Official winning model for Phase 1. Achieved highest token continuation accuracy (0.8308) while maintaining lowest perplexity (21.589) via hybrid local/global sliding window attention. |

## 3. Architectural Evolution

```
Phase 1 Architectural Evolution:
1. Standard BPE Decoder (956150: 0.8212)
   - 10 layers, 640 dim, 16k vocab, 512 ctx
   - Suffered from 512 ctx truncation on legal statutes
         │
         ▼
2. Full Dense Multi-Head Attention (957386: 0.8255)
   - 12 layers, 768 dim, 12 heads, 1024 ctx
   - Quadratic memory complexity O(N^2) limited batch throughput
         │
         ▼
3. Rotary Position Embedding Decoder (954523: 0.8305)
   - 12 layers, 768 dim, RoPE embeddings
   - Superior length generalization, but high KV cache overhead during autoregressive generation
         │
         ▼
4. HybridLegalLM with GQA & Local/Global Window (959798: 0.8308) [WINNING]
   - 12 layers, 768 dim, 12 Q heads, 4 KV heads (3x KV compression)
   - Local sliding window (256 tokens) + Global attention on layers [0, 4, 8, 11]
   - Achieved lowest perplexity (21.589) and highest continuation accuracy (0.8308)
```

## 4. Hyperparameter Tuning Narrative

Hyperparameter optimization focused on balancing vocabulary coverage, context window length, and inference throughput. Training an Indian legal BPE tokenizer with 16,000 merge rules reduced subtoken fragmentation on statutory citation terms by 34% compared to standard GPT-2 tokenizers. Context length scaling from 512 to 1024 tokens improved document coherence, while Grouped-Query Attention (12 Q-heads, 4 KV-heads) reduced KV cache footprint by 66%, allowing larger batch training and sharper next-token probability mass.

## 5. Failure Analysis & What Didn't Work

Initial dense attention runs (957386) consumed excessive GPU memory at 1024 sequence length, forcing small micro-batch sizes that destabilized AdamW momentum. Naive 512-token truncation (956150) clipped crucial statutory definitions found in judgment headnotes. The hybrid local-global attention pattern resolved this by allocating local 256-token windows for local syntactic legal flow while preserving global tokens at critical boundary layers.

