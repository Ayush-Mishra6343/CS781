# Methodology and Technical Limitations

**Project:** In-House Legal Intelligence System (CS781 Capstone Project)  
**Student:** Ayush Mishra (Roll Number: 250253)  
**Institution:** Indian Institute of Technology Kanpur

---

## 1. Research Methodology Across Phases

### Phase 1: Domain-Specific Language Modeling
- **Motivation:** Generic language models lack domain-specific syntactic understanding of Indian statutory structures, citation patterns, and archaic legal phraseology.
- **Engineering Decision:** Pretrained a 94.76M parameter model from scratch on 100M tokens of Indian Supreme Court decisions using Grouped-Query Attention (GQA factor 3) and hybrid local-global sliding window attention.
- **Finding:** GQA delivered a $2.8\times$ speedup during autoregressive KV-cache decoding while maintaining competitive validation perplexity (21.589).

### Phase 2: Calibrated Strict-Span Named Entity Recognition
- **Motivation:** Standard token classification produces fragmented entity boundaries on multi-token legal citations (e.g., *Section 302 of the Indian Penal Code*).
- **Engineering Decision:** Implemented character-level offset alignment with probability calibration.
- **Finding:** While boundary calibration elevated local validation F1 to 0.7226, official uncalibrated server baseline scored 0.3223 due to strict span scoring criteria on the competition server.

### Phase 3: Parameter-Efficient Fine-Tuning Benchmarking
- **Motivation:** Full model fine-tuning is compute-prohibitive across multiple specialized legal downstream tasks.
- **Engineering Decision:** Evaluated LoRA ($r=8$), Multiplicative LoRMA ($r=8$), BitFit, and 4-bit NormalFloat (NF4) quantization.
- **Finding:** Multiplicative LoRMA provided superior adapter stability compared to additive LoRA on domain-shifted sequences, achieving 0.8872 validation accuracy on IMDb benchmarks.

### Phase 4: Court Judgment Prediction and Explanation
- **Motivation:** Indian judgments average 4,500+ tokens, exceeding standard 512-token transformer context limits.
- **Engineering Decision:** Developed a Hierarchical Local-Global SwiGLU-LoRMA architecture where 256-token chunk embeddings are processed by a global document transformer.
- **Finding:** Outperformed 3-region heuristics, achieving an official Dev Macro-F1 score of 0.6886.

### Phase 5: Multi-Label Legal Statute Identification
- **Motivation:** Judicial decisions simultaneously trigger multiple statutory provisions across different legal codes.
- **Engineering Decision:** Multi-label binary cross-entropy classification across 100 target penal sections with step-zero identity initialization.
- **Finding:** Step-zero initialization prevented early gradient explosion, achieving an official Macro-F1 score of 0.3700.

### Phase 6: Abstractive Summarization with MBR Consensus Decoding
- **Motivation:** Standard beam search generation suffers from hallucinated factual distortions in judicial summaries.
- **Engineering Decision:** Distilled teacher model into a Deep-16 student and applied Minimum Bayes Risk (MBR) consensus decoding over generated candidates.
- **Finding:** MBR consensus elevated the composite Codabench evaluation score to 0.8363.

### Phase 7: Prior Case Retrieval Multi-Stage Reranking
- **Motivation:** Lexical keyword search (BM25) fails on semantic synonymy, while dense neural retrieval misses exact statutory article numbers.
- **Engineering Decision:** Two-stage hybrid pipeline combining BM25 candidate retrieval with a dense neural cross-encoder reranker and reciprocal rank fusion.
- **Finding:** Boosted local dev MRR@10 to 0.4128 and achieved an official competition score of 0.3739.

---

## 2. Technical Limitations & Future Work

1. **GPU Memory Ceiling:** Document context lengths were constrained to 16 chunks ($16 \times 256 = 4,096$ tokens) during hierarchical processing due to 16 GB VRAM limits.
2. **Class Imbalance in Statutes:** Rare penal provisions (< 20 occurrences in training corpus) suffer from lower recall compared to ubiquitous sections (e.g., IPC Section 302 or Section 34).
3. **External Dependencies:** Offline reproduction requires pre-downloaded token caches and checkpoints.
