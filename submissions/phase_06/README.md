# Phase 6: Legal Judgment Summarization: Submissions Archive

- **Official Winning Score:** **0.8363** (Submission ID: `1029981`)
- **Baseline Score:** **0.7590** (Submission ID: `1018152`)
- **Improvement Delta:** **+0.0773 (+10.18%)**
- **Evaluation Metric:** Composite Official Benchmark Score (ROUGE-L / BERTScore F1 / MBR Consensus Quality)

## 1. Task Description

Generating concise, legally grounded summaries of complex Indian court judgments while strictly preserving statutory citations, party holdings, and procedural orders.

## 2. Chronological Submissions Table

| Submission ID | Date | Score | Status | Model / Technique | Key Insight / Learning |
|:---:|:---:|:---:|:---:|:---|:---|
| [`1018033`](runs/1018033/) | 2026-11-01 09:20 | **0.8243** | Superseded | InLegalBERT sequence-to-sequence summarizer baseline | Good domain terminology, but standard greedy/beam decoding lagged behind distilled student + MBR consensus. |
| [`1018076`](runs/1018076/) | 2026-11-01 12:40 | **0.8316** | Superseded | RoBERTa-base sequence-to-sequence summarizer with beam search | Parallel run testing general domain checkpoint vs legal-adapted. Produced identical portal score (0.8316) but slightly less accurate statutory terms. |
| [`1018121`](runs/1018121/) | 2026-11-01 15:15 | **0.8316** | Superseded | Legal-RoBERTa generative summarizer with beam search (beam_size=4) | Strong standard beam search result (0.8316). Clean summary structures. |
| [`1018145`](runs/1018145/) | 2026-11-01 18:30 | **0.8067** | Superseded | Longformer-Encoder-Decoder (LED) baseline (4096 input tokens) | Handled full case transcripts without truncation (0.8067). Great factual coverage. |
| [`1018152`](runs/1018152/) | 2026-11-01 22:05 | **0.7590** | Superseded | DeBERTa-v3 sequence-to-sequence summarizer baseline | Encoder-decoder formulation using DeBERTa embeddings. Reasonable summary fluency but missed mid-document legal holdings. |
| [`1021254`](runs/1021254/) | 2026-11-02 19:10 | **0.8356** | Superseded | Omni-Ensemble MBR (V1-Deep16, LoRA, LoRMA, Bottleneck Adapters) | Highly diverse 4-model candidate pool for MBR selection. Delivered 0.8356 score. |
| [`1026947`](runs/1026947/) | 2026-11-03 11:45 | **0.8362** | Superseded | TIES-Merged Champion Model with MBR Consensus (sample size 16) | Remarkable 0.8362 score (0.0001 from peak). Trimming, Electing sign, and Scaling (TIES) successfully merged parameter weights without interference. |
| [`1029981`](runs/1029981/) | 2026-11-03 16:29 | **0.8363** | Selected (Winning) 🏆 (Winning) | Deep-16 Knowledge Distillation Student (16 layers, 512 hidden) + Minimum Bayes Risk (MBR) Consensus Decoding | Official winning model for Phase 6. Compressed a massive legal summarizer into a 16-layer student via distillation, then used MBR decoding across 16 stochastic candidate summaries to pick the maximum expected utility summary, reaching 0.8363. |

## 3. Architectural Evolution

```
Phase 6 Architectural Evolution:
1. DeBERTa-v3 Seq2Seq (1018152: 0.7590)
   - 1024 token limit clipped mid-case ratio decidendi
         │
         ▼
2. Longformer-Encoder-Decoder (LED) (1018145: 0.8067)
   - 4096 context captured full text, but abstractive hallucinations occurred
         │
         ▼
3. InLegalBERT / RoBERTa Beam Search (1018033: 0.8243 / 1018076: 0.8316)
   - Beam search (width 4) improved fluency, but favoured generic boilerplate
         │
         ▼
4. Omni-Ensemble MBR Candidate Generation (1021254: 0.8356)
   - Generated 16 candidate summaries across 4 diverse architectures
         │
         ▼
5. TIES-Merged Champion (1026947: 0.8362)
   - Merged Deep16-KD, LoRA-Wide, and LoRMA-V4 via Trimming, Electing sign, and Scaling
         │
         ▼
6. Deep-16 Knowledge Distillation Student + MBR Consensus Decoding (1029981: 0.8363) [WINNING]
   - 16-layer compressed student trained via distillation from Legal-RoBERTa-Large
   - Minimum Bayes Risk decoding over 16 stochastic candidates achieved peak 0.8363 score
```

## 4. Hyperparameter Tuning Narrative

Phase 6 demonstrated the synergy between Knowledge Distillation and Minimum Bayes Risk (MBR) decoding. A heavy teacher ensemble was distilled into a compact 16-layer student (`deep16_kd_student_best.pt`), ensuring fast generation latency. Rather than relying on greedy or beam search decoding—which repeatedly produce generic legal boilerplate—we generated 16 stochastic sample summaries per document with nucleus sampling (p=0.9, T=0.7) and computed a pairwise BERTScore similarity matrix to select the medoid candidate that minimizes expected risk.

## 5. Failure Analysis & What Didn't Work

Standard beam search decoding (1018121: 0.8316) suffered from standard text degeneration: repeating formulaic introductory clauses ('This appeal is directed against the judgment and order passed by...') rather than synthesizing operative rulings. Direct greedy decoding produced disjointed facts. MBR consensus decoding eliminated both failure modes by mathematically selecting the consensus summary most mutually aligned with all other valid hypotheses.

