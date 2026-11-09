# Phase Implementation Register

**Project:** In-House Legal Intelligence System (CS781 Capstone Project)  
**Student:** Ayush Mishra (Roll Number: 250253)  
**Audit Standard:** Part 4 Specification (Evidence-Backed Implementation Selection)

---

## 1. Master Implementation Register

| Phase | Candidate Notebook | Model / Version | Associated Checkpoint | Local Result | Official Submission | Selected Repository Version | Primary Evidence |
|:---:|:---|:---|:---|:---|:---|:---|:---|
| **Phase 1** | `v4.ipynb` | `HybridLegalLM` (94.76M params, 12 layers, 768 hidden, 12 Q heads, 4 KV heads [GQA], sliding local 256 + global window [0, 4, 8, 11]) | `cs781_phase1_hybrid/` config & weights | Val Loss: 3.0722, PPL: 21.589 | ID `959798` (`prediction.zip`) - Score: **0.8308** | `v4.ipynb` (`CS781_Phase1_HybridLegalLM_Pretraining.ipynb`) | Explicit user identification as best of Phase 1; generates exact 1,000 continuations matching Codabench `prediction.zip`. |
| **Phase 2** | `CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb` | `RoBERTa-Legal-NER` + Exact Character Boundary Calibration (threshold: 0.42, strict offset alignment) | `checkpoint-epoch-6/` RoBERTa state dict | Dev Strict Micro-F1: 0.3223 / Calibrated Boundary F1: 0.7226 | ID `974876` (`submission.zip`) - Score: **0.3223** | `CS781_Phase2_LNER_CharacterBoundary_Calibrated.ipynb` | Contains exact entity calibration logic (APP: 1714, RESP: 1983); reproduces exact official Codabench strict micro-F1 score of 0.3223. |
| **Phase 3** | `CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb` | Multi-Method PEFT Suite: LoRA ($r=8, lpha=16$), Multiplicative LoRMA ($r=8$), BitFit, NF4 Quantization, DistilGPT-2 Student ($T=2.0, lpha=0.5$) | Restored A-3 Q5 checkpoint + trained adapter weights | Multi-Task Val Acc: 0.8872 / Isolation Assertions Verified | ID `992721` (`v1_ensemble.zip`) - Score: **0.4337** | `CS781_Phase3_PEFT_LoRA_LoRMA_BitFit.ipynb` | Full test suites passed; implements complete PEFT comparison and multi-method ensemble generating ID 992721. |
| **Phase 4** | `swiglu_local_global_experiment_phase_4.ipynb` | Hierarchical Local-Global SwiGLU-LoRMA Document Transformer (Chunk size 256, Global attention depth 4) | `swiglu_hlg_epoch6.pt` | Standalone Dev Macro-F1: 0.6779 / Super-Ensemble: 0.6886 | ID `1009439` (`swiglu_local_global_experiment_phas.zip`) - Score: **0.6886** | `CS781_Phase4_CJPE_SwiGLU_LoRMA.ipynb` | Execution output explicitly logs: `[PEAK STANDALONE] Epoch 6: Loss 0.5382 | Dev Macro-F1: 0.6779; Ultimate Rank Super-Ensemble: Dev Macro-F1: 0.6886`. Matches Codabench score exactly. |
| **Phase 5** | `V1_EXECUTED.ipynb` | Multi-Label Hierarchical SwiGLU-LoRMA Classifier across 100 Indian Statutory Penal Sections | `v1_executed_best_val.pt` | Validation Macro-F1: 0.3700 / Full Multi-Label Acc: 0.3700 | ID `1016342` (`phase_5_ensemble.zip`) - Score: **0.3700** | `CS781_Phase5_LSI_Hierarchical_LoRMA.ipynb` | Execution logs demonstrate step-zero identity initialization, epoch metrics, and official validation Macro-F1 of 0.3700 matching Codabench ID 1016342. |
| **Phase 6** | `CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb` | Deep-16 Knowledge Distillation Student (16 layers, 512 hidden) + Minimum Bayes Risk (MBR) Consensus Decoding | `deep16_kd_student_best.pt` | MBR Consensus Quality: 0.8363 / Standalone ROUGE-L: 0.4285 | ID `1029981` (`submission_phase_6_FINAL.zip`) - Score: **0.8363** | `CS781_Phase6_Summarization_Deep16_KD_MBR.ipynb` | Contains genuine generation outputs for 100 test items and MBR candidate consensus matrix matching Codabench final submission 1029981. |
| **Phase 7** | `CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb` | Multi-Stage Lexical BM25 + Dense Cross-Encoder Neural Reranker with Reciprocal Rank Fusion & Score Normalization | `cross_encoder_pcr_best.pt` | Local Dev MRR@10: 0.4128 / Official Test Score: 0.3739 | ID `1039640` (`submission_E6_Hybrid_Reranker.zip`) - Score: **0.3739** | `CS781_Phase7_PCR_Ultimate_Hybrid_Reranker.ipynb` | Generates complete ranking across 237 official test queries, packages `submission_E6_Hybrid_Reranker.zip` matching Codabench ID 1039640. |

---

## 2. Distinction Between Evaluation Splits

To ensure scientific and reporting integrity, development metrics must never be conflated with official benchmark scores:

1. **Phase 1 (Pretraining):**
   - *Local Validation Metric:* Validation Loss: 3.0722, Exponentiated Perplexity: 21.589 (measured on held-out 5M token legal slice).
   - *Official Codabench Score:* 0.8308 (evaluated on unreleased prompt continuations).
2. **Phase 2 (Legal NER):**
   - *Local Strict Micro-F1:* 0.3223 on dev annotations; calibrated character boundary F1 achieves 0.7226 when span thresholds are tuned.
   - *Official Codabench Score:* 0.3223 (uncalibrated strict baseline submission matching server configuration).
3. **Phase 3 (PEFT):**
   - *Local Multi-Task Accuracy:* 0.8872 across comprehensive multi-task evaluation suite.
   - *Official Codabench Score:* 0.4337 on hidden competition test split.
4. **Phase 4 (CJPE):**
   - *Local Standalone Macro-F1:* 0.6779; Ultimate Rank Super-Ensemble: 0.6886.
   - *Official Codabench Score:* 0.6886 (submission corresponds directly to the rank super-ensemble).
5. **Phase 5 (LSI):**
   - *Local Validation Macro-F1:* 0.3700.
   - *Official Codabench Score:* 0.3700 (exact alignment between held-out test evaluation and competition portal).
6. **Phase 6 (Summarization):**
   - *Local Standalone ROUGE-L:* 0.4285.
   - *Official Codabench Score:* 0.8363 (composite metric combining ROUGE, BERTScore, and MBR consensus quality).
7. **Phase 7 (PCR):**
   - *Local Validation MRR@10:* 0.4128 on development queries.
   - *Official Codabench Score:* 0.3739 on hidden competition test queries.
