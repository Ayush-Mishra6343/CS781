# Model Checkpoints Manifest

**Project:** In-House Legal Intelligence System (CS781 Capstone Project)

---

## Checkpoint Manifest & Parameter Summary

Checkpoints exceeding 100MB are excluded from direct Git tracking to maintain repository health. The architecture configurations, initialization weights, and download manifests are documented below:

| Phase | Model Identifier | Parameter Count | Architecture Description | Checkpoint SHA-256 Manifest |
|:---:|:---|:---:|:---|:---|
| **P1** | `HybridLegalLM` | 94,764,288 | 12 layers, 12 Q / 4 KV heads (GQA), sliding local 256 + global layers | `cs781_phase1_hybrid/model.pt` |
| **P2** | `RoBERTa-Legal-NER` | 124,655,629 | RoBERTa token classification head + calibrated span decoder | `checkpoint-epoch-6/model.pt` |
| **P3** | `PEFT-LoRMA-Student` | 81,912,576 | DistilGPT-2 Student + LoRMA multiplicative adapters ($r=8$) | `student_best_lorma.pt` |
| **P4** | `SwiGLU-LoRMA-CJPE` | 42,118,146 | Hierarchical chunk SwiGLU encoders + global document transformer | `swiglu_hlg_epoch6.pt` |
| **P5** | `Hierarchical-LSI` | 44,228,836 | 100-class multi-label statutory classifier with identity initialization | `v1_executed_best_val.pt` |
| **P6** | `Deep16-KD-Summarizer` | 86,412,288 | 16-layer compressed student model + MBR consensus generator | `deep16_kd_student_best.pt` |
| **P7** | `CrossEncoder-PCR` | 110,882,305 | Dense cross-encoder reranker + reciprocal rank fusion | `cross_encoder_pcr_best.pt` |

To download pretrained weights, use the acquisition script provided in `scripts/download_checkpoints.py` or load the model directly via the HuggingFace Hub using the model card tags documented in each phase directory.
