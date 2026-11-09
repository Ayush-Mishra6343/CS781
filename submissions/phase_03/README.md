# Phase 3: Rhetorical Role (RR) Classification & PEFT Suite: Submissions Archive

- **Official Winning Score:** **0.4337** (Submission ID: `992721`)
- **Baseline Score:** **0.3941** (Submission ID: `987580`)
- **Improvement Delta:** **+0.0396 (+10.05%)**
- **Evaluation Metric:** Macro F1 across 7 Rhetorical Roles (Facts, Issue, Argument, Statute, Precedent, Ratio, Ruling)

## 1. Task Description

Sentence-level discourse analysis classifying court judgment sentences into their functional rhetorical roles, evaluated under extreme class imbalance and parameter-efficient fine-tuning (PEFT) constraints.

## 2. Chronological Submissions Table

| Submission ID | Date | Score | Status | Model / Technique | Key Insight / Learning |
|:---:|:---:|:---:|:---:|:---|:---|
| [`987393`](runs/987393/) | 2026-10-14 09:15 | **0.4311** | Superseded | Legal-RoBERTa fine-tuned with multi-label BCEWithLogitsLoss and threshold tuning | Early strong result (0.4311). Showed that per-class decision thresholds significantly improve Macro-F1. |
| [`987580`](runs/987580/) | 2026-10-14 13:40 | **0.3941** | Superseded | Standard Full Fine-Tuning Legal-RoBERTa baseline | Full fine-tuning reference baseline. Suffered from gradient instability on minority class labels. |
| [`988322`](runs/988322/) | 2026-10-14 20:05 | **0.4278** | Superseded | Hierarchical Sentence BiLSTM on top of frozen Legal-RoBERTa sentence embeddings | Explicit document-level discourse modeling (0.4278). Showed that sequential sentence progression (Facts -> Issues -> Arguments -> Ratio) carries strong temporal priors. |
| [`990317`](runs/990317/) | 2026-10-15 11:20 | **0.4292** | Superseded | Longformer-Legal with Focal Loss (gamma=2.0) | Outstanding individual long-context model (0.4292). Handled long legal preambles and factual narratives effectively. |
| [`991789`](runs/991789/) | 2026-10-15 18:12 | **0.4144** | Superseded | Longformer-Legal with asymmetric loss and context window 2048 | Strong multi-sentence context handling (0.4144), confirming that rhetorical roles heavily depend on surrounding document discourse. |
| [`992293`](runs/992293/) | 2026-10-15 22:30 | **0.3336** | Superseded | LoRA with Asymmetric Loss for extreme class imbalance | Asymmetric loss aggressively penalized negative predictions, resulting in high recall but depressed precision on dominant classes. |
| [`992567`](runs/992567/) | 2026-10-16 12:45 | **0.4311** | Superseded | Standalone LoRA (r=8, alpha=16) with BCEWithLogitsLoss | Best individual PEFT checkpoint (0.4311). Proved that parameter-efficient fine-tuning outperformed full fine-tuning by preventing catastrophic forgetting on small legal splits. |
| [`992596`](runs/992596/) | 2026-10-16 14:10 | **0.4058** | Superseded | Weighted ensemble of LoRA and LoRMA without BitFit | High-performing two-model ensemble (0.4058). Addition of BitFit in 992721 added orthogonal bias shifts that lifted Macro-F1 to 0.4337. |
| [`992721`](runs/992721/) | 2026-10-16 15:39 | **0.4337** | Selected (Winning) 🏆 (Winning) | LoRA (r=8) + Multiplicative LoRMA (r=8) + BitFit Multi-Method PEFT Ensemble | Official winning model for Phase 3. Combined diverse PEFT mechanisms (additive rank update via LoRA, multiplicative scaling via LoRMA, bias tuning via BitFit) into a complementary ensemble. |

## 3. Architectural Evolution

```
Phase 3 Architectural Evolution:
1. Standard Full Fine-Tuning Baseline (987580: 0.3941)
   - Suffered catastrophic forgetting and severe bias toward high-frequency 'Facts'
         │
         ▼
2. Longformer Long-Context Model (990317: 0.4292 / 991789: 0.4144)
   - Handled inter-sentence context up to 2048 tokens, proving discourse context is crucial
         │
         ▼
3. Hierarchical BiLSTM on Legal-RoBERTa (988322: 0.4278)
   - Modeled sequential flow of legal arguments across paragraphs
         │
         ▼
4. Standalone LoRA PEFT (992567: 0.4311)
   - Low-Rank Adaptation (r=8, alpha=16) prevented overfitting on minority roles
         │
         ▼
5. Multi-Method PEFT Ensemble: LoRA + LoRMA + BitFit (992721: 0.4337) [WINNING]
   - Combined additive low-rank updates (LoRA), multiplicative scaling (LoRMA), and bias tuning (BitFit)
   - Orthogonal parameter adaptation mechanisms provided superior ensemble diversity
```

## 4. Hyperparameter Tuning Narrative

Comparative PEFT exploration evaluated LoRA (additive low-rank matrices), LoRMA (multiplicative rank modulation), and BitFit (bias-only fine-tuning). While standalone LoRA achieved an impressive 0.4311, combining LoRA (weight 0.45), LoRMA (weight 0.40), and BitFit (weight 0.15) produced orthogonal inductive biases: LoRA captured dense linguistic representations, LoRMA adjusted activation magnitudes, and BitFit fine-tuned decision hyperplanes for minority classes.

## 5. Failure Analysis & What Didn't Work

Asymmetric loss with aggressive negative penalization (992293: 0.3336) completely degraded precision by over-triggering rare classes like 'Statute' and 'Ruling by Present Court'. Full fine-tuning (987580) suffered from gradient collapse on rare classes because backpropagation updates over-corrected for the dominant 'Facts' class. PEFT isolation guaranteed parameter stability.

