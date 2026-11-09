# Dataset Documentation & Setup

**Project:** In-House Legal Intelligence System (CS781 Capstone Project)

---

## Legal Corpora Overview

The project relies on public judgments from the Supreme Court of India and High Courts:

1. **Phase 1 (Pretraining):** ~105M tokens of raw Indian court judgments tokenized with custom 25k BPE.
2. **Phase 2 (L-NER):** Annotated judgments with character-level spans for 13 entity types (Petitioner, Respondent, Court, Judge, Statute, Provision, Case Number, etc.).
3. **Phase 3 (PEFT):** Multi-task legal reasoning and sentiment/rhetorical classification datasets.
4. **Phase 4 (CJPE):** Court judgment prediction benchmark comprising 994 balanced development judgments (497 class 0, 497 class 1) and 1,517 blind test judgments, processed via sliding-window tokens (from an initial candidate pool of 2,471 train and 618 validation cases).
5. **Phase 5 (LSI):** 5,120 annotated cases covering 100 focal Indian statutory penal sections.
6. **Phase 6 (Summarization):** Long-document judgment texts paired with headnotes and editorial summaries.
7. **Phase 7 (PCR):** 237 test legal queries paired with precedent candidate document pools.

---

## Dataset Acquisition & Kaggle Attachment

Because redistribution of full raw training corpora is restricted by competition guidelines:
1. In Kaggle, attach the official course dataset `cs781-legal-dataset` to `/kaggle/input/`.
2. For local execution, set the `CS781_DATA_DIR` environment variable to your local dataset directory:
   ```bash
   export CS781_DATA_DIR="/path/to/downloaded/dataset"
   ```
