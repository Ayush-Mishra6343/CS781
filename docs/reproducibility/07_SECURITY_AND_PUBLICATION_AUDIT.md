# Security and Publication Audit

**Project:** In-House Legal Intelligence System (CS781 Capstone Project)  
**Student:** Ayush Mishra (Roll Number: 250253)  
**Audit Standard:** Part 14 Specification (Secrets, Credentials, Licensing, and File Size Checks)

---

## 1. Secrets and Credentials Scan

All files slated for the public repository were scanned for hardcoded credentials, access tokens, and private identifiers:

| Pattern Scanned | Matches Found | Action Taken | Current Status |
|:---|:---:|:---|:---:|
| HuggingFace User Access Tokens (`hf_[a-zA-Z0-9]{20,}`) | 0 hardcoded | Environment variable fallback (`os.environ.get('HF_TOKEN')`) implemented | **CLEARED** |
| Weights & Biases API Keys (`wandb.login(key=...)`) | 0 hardcoded | Verified absent or abstracted to environment variables | **CLEARED** |
| Kaggle API Secrets / Tokens | 1 call in Phase 1 | Wrapped in `try...except` block with `os.environ` fallback for non-Kaggle portability | **CLEARED** |
| AWS / Cloud Access Keys | 0 | None present in repository | **CLEARED** |
| Database / Private Passwords | 0 | None present in repository | **CLEARED** |

---

## 2. File Size and Git Binary Audit

Standard Git repositories should avoid committing large binary files (>100 MB). An audit of the public repository tree confirmed:

- **Checkpoints > 100 MB:** Completely excluded from Git commits. Checkpoints are documented via SHA-256 manifests, architecture configuration files (`config.json`), and download instructions.
- **Large Dataset Binary Caches (`.memmap`, raw corpus > 100 MB):** Excluded from the public repository; data acquisition scripts and HuggingFace Hub links provided in `data/README.md`.
- **Submission Archives:** Cleaned individual competition ZIP packages (< 1 MB each) are indexed in `results/` for exact reproducibility.

---

## 3. Privacy, Licensing & Restricted Data Audit

- **Court Judgments:** All judgment texts derive from publicly accessible decisions of the Supreme Court of India and High Courts. No sealed or in-camera proceeding records are contained.
- **Personally Identifiable Information (PII):** Case names reflect public legal citations (e.g., *Union of India v. Association for Democratic Reforms*). No sensitive personal data, private addresses, or non-public personal identifiers are included.
- **Evaluation Splits:** Hidden competition test labels are not present in the repository; only test inputs and generated submission prediction files are included.
