# Contributing

This repository contains student coursework developed for the CS781 Capstone Project at IIT Kanpur.

---

## 1. Project Context

- **Author:** Ayush Mishra
- **Institution:** Department of Computer Science & Engineering, IIT Kanpur
- **Course:** CS781: Deep Learning for Natural Language Processing
- **Academic Report:** The submitted capstone report (`report/FINAL_REPORT.pdf`) is permanent and reflects the evaluated academic work.

---

## 2. Development Guidelines

### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Ayush-Mishra6343/CS781.git
   cd CS781
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Code and Notebook Standards
- Code should be clean, readable, and properly documented.
- Notebook paths should remain portable using configuration variables or environment variables (`CS781_DATA_DIR`).
- Do not commit hardcoded credentials, API tokens, or personal paths.
- Run `python -m unittest discover tests/` before submitting changes.
