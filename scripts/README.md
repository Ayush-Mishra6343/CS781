# Research Scripts & Pipeline Entry Points

This directory contains command-line entry points and recovered standalone research scripts across the seven project phases.

## Execution Architecture

- **Primary Research Workflows:** All end-to-end training routines, architectural ablations, context window calibrations, and official submission generation are self-contained within the research notebooks in [`notebooks/`](../notebooks/).
- **Standalone Execution Scripts:** 
  - `phase_05/train.py`: Recovered standalone modular training script for Phase 5 (SwiGLU-LoRMA / HLG multi-label statute identification).
  - `phase_07/infer.py`: Recovered standalone inference and validation script for Phase 7 (Track E6 Hybrid Reranker test prediction and submission packaging).
  - `phase_01/` through `phase_04/`, `phase_06/`: Standard CLI entry points and invocation scaffolds.
