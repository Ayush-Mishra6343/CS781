# Local Workstation Setup Guide

This document describes setting up a local Python environment for running the models, scripts, and unit tests.

---

## 1. Prerequisites
- Linux (Ubuntu 20.04+ recommended) or Windows 10/11 with WSL2 / PowerShell.
- Python 3.10 or 3.11.
- NVIDIA GPU with CUDA 11.8+ or 12.1+ and at least 8 GB VRAM.

---

## 2. Conda Environment Setup
```bash
conda create -n cs781 python=3.11 -y
conda activate cs781

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install repository dependencies
pip install -r requirements.txt
```

---

## 3. Verify Installation
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m unittest discover tests/
```
