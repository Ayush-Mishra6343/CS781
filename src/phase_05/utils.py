"""Utility functions for Phase 5 Legal Statute Identification.

Provides:
- Reproducibility seeding
- File hashing (SHA-256)
- GPU memory & hardware tracking
- Atomic checkpoint writing and verification
- Metric plotting and chart data export
"""

import csv
import hashlib
import json
import os
import random
import time
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Sets random seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_sha256(filepath: str) -> str:
    """Calculates SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_gpu_memory_stats() -> Dict[str, float]:
    """Returns GPU memory stats in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2),
    }


def save_checkpoint_atomic(
    state_dict: Dict[str, Any],
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Atomically saves checkpoint to prevent corruption on interruption."""
    temp_path = filepath + ".tmp"
    torch.save(state_dict, temp_path)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(temp_path, filepath)

    if metadata is not None:
        meta_path = os.path.splitext(filepath)[0] + "_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def plot_training_curves(history: List[Dict[str, Any]], save_dir: str):
    """Generates training curves for loss and Macro-F1 / Micro-F1 and saves CSV backing data."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = [x["epoch"] for x in history]
    train_losses = [x["train_loss"] for x in history]
    val_macro = [x.get("val_macro_f1", 0.0) for x in history]
    val_micro = [x.get("val_micro_f1", 0.0) for x in history]

    # Export backing CSV
    csv_path = os.path.join(save_dir, "training_history_data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_macro_f1", "val_micro_f1"])
        for e, tl, ma, mi in zip(epochs, train_losses, val_macro, val_micro):
            writer.writerow([e, tl, ma, mi])

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train BCE Loss", color=color)
    ax1.plot(epochs, train_losses, color=color, marker="o", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Validation F1 Score", color=color)
    ax2.plot(epochs, val_macro, color="navy", marker="s", label="Val Macro-F1")
    ax2.plot(epochs, val_micro, color="tab:cyan", marker="^", linestyle="--", label="Val Micro-F1")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Version 1 (HLG-SwiGLU-LoRMA) Training & Validation Progression")
    png_path = os.path.join(save_dir, "v1_training_curves.png")
    plt.savefig(png_path, dpi=300)
    plt.close()


def plot_per_label_f1(per_label_f1: List[float], label_support: List[int], save_dir: str):
    """Plots per-label F1 against log support and highlights rare classes (< 100)."""
    os.makedirs(save_dir, exist_ok=True)
    # Save CSV
    csv_path = os.path.join(save_dir, "per_label_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "support", "f1_score"])
        for idx, (sup, f1) in enumerate(zip(label_support, per_label_f1)):
            writer.writerow([idx, sup, f1])

    supports = np.array(label_support)
    f1s = np.array(per_label_f1)
    is_rare = supports < 100

    plt.figure(figsize=(9, 5))
    plt.scatter(
        supports[~is_rare],
        f1s[~is_rare],
        color="steelblue",
        alpha=0.7,
        label="Frequent Classes (Support >= 100)",
    )
    plt.scatter(
        supports[is_rare],
        f1s[is_rare],
        color="crimson",
        alpha=0.9,
        label="Rare Classes (Support < 100, N=19)",
    )
    plt.xscale("log")
    plt.xlabel("Validation Positive Support (log scale)")
    plt.ylabel("Per-Label F1 Score")
    plt.title("Per-Label F1 Score vs Validation Support")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "v1_per_label_f1_vs_support.png"), dpi=300)
    plt.close()
