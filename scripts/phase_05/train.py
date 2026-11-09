"""Training Loop and Optimization Pipeline for Version 1 HLG-SwiGLU-LoRMA.

Implements:
1. Mixed precision training (AMP fp16) with GradScaler.
2. Microbatching and gradient accumulation to respect 8 GB VRAM.
3. Resumable atomic checkpointing (last_checkpoint.pt and best_checkpoint.pt).
4. Validation Macro-F1, Micro-F1, Subset Accuracy, and threshold tracking per epoch.
5. GPU memory monitoring and execution logging.
"""

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import compute_lsi_metrics, find_best_global_threshold
from .utils import get_gpu_memory_stats, save_checkpoint_atomic


class LSITrainer:
    """End-to-End Trainer for Phase 5 Legal Statute Identification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        output_dir: str,
        scheduler: Optional[Any] = None,
        grad_accum_steps: int = 16,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = output_dir
        self.scheduler = scheduler
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.logs_dir = os.path.join(output_dir, "logs")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.best_macro_f1 = -1.0
        self.best_epoch = -1

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        num_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        start_time = time.time()
        for step, batch in enumerate(pbar):
            w_ids = batch["window_ids"].to(self.device, non_blocking=True)
            t_mask = batch["token_mask"].to(self.device, non_blocking=True)
            w_mask = batch["window_mask"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(w_ids, t_mask, w_mask)
                loss = self.criterion(logits, targets)
                loss_scaled = loss / self.grad_accum_steps

            self.scaler.scale(loss_scaled).backward()
            total_loss += loss.item()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        self.model.eval()
        all_probs = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc="[Validation]", leave=False)
        for batch in pbar:
            w_ids = batch["window_ids"].to(self.device, non_blocking=True)
            t_mask = batch["token_mask"].to(self.device, non_blocking=True)
            w_mask = batch["window_mask"].to(self.device, non_blocking=True)
            targets = batch["targets"]

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(w_ids, t_mask, w_mask)
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

        y_probs = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        # Baseline metrics at 0.5 threshold
        base_metrics = compute_lsi_metrics(y_true, y_probs, threshold=0.5)

        # Calibrated global threshold
        best_t, best_macro, cal_metrics = find_best_global_threshold(y_true, y_probs)

        result = {
            "val_loss": 0.0,
            "raw_macro_f1_0_5": base_metrics["macro_f1"],
            "raw_micro_f1_0_5": base_metrics["micro_f1"],
            "raw_subset_accuracy_0_5": base_metrics["subset_accuracy"],
            "calibrated_threshold": best_t,
            "calibrated_macro_f1": best_macro,
            "calibrated_micro_f1": cal_metrics["micro_f1"],
            "calibrated_subset_accuracy": cal_metrics["subset_accuracy"],
            "per_label_f1": cal_metrics["per_label_f1"],
            "per_label_support": cal_metrics["per_label_support"],
            "empty_prediction_rate": cal_metrics["empty_prediction_rate"],
        }
        return result, y_probs, y_true

    def fit(self, num_epochs: int) -> List[Dict[str, Any]]:
        epoch_csv_path = os.path.join(self.metrics_dir, "epoch_metrics.csv")
        with open(epoch_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "calibrated_threshold",
                "calibrated_macro_f1",
                "calibrated_micro_f1",
                "subset_accuracy",
                "empty_prediction_rate",
                "peak_vram_mb",
                "duration_seconds",
                "is_best",
            ])

        print(f"\n{'='*70}\nStarting Training: {num_epochs} Epochs on {self.device}\n{'='*70}")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            train_loss = self.train_epoch(epoch)
            val_metrics, val_probs, val_true = self.evaluate()
            epoch_duration = round(time.time() - epoch_start, 2)
            mem_stats = get_gpu_memory_stats()
            peak_vram = mem_stats.get("max_allocated_mb", 0.0)

            macro_f1 = val_metrics["calibrated_macro_f1"]
            micro_f1 = val_metrics["calibrated_micro_f1"]
            thresh = val_metrics["calibrated_threshold"]
            subset_acc = val_metrics["calibrated_subset_accuracy"]

            is_best = macro_f1 > self.best_macro_f1
            if is_best:
                self.best_macro_f1 = macro_f1
                self.best_epoch = epoch

                # Save best checkpoint
                best_ckpt_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
                meta = {
                    "epoch": epoch,
                    "best_macro_f1": macro_f1,
                    "calibrated_threshold": thresh,
                    "micro_f1": micro_f1,
                    "subset_accuracy": subset_acc,
                    "selection_reason": "Highest held-out development Macro-F1",
                }
                save_checkpoint_atomic(
                    state_dict={
                        "model_state_dict": self.model.state_dict(),
                        "epoch": epoch,
                        "macro_f1": macro_f1,
                        "threshold": thresh,
                    },
                    filepath=best_ckpt_path,
                    metadata=meta,
                )

            # Save last checkpoint
            last_ckpt_path = os.path.join(self.checkpoint_dir, "last_checkpoint.pt")
            save_checkpoint_atomic(
                state_dict={
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "epoch": epoch,
                    "macro_f1": macro_f1,
                },
                filepath=last_ckpt_path,
            )

            # Log to CSV
            with open(epoch_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss:.4f}",
                    f"{thresh:.3f}",
                    f"{macro_f1:.4f}",
                    f"{micro_f1:.4f}",
                    f"{subset_acc:.4f}",
                    f"{val_metrics['empty_prediction_rate']:.4f}",
                    f"{peak_vram:.1f}",
                    f"{epoch_duration:.1f}",
                    is_best,
                ])

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1": macro_f1,
                "val_micro_f1": micro_f1,
                "val_threshold": thresh,
                "subset_accuracy": subset_acc,
                "peak_vram_mb": peak_vram,
                "duration_seconds": epoch_duration,
                "is_best": is_best,
                "per_label_f1": val_metrics["per_label_f1"],
                "per_label_support": val_metrics["per_label_support"],
            }
            self.history.append(record)

            best_tag = " [NEW BEST]" if is_best else ""
            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Dev Macro-F1: {macro_f1:.4f} @ Thr {thresh:.2f} | "
                f"Micro-F1: {micro_f1:.4f} | Peak VRAM: {peak_vram:.0f} MB | Time: {epoch_duration:.1f}s{best_tag}"
            )

        print(f"\nTraining Complete. Best Dev Macro-F1 = {self.best_macro_f1:.4f} at Epoch {self.best_epoch}.")
        return self.history
