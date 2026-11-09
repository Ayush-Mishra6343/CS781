"""
Training loop, optimization, early stopping, and development evaluation for Phase 4 V4 LoRA.
Preserves the exact notebook optimization, learning rates, schedules, and model-selection rules.
"""

import json
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import CaseClassifier, save_adapter_checkpoint
from src.data import TrainDataset
from src.predict import scores, predict_windows


def train_phase4(
    model: CaseClassifier,
    train_rows: List[Dict[str, Any]],
    train_windows: List[np.ndarray],
    dev_rows: List[Dict[str, Any]],
    dev_windows: List[np.ndarray],
    pad_token_id: int,
    output_dir: str,
    device: torch.device,
    batch_cases: int = 8,
    grad_accum_steps: int = 2,
    lora_lr: float = 1e-4,
    head_lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_epochs: int = 4,
    patience: int = 2,
    warmup_ratio: float = 0.03,
    max_grad_norm: float = 1.0,
    progress_every: int = 50,
    infer_chunk_batch: int = 16,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    chunk_tokens: int = 512,
    max_chunks: int = 8,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], int, float]:
    """
    Executes full training of LoRA adapters and task classification head.
    Evaluates on dev set every epoch and applies early stopping on dev macro-F1.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if progress_fn:
            progress_fn(msg)
        else:
            print(msg, flush=True)

    train_ds = TrainDataset(train_rows, train_windows, pad_token_id=pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_cases,
        shuffle=True,
        num_workers=0,
        pin_memory=use_amp,
    )

    # Class weights from training labels
    counts = np.bincount([r["label"] for r in train_rows], minlength=2)
    class_weights = torch.tensor(
        len(train_rows) / (2.0 * np.maximum(counts, 1)),
        dtype=torch.float,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Separate parameter groups for LoRA weights vs Head
    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and (".A" in n or ".B" in n)
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("head.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lora_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    total_updates = math.ceil(len(train_loader) / grad_accum_steps) * max_epochs
    warmup_steps = max(1, int(total_updates * warmup_ratio))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return min((step + 1) / warmup_steps, 1.0)
        return max(0.0, (total_updates - step) / (total_updates - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    history = []
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    update = 0

    log(f"Starting training: {max_epochs} epochs, {len(train_loader)} batches/epoch, grad_accum={grad_accum_steps}")
    log(f"LoRA params: {sum(p.numel() for p in lora_params):,}, Head params: {sum(p.numel() for p in head_params):,}")
    log(f"Class weights: {class_weights.cpu().tolist()}")

    for epoch in range(max_epochs):
        train_ds.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        seen = 0
        pending = 0
        epoch_started = time.time()

        log(f"\n--- Epoch {epoch + 1}/{max_epochs} ---")
        for step, (ids, mask, y) in enumerate(train_loader):
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device, dtype=torch.long)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(ids, mask)
                loss = criterion(logits, y)

            scaler.scale(loss / grad_accum_steps).backward()
            pending += 1
            loss_sum += float(loss.detach()) * len(y)
            seen += len(y)

            if step == 0 or (step + 1) % progress_every == 0 or step + 1 == len(train_loader):
                elapsed = time.time() - epoch_started
                rate = (step + 1) / max(elapsed, 1e-6)
                eta = (len(train_loader) - step - 1) / max(rate, 1e-6)
                log(
                    f"Epoch {epoch + 1}/{max_epochs}: batch {step + 1}/{len(train_loader)} | "
                    f"running loss {loss_sum / max(seen, 1):.4f} | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m"
                )

            if pending == grad_accum_steps or step == len(train_loader) - 1:
                if pending < grad_accum_steps:
                    correction = grad_accum_steps / pending
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(correction)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update += 1
                pending = 0

        # Development evaluation
        log(f"Evaluating development set after epoch {epoch + 1}...")
        dev_logits, dev_probs, dev_pred = predict_windows(
            model=model,
            rows=dev_rows,
            windows=dev_windows,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=infer_chunk_batch,
            mode="all",
            threshold=0.5,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            progress_fn=log,
            progress_every=progress_every,
        )
        dev_y = [r["label"] for r in dev_rows]
        dev_metrics = scores(dev_y, dev_pred)
        train_epoch_loss = loss_sum / max(seen, 1)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_epoch_loss,
            "dev": dev_metrics,
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(record)

        log(
            f"Epoch {epoch + 1}/{max_epochs} Summary: train loss {train_epoch_loss:.4f} | "
            f"dev macro-F1 {dev_metrics['macro_f1']:.4f} | accuracy {dev_metrics['accuracy']:.4f}"
        )

        # Model selection on dev macro-F1
        best_pt_path = out_dir / "v4_lora_best.pt"
        if dev_metrics["macro_f1"] > best_f1 or (
            dev_metrics["macro_f1"] == best_f1 and train_epoch_loss < best_loss
        ):
            best_f1 = dev_metrics["macro_f1"]
            best_loss = train_epoch_loss
            best_epoch = epoch + 1
            stale = 0
            log(f"New best model at epoch {best_epoch} with dev macro-F1 {best_f1:.4f}! Saving checkpoint...")
            save_adapter_checkpoint(
                model=model,
                save_path=str(best_pt_path),
                epoch=best_epoch,
                dev_macro_f1=best_f1,
                rank=lora_rank,
                alpha=lora_alpha,
                chunk_tokens=chunk_tokens,
                max_chunks=max_chunks,
            )
        else:
            stale += 1
            log(f"Dev macro-F1 did not improve. Stale count: {stale}/{patience}")

        (out_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        if stale >= patience:
            log("Early stopping triggered: development macro-F1 did not improve.")
            break

    log(f"Training completed. Best epoch: {best_epoch}, best dev macro-F1: {best_f1:.4f}")
    return history, best_epoch, best_f1
