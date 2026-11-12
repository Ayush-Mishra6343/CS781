"""Phase 01: Legal Language Model Pre-Training CLI Entry Point.

Initializes the HybridLegalLM architecture and executes pretraining if a tokenized
binary dataset is provided, or validates model parameter initialization.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.phase_01.model import HybridLegalLM, ModelConfig


class LegalTokenDataset(Dataset):
    """Memory-mapped dataset of sequential token blocks for causal LM pretraining."""

    def __init__(self, token_path: str, block_size: int = 1024, stride: int = 1024):
        self.token_path = token_path
        self.block_size = block_size
        self.stride = stride
        file_size_bytes = os.path.getsize(token_path)
        self.num_tokens = file_size_bytes // 4  # uint32 tokens
        self.tokens = np.memmap(token_path, dtype=np.uint32, mode="r", shape=(self.num_tokens,))

    def __len__(self):
        return max(0, 1 + (self.num_tokens - self.block_size - 1) // self.stride)

    def __getitem__(self, index):
        start = index * self.stride
        end = start + self.block_size + 1
        chunk = self.tokens[start:end].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])
        return {"input_ids": input_ids, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="Phase 01: Pretrain HybridLegalLM")
    parser.add_argument("--config", type=str, default="configs/phase_01/config.json", help="Path to configuration JSON")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("CS781_DATA_DIR", "./data"), help="Path to tokenized legal data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase_01", help="Directory to save model checkpoints")
    parser.add_argument("--steps", type=int, default=100, help="Number of pretraining steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=== CS781 Phase 01: Legal Language Model Pretraining ===")
    print(f"Device: {args.device} | Config: {args.config}")
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = ModelConfig(max_seq_len=args.max_seq_len)
    if os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as cf:
                data = json.load(cf)
                cfg.vocab_size = data.get("vocab_size", cfg.vocab_size)
                cfg.n_embd = data.get("hidden_dim", cfg.n_embd)
                cfg.n_layer = data.get("n_layer", cfg.n_layer)
        except Exception as e:
            print(f"Notice: Could not parse {args.config}: {e}")

    model = HybridLegalLM(
        vocab_size=cfg.vocab_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        max_seq_len=cfg.max_seq_len,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {total_params:,} parameters ({trainable_params:,} trainable)")

    token_candidates = [
        os.path.join(args.data_dir, "train_tokens.bin"),
        os.path.join(args.data_dir, "train.bin"),
        os.path.join("data", "phase_01", "train_tokens.bin"),
    ]
    token_file = next((p for p in token_candidates if os.path.exists(p)), None)

    if token_file:
        print(f"Found tokenized dataset at: {token_file}")
        dataset = LegalTokenDataset(token_file, block_size=args.max_seq_len, stride=args.max_seq_len)
        if len(dataset) == 0:
            print("[NOTICE] Dataset contains insufficient tokens for sequence length. Exiting.")
            return

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        model.train()

        step = 0
        running_loss = 0.0
        start_time = time.time()
        print(f"Executing pretraining for {args.steps} steps...")

        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)

            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(input_ids, targets=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            running_loss += loss.item()

            if step % 20 == 0 or step == args.steps:
                elapsed = time.time() - start_time
                avg_loss = running_loss / max(1, step)
                print(f"  Step {step}/{args.steps} | Loss: {avg_loss:.4f} | Elapsed: {elapsed:.1f}s")

            if step >= args.steps:
                break

        ckpt_path = os.path.join(args.output_dir, "hybrid_legal_lm_step.pt")
        torch.save({"step": step, "model_state_dict": model.state_dict()}, ckpt_path)
        print(f"Pretraining steps complete. Saved checkpoint to: {ckpt_path}")
    else:
        print(f"\n[NOTE] Tokenized training data ('train_tokens.bin') not found in '{args.data_dir}'.")
        print("To run pretraining, place tokenized memmap binary in the data directory.")
        print("Model architecture initialization and parameter verification complete.")


if __name__ == "__main__":
    main()
