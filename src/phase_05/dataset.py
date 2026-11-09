"""Dataset and Long-Document Windowing for Phase 5 Legal Statute Identification.

Implements:
1. Deterministic window coverage policy (beginning, middle, end) for long legal judgments.
2. Binary multi-hot target construction with row-level duplicate-ID deduplication and logging.
3. Fast token-caching to NVMe/SSD on D: drive for high throughput.
4. Clean diagnostic views and group-aware text-hash disjoint holdouts.
"""

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import ByteLevelBPE25k


def segment_tokens_into_windows(
    token_ids: List[int],
    window_len: int = 512,
    max_windows: int = 4,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Applies the deterministic multi-window policy.

    Always samples beginning and end, with intermediate windows spaced uniformly.
    Returns:
    - window_ids: [max_windows, window_len] (torch.long)
    - token_mask: [max_windows, window_len] (torch.float, 1 for valid, 0 for pad)
    - window_mask: [max_windows] (torch.float, 1 for valid window, 0 for pad)
    - coverage_ratio: fraction of distinct tokens covered
    """
    total_tokens = len(token_ids)
    if total_tokens == 0:
        token_ids = [pad_token_id]
        total_tokens = 1

    if total_tokens <= window_len:
        # Document fits entirely in one window
        num_valid_windows = 1
        starts = [0]
    else:
        # Long document: sample up to max_windows starting positions
        if max_windows == 1:
            num_valid_windows = 1
            starts = [0]
        else:
            num_valid_windows = min(max_windows, (total_tokens + window_len - 1) // window_len)
            max_start = total_tokens - window_len
            if num_valid_windows == 2:
                starts = [0, max_start]
            else:
                step = max_start / (num_valid_windows - 1)
                starts = [int(round(i * step)) for i in range(num_valid_windows)]
                # Ensure strictly non-decreasing and within bounds
                starts = sorted(list(set(min(max(s, 0), max_start) for s in starts)))
                num_valid_windows = len(starts)

    window_ids = torch.full((max_windows, window_len), pad_token_id, dtype=torch.long)
    token_mask = torch.zeros((max_windows, window_len), dtype=torch.float32)
    window_mask = torch.zeros(max_windows, dtype=torch.float32)

    covered_indices = set()
    for w_idx, s in enumerate(starts):
        chunk = token_ids[s : s + window_len]
        L = len(chunk)
        window_ids[w_idx, :L] = torch.tensor(chunk, dtype=torch.long)
        token_mask[w_idx, :L] = 1.0
        window_mask[w_idx] = 1.0
        covered_indices.update(range(s, s + L))

    coverage_ratio = len(covered_indices) / max(total_tokens, 1)
    return window_ids, token_mask, window_mask, coverage_ratio


class LSIDataset(Dataset):
    """Multi-label dataset for Phase 5 Legal Statute Identification."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: ByteLevelBPE25k,
        window_len: int = 512,
        max_windows: int = 4,
        num_labels: int = 100,
        is_test: bool = False,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.window_len = window_len
        self.max_windows = max_windows
        self.num_labels = num_labels
        self.is_test = is_test

        self.duplicate_label_rows = []
        if not is_test:
            for r in records:
                raw_labs = r.get("labels", [])
                if len(raw_labs) != len(set(raw_labs)):
                    self.duplicate_label_rows.append(r["id"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        doc_id = str(item["id"])
        text = item.get("text", "")

        # Tokenize text
        token_ids = self.tokenizer.encode(text)

        # Multi-window segmentation
        w_ids, t_mask, w_mask, cov = segment_tokens_into_windows(
            token_ids=token_ids,
            window_len=self.window_len,
            max_windows=self.max_windows,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        out = {
            "id": doc_id,
            "window_ids": w_ids,
            "token_mask": t_mask,
            "window_mask": w_mask,
            "coverage_ratio": cov,
            "num_tokens": len(token_ids),
        }

        if not self.is_test:
            raw_labels = item.get("labels", [])
            # Deduplicate labels for multi-hot target
            unique_labels = sorted(set(int(l) for l in raw_labels if 0 <= int(l) < self.num_labels))
            target = torch.zeros(self.num_labels, dtype=torch.float32)
            if unique_labels:
                target[unique_labels] = 1.0
            out["target"] = target
            out["gold_list"] = unique_labels

        return out


def collate_lsi_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates individual items into padded batch tensors."""
    ids = [x["id"] for x in batch]
    window_ids = torch.stack([x["window_ids"] for x in batch], dim=0)  # [B, W, L]
    token_mask = torch.stack([x["token_mask"] for x in batch], dim=0)  # [B, W, L]
    window_mask = torch.stack([x["window_mask"] for x in batch], dim=0)  # [B, W]
    coverages = [x["coverage_ratio"] for x in batch]

    out = {
        "ids": ids,
        "window_ids": window_ids,
        "token_mask": token_mask,
        "window_mask": window_mask,
        "coverage_ratio": coverages,
    }

    if "target" in batch[0]:
        targets = torch.stack([x["target"] for x in batch], dim=0)  # [B, 100]
        out["targets"] = targets
        out["gold_lists"] = [x["gold_list"] for x in batch]

    return out


def create_group_aware_local_holdout(
    train_records: List[Dict[str, Any]],
    holdout_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits train records into training subset and local holdout grouped by canonical text hash.

    Guarantees zero text-duplicate leakage between the local train and local holdout!
    """
    import random
    rng = random.Random(seed)

    # Group record indices by canonical text hash
    hash_to_indices = {}
    for i, r in enumerate(train_records):
        h = hashlib.sha256(" ".join(r["text"].split()).encode("utf-8")).hexdigest()
        hash_to_indices.setdefault(h, []).append(i)

    all_hashes = list(hash_to_indices.keys())
    rng.shuffle(all_hashes)

    target_holdout_rows = int(len(train_records) * holdout_ratio)
    holdout_indices = []
    train_indices = []

    current_holdout_count = 0
    for h in all_hashes:
        group = hash_to_indices[h]
        if current_holdout_count + len(group) <= target_holdout_rows:
            holdout_indices.extend(group)
            current_holdout_count += len(group)
        else:
            train_indices.extend(group)

    local_train = [train_records[i] for i in train_indices]
    local_holdout = [train_records[i] for i in holdout_indices]
    return local_train, local_holdout
