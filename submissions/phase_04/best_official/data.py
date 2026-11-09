"""
Data loading, BPE tokenization, window extraction, and PyTorch dataset definitions.
Preserves the exact preprocessing, deduplication, and windowing strategy from Phase 4 V4.
"""

import json
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset


class ByteLevelBPE25k:
    """Byte-level BPE tokenizer with 25,000 vocabulary size."""
    def __init__(self, itos: List[str], stoi: Dict[str, int], merge_to_id: Dict[Tuple[int, int], int], merges: List[List[str]]):
        self.itos = itos
        self.stoi = stoi
        self.merge_to_id = merge_to_id
        self.merges = merges
        self.vocab_size = len(itos)
        self.pad_token_id = stoi["<pad>"]
        self.bos_token_id = stoi["<s>"]
        self.eos_token_id = stoi["</s>"]
        self.unk_token_id = stoi["<unk>"]
        self.word_boundary_id = stoi["<wb>"]

    @lru_cache(maxsize=200000)
    def _encode_word(self, word: str) -> List[int]:
        seq = [4 + b for b in word.encode("utf-8")] + [self.word_boundary_id]
        while len(seq) > 1:
            best_pos = None
            best_id = None
            for i in range(len(seq) - 1):
                new_id = self.merge_to_id.get((seq[i], seq[i + 1]))
                if new_id is not None and (best_id is None or new_id < best_id):
                    best_id = new_id
                    best_pos = i
            if best_pos is None:
                break
            seq = seq[:best_pos] + [best_id] + seq[best_pos + 2:]
        return seq

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.bos_token_id] if add_bos else []
        for word in text.split():
            ids.extend(self._encode_word(word))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def _to_bytes(self, tid: int) -> bytes:
        tok = self.itos[tid]
        if tok.startswith("<0x") and tok.endswith(">"):
            return bytes([int(tok[3:-1], 16)])
        if tok == "<wb>":
            return b" "
        if tok in {"<pad>", "<unk>", "<s>", "</s>"}:
            return b""
        if tok.startswith("<bpe_"):
            j = tid - 261
            if 0 <= j < len(self.merges):
                a, b = self.merges[j]
                return self._to_bytes(self.stoi[a]) + self._to_bytes(self.stoi[b])
        return b""

    def decode(self, ids: List[int]) -> str:
        out = bytearray()
        for tid in ids:
            if 0 <= int(tid) < self.vocab_size:
                out.extend(self._to_bytes(int(tid)))
        return bytes(out).decode("utf-8", errors="replace")


def load_tokenizer(bpe_path: str) -> ByteLevelBPE25k:
    """Loads and validates the 25k BPE vocabulary from JSON."""
    with open(bpe_path, "r", encoding="utf-8") as f:
        bpe = json.load(f)

    bpe_itos = bpe["itos"]
    bpe_merges = bpe["merges"]
    bpe_stoi = {t: i for i, t in enumerate(bpe_itos)}

    if bpe["vocab_size"] != 25000 or len(bpe_itos) != 25000:
        raise ValueError(f"Expected V4 BPE-25k vocabulary size 25000, got {len(bpe_itos)}")

    merge_to_id = {}
    for i, (left, right) in enumerate(bpe_merges):
        merge_to_id[(bpe_stoi[left], bpe_stoi[right])] = 261 + i

    tokenizer = ByteLevelBPE25k(bpe_itos, bpe_stoi, merge_to_id, bpe_merges)

    # Standard notebook probe assertion
    probe = "The Supreme Court of India has jurisdiction over constitutional matters."
    decoded = tokenizer.decode(tokenizer.encode(probe))
    assert decoded.rstrip() == probe, f"Tokenizer probe failed: {decoded!r} vs {probe!r}"
    return tokenizer


def load_cases(path: str, labelled: bool) -> List[Dict[str, Any]]:
    """
    Loads JSON cases, validates schema and labels, and deduplicates identical records.
    If labelled=False (test set), only 'id' and 'text' are retained; all extra fields (including any labels) are discarded.
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise TypeError(f"{path}: expected JSON list")

    result = []
    seen = {}
    duplicate_count = 0

    for i, r in enumerate(rows):
        if not isinstance(r, dict) or "id" not in r or "text" not in r:
            raise ValueError(f"{path}: row {i} lacks id or text")

        cid = str(r["id"])
        text = r["text"]
        if not isinstance(text, str):
            raise TypeError(f"{cid}: text must be string")

        label = r.get("label") if labelled else None
        if labelled and (type(label) is not int or label not in (0, 1)):
            raise ValueError(f"{cid}: label must be integer 0/1")

        normalized = {"id": cid, "text": text}
        if labelled:
            normalized["label"] = label

        if cid in seen:
            if result[seen[cid]] != normalized:
                raise ValueError(f"{path}: conflicting duplicate id {cid} (text or label differs)")
            duplicate_count += 1
            continue

        seen[cid] = len(result)
        result.append(normalized)

    return result


def select_windows(
    tokenizer: ByteLevelBPE25k,
    text: str,
    chunk_tokens: int = 512,
    chunk_stride: int = 384,
    max_chunks: int = 8,
) -> np.ndarray:
    """
    Extracts up to max_chunks overlapping windows of chunk_tokens, spaced across the document.
    """
    ids = tokenizer.encode(text, add_bos=True)
    if not ids:
        ids = [tokenizer.bos_token_id]

    if len(ids) <= chunk_tokens:
        starts = [0]
    else:
        all_starts = list(range(0, len(ids) - chunk_tokens + 1, chunk_stride))
        if all_starts[-1] != len(ids) - chunk_tokens:
            all_starts.append(len(ids) - chunk_tokens)
        take = min(max_chunks, len(all_starts))
        starts = sorted(set(np.linspace(0, len(all_starts) - 1, take).round().astype(int).tolist()))
        starts = [all_starts[i] for i in starts]

    chunks = []
    for start in starts:
        x = ids[start : start + chunk_tokens]
        chunks.append(x + [tokenizer.pad_token_id] * (chunk_tokens - len(x)))

    return np.asarray(chunks, dtype=np.int32)


def encode_rows(
    tokenizer: ByteLevelBPE25k,
    rows: List[Dict[str, Any]],
    name: str = "dataset",
    chunk_tokens: int = 512,
    chunk_stride: int = 384,
    max_chunks: int = 8,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> List[np.ndarray]:
    """Pre-encodes all rows into document window arrays."""
    counts = []
    chunks = []
    total = len(rows)
    for i, row in enumerate(rows):
        c = select_windows(
            tokenizer,
            row["text"],
            chunk_tokens=chunk_tokens,
            chunk_stride=chunk_stride,
            max_chunks=max_chunks,
        )
        chunks.append(c)
        counts.append(len(c))
        if (i + 1) % 5000 == 0 or (i + 1) == total:
            msg = f"{name} encoding: {i + 1}/{total} documents ({100 * (i + 1) / total:.1f}%)"
            if progress_fn:
                progress_fn(msg)
            else:
                print(msg, flush=True)

    summary_msg = f"{name} total windows: {sum(counts):,}, documents: {total:,}, median windows: {int(np.median(counts))}"
    if progress_fn:
        progress_fn(summary_msg)
    else:
        print(summary_msg, flush=True)

    return chunks


class TrainDataset(Dataset):
    """
    Dataset for training. Rotates the active document window per epoch to control compute.
    """
    def __init__(self, rows: List[Dict[str, Any]], windows: List[np.ndarray], pad_token_id: int):
        self.rows = rows
        self.windows = windows
        self.pad_token_id = pad_token_id
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        ws = self.windows[i]
        idx = (self.epoch + i) % len(ws)
        x = ws[idx]
        mask = (x != self.pad_token_id)
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            self.rows[i]["label"],
        )


class EvalDataset(Dataset):
    """Dataset for development or test evaluation."""
    def __init__(self, rows: List[Dict[str, Any]], windows: List[np.ndarray]):
        self.rows = rows
        self.windows = windows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, str, Optional[int]]:
        return self.windows[i], self.rows[i]["id"], self.rows[i].get("label")
