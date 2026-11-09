"""Phase 1 BPE Tokenizer Module for Legal Statute Identification (LSI).

Loads and wraps the official 25k Byte-Level BPE tokenizer from Phase 1.
"""

import json
import os
from typing import Dict, List, Optional, Tuple


class ByteLevelBPE25k:
    """Byte-Level BPE tokenizer with 25,000 vocabulary size from Phase 1."""

    def __init__(
        self,
        itos: List[str],
        stoi: Dict[str, int],
        merge_to_id: Dict[Tuple[int, int], int],
        merges: List[List[str]],
    ):
        self.itos = itos
        self.stoi = stoi
        self.merge_to_id = merge_to_id
        self.merges = merges
        self.vocab_size = len(itos)

        self.pad_token_id = stoi["<pad>"]
        self.unk_token_id = stoi["<unk>"]
        self.bos_token_id = stoi["<s>"]
        self.eos_token_id = stoi["</s>"]
        self.word_boundary_id = stoi["<wb>"]

    def _encode_word(self, word: str) -> List[int]:
        sequence = [4 + byte for byte in word.encode("utf-8")]
        sequence.append(self.word_boundary_id)

        while len(sequence) > 1:
            best_position = None
            best_token_id = None

            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                new_token_id = self.merge_to_id.get(pair)
                if new_token_id is None:
                    continue

                if best_token_id is None or new_token_id < best_token_id:
                    best_token_id = new_token_id
                    best_position = i

            if best_position is None:
                break

            i = best_position
            sequence = sequence[:i] + [best_token_id] + sequence[i + 2 :]

        return sequence

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        words = text.split()
        token_ids: List[int] = []

        if add_bos:
            token_ids.append(self.bos_token_id)

        for word in words:
            token_ids.extend(self._encode_word(word))

        if add_eos:
            token_ids.append(self.eos_token_id)

        return token_ids

    def _token_to_bytes(self, token_id: int) -> bytes:
        token = self.itos[token_id]

        if token.startswith("<0x") and token.endswith(">"):
            return bytes([int(token[3:-1], 16)])

        if token == "<wb>":
            return b" "

        if token in {"<pad>", "<unk>", "<s>", "</s>"}:
            return b""

        if token.startswith("<bpe_"):
            merge_index = token_id - 261
            if 0 <= merge_index < len(self.merges):
                left_token, right_token = self.merges[merge_index]
                left_id = self.stoi[left_token]
                right_id = self.stoi[right_token]
                return self._token_to_bytes(left_id) + self._token_to_bytes(right_id)

        return b""

    def decode(self, token_ids: List[int]) -> str:
        output = bytearray()
        for token_id in token_ids:
            if 0 <= int(token_id) < self.vocab_size:
                output.extend(self._token_to_bytes(int(token_id)))
        return bytes(output).decode("utf-8", errors="replace")


def load_phase1_tokenizer(bpe_path: str = r"d:\STORAGE\bpe_en_25k.json") -> ByteLevelBPE25k:
    """Loads and instantiates the verified Phase 1 tokenizer from JSON vocabulary file."""
    if not os.path.exists(bpe_path):
        raise FileNotFoundError(f"Tokenizer vocabulary file not found at: {bpe_path}")

    with open(bpe_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bpe_itos = data["itos"]
    bpe_merges = data["merges"]
    assert len(bpe_itos) == 25000, f"Expected 25,000 vocab tokens, got {len(bpe_itos)}"

    bpe_stoi = {token: idx for idx, token in enumerate(bpe_itos)}
    merge_to_id = {
        (bpe_stoi[m[0]], bpe_stoi[m[1]]): 261 + i for i, m in enumerate(bpe_merges)
    }

    return ByteLevelBPE25k(bpe_itos, bpe_stoi, merge_to_id, bpe_merges)
