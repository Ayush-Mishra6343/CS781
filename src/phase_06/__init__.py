"""Phase 06: Legal Document Summarization Package."""

from .tokenizer import ByteLevelBPE25k, load_phase1_tokenizer
from .model import (
    RMSNorm,
    RotaryEmbedding,
    V1GQAAttention,
    WideFeedForward,
    V1DeepTransformerBlock,
    V1DeepLegalLM,
    morph_wide_to_deep,
)
from .mbr_decoder import compute_mbr_consensus, pairwise_f1_similarity, BERTScoreMBR

__all__ = [
    "ByteLevelBPE25k",
    "load_phase1_tokenizer",
    "RMSNorm",
    "RotaryEmbedding",
    "V1GQAAttention",
    "WideFeedForward",
    "V1DeepTransformerBlock",
    "V1DeepLegalLM",
    "morph_wide_to_deep",
    "compute_mbr_consensus",
    "pairwise_f1_similarity",
    "BERTScoreMBR",
]
