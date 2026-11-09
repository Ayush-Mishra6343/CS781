"""Model Architecture Definitions for Phase 5 Legal Statute Identification (LSI).

Contains:
1. Exact original Phase 1 V1LegalLM components (RMSNorm, RotaryEmbedding, GQAAttention,
   FeedForward, TransformerBlock, V1LegalLM).
2. Phase 5 Hierarchical Components:
   - WindowDualPooling (Masked mean + learned attention pooling)
   - CrossWindowTransformer (Bidirectional document-level transformer)
   - LabelQueryEvidenceAttention (100 statute queries attending over document windows)
   - HLGSwiGLULoRMALSI (End-to-end hierarchical architecture)
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import apply_lorma_to_v1


# ---------------------------------------------------------------------------
# 1. Exact Phase 1 Backbone Components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization from Phase 1."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(rms + self.eps))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from Phase 1."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        required_len = position_offset + seq_len
        if required_len > self.max_seq_len:
            self._build_cache(required_len)
            self.max_seq_len = required_len

        cos = self.cos_cached[position_offset : position_offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[position_offset : position_offset + seq_len].unsqueeze(0).unsqueeze(0)
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        return q, k

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        cos_full = torch.repeat_interleave(cos, 2, dim=-1)
        sin_full = torch.repeat_interleave(sin, 2, dim=-1)
        return x * cos_full + rotated * sin_full


class GQAAttention(nn.Module):
    """Grouped-Query Attention from Phase 1."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.kv_repeat = n_head // n_kv_head

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.rope = RotaryEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        offset = past_key_value[0].size(-2) if past_key_value is not None else 0
        q, k = self.rope(q, k, position_offset=offset)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        present = (k, v) if use_cache else None

        if self.kv_repeat > 1:
            k = torch.repeat_interleave(k, self.kv_repeat, dim=1)
            v = torch.repeat_interleave(v, self.kv_repeat, dim=1)

        is_causal = past_key_value is None and T > 1
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y), present


class FeedForward(nn.Module):
    """GELU FeedForward module from Phase 1."""

    def __init__(self, n_embd: int, multiplier: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(n_embd * multiplier)
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Decoder Transformer block from Phase 1."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        max_seq_len: int,
        ffn_multiplier: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attention = GQAAttention(n_embd, n_head, n_kv_head, max_seq_len, dropout)
        self.norm2 = RMSNorm(n_embd)
        self.ffn = FeedForward(n_embd, ffn_multiplier, dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present = self.attention(self.norm1(x), past_key_value, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, present


class V1LegalLM(nn.Module):
    """Phase 1 Pretrained Legal Language Model (113.96M parameters)."""

    def __init__(
        self,
        vocab_size: int = 25000,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_kv_head: int = 4,
        max_seq_len: int = 1024,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, n_kv_head, max_seq_len, ffn_multiplier, dropout)
            for _ in range(n_layer)
        ])
        self.final_norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        x = self.token_embedding(input_ids)
        presents = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, past, use_cache)
            if use_cache:
                presents.append(present)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, presents

    def extract_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extracts contextual token representations before LM head."""
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x, _ = block(x, past_key_value=None, use_cache=False)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# 2. Phase 5 Hierarchical & Label-Query Multi-Label Components
# ---------------------------------------------------------------------------

class WindowDualPooling(nn.Module):
    """Combines masked mean pooling and learned token-attention pooling per window."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, token_reps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        token_reps: [B, L, D]
        mask: [B, L] (1 for valid token, 0 for padding)
        Returns: [B, D] window vector
        """
        mask_f = mask.unsqueeze(-1).to(token_reps.dtype)  # [B, L, 1]
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pooled = (token_reps * mask_f).sum(dim=1) / denom  # [B, D]

        scores = self.attn_proj(token_reps)  # [B, L, 1]
        scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
        attn_weights = F.softmax(scores, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_pooled = (token_reps * attn_weights).sum(dim=1)  # [B, D]

        concat = torch.cat([mean_pooled, attn_pooled], dim=-1)  # [B, 2D]
        return self.fuse_proj(concat)  # [B, D]


class CrossWindowTransformer(nn.Module):
    """Bidirectional Transformer over document windows."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        max_windows: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_windows, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, window_embeds: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        """
        window_embeds: [B, W, D]
        window_mask: [B, W] (1 for valid window, 0 for pad window)
        Returns: [B, W, D] contextualized window representations
        """
        B, W, D = window_embeds.size()
        positions = torch.arange(W, device=window_embeds.device).unsqueeze(0).expand(B, W)
        x = window_embeds + self.pos_emb(positions)
        # In PyTorch TransformerEncoder, src_key_padding_mask=True means IGNORED
        key_padding_mask = ~window_mask.bool()
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.norm(out)


class LabelQueryEvidenceAttention(nn.Module):
    """100 learnable statute queries attending over document window representations."""

    def __init__(self, num_labels: int = 100, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, window_context: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        """
        window_context: [B, W, D]
        window_mask: [B, W] (1 for valid window, 0 for pad window)
        Returns: [B, 100] independent classification logits
        """
        B, W, D = window_context.size()
        queries = self.label_queries.unsqueeze(0).expand(B, self.num_labels, D)  # [B, 100, D]
        key_padding_mask = ~window_mask.bool()  # True means ignore
        attn_out, _ = self.cross_attn(
            query=queries,
            key=window_context,
            value=window_context,
            key_padding_mask=key_padding_mask,
        )
        label_reps = self.norm(queries + attn_out)  # [B, 100, D]
        logits = self.classifier(label_reps).squeeze(-1)  # [B, 100]
        return logits


class HLGSwiGLULoRMALSI(nn.Module):
    """Full End-to-End Hierarchical Local-Global SwiGLU-LoRMA Architecture for LSI."""

    def __init__(
        self,
        backbone: V1LegalLM,
        num_labels: int = 100,
        hidden_dim: int = 768,
        lorma_rank: int = 16,
        lorma_alpha: float = 32.0,
        lorma_dropout: float = 0.05,
        max_windows: int = 16,
        use_label_queries: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_label_queries = use_label_queries

        # Apply activation-space LoRMA with SwiGLU gating
        self.trainable_adapter_params = apply_lorma_to_v1(
            self.backbone,
            target_modules=["q_proj", "v_proj"],
            rank=lorma_rank,
            alpha=lorma_alpha,
            dropout=lorma_dropout,
        )

        # Local window pooling
        self.window_pooling = WindowDualPooling(hidden_dim)

        # Global cross-window context
        self.cross_window_transformer = CrossWindowTransformer(
            hidden_dim=hidden_dim, max_windows=max_windows
        )

        # Multi-label head
        if self.use_label_queries:
            self.head = LabelQueryEvidenceAttention(
                num_labels=num_labels, hidden_dim=hidden_dim
            )
        else:
            # Baseline ablation: uniform mean window pooling
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_labels),
            )

    def forward(
        self, window_input_ids: torch.Tensor, token_mask: torch.Tensor, window_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        window_input_ids: [B, W, L]
        token_mask: [B, W, L] (1 for token, 0 for pad)
        window_mask: [B, W] (1 for window, 0 for pad)
        Returns: [B, 100] statute logits
        """
        B, W, L = window_input_ids.size()
        # Flatten batch and windows to pass through backbone
        flat_input_ids = window_input_ids.view(B * W, L)
        flat_token_mask = token_mask.view(B * W, L)

        # Extract representations from frozen backbone + LoRMA
        flat_token_reps = self.backbone.extract_features(flat_input_ids)  # [B*W, L, D]

        # Pool each window
        flat_window_embeds = self.window_pooling(flat_token_reps, flat_token_mask)  # [B*W, D]
        window_embeds = flat_window_embeds.view(B, W, -1)  # [B, W, D]

        # Bidirectional cross-window contextualization
        window_context = self.cross_window_transformer(window_embeds, window_mask)  # [B, W, D]

        # Multi-label statute logits
        if self.use_label_queries:
            logits = self.head(window_context, window_mask)  # [B, 100]
        else:
            # Masked mean over windows
            w_mask_f = window_mask.unsqueeze(-1).to(window_context.dtype)
            doc_rep = (window_context * w_mask_f).sum(dim=1) / w_mask_f.sum(dim=1).clamp_min(1.0)
            logits = self.head(doc_rep)  # [B, 100]

        return logits
