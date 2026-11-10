"""Phase 6 Neural Architecture & Model Definition.

Stage: Net2Net Structural Depth Growth (12 -> 16 Layers).
Expands the V1WideLegalLM architecture from 12 layers to 16 layers (~158M parameters).
Functions:
- V1DeepLegalLM: 16-layer causal transformer with intermediate FFN dimension 4,352.
- morph_wide_to_deep: Function-preserving weight transfer with identity-block insertion.
  Ensures exact step-0 mathematical identity (max |Delta logit| == 0.0) by zero-initializing
  out_proj, fc2, and bias in the inserted identity blocks.
"""

from typing import Dict, List, Optional, Tuple
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x shape: [batch, heads, seq_len, head_dim]
    return (x * cos.unsqueeze(0).unsqueeze(1)) + (rotate_half(x) * sin.unsqueeze(0).unsqueeze(1))


class V1GQAAttention(nn.Module):
    """Grouped Query Attention (GQA) with RoPE."""

    def __init__(
        self,
        n_embd: int = 768,
        n_head: int = 12,
        n_kv_head: int = 4,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.num_queries_per_kv = n_head // n_kv_head

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(T, x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        present = (k, v) if use_cache else None

        # Repeat KV heads for GQA
        if self.num_queries_per_kv > 1:
            k_expanded = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v_expanded = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        else:
            k_expanded, v_expanded = k, v

        # Scaled dot-product attention
        att = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            is_causal=(past_key_value is None and T > 1),
            dropout_p=0.0,
        )
        out = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), present


class WideFeedForward(nn.Module):
    """Feed-Forward Network with expanded intermediate dimension (4352)."""

    def __init__(self, n_embd: int = 768, intermediate_dim: int = 4352, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, n_embd)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class V1DeepTransformerBlock(nn.Module):
    """Transformer block with RMSNorm, GQA, and Wide FFN."""

    def __init__(
        self,
        n_embd: int = 768,
        n_head: int = 12,
        n_kv_head: int = 4,
        max_seq_len: int = 2048,
        intermediate_dim: int = 4352,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attention = V1GQAAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.norm2 = RMSNorm(n_embd)
        self.ffn = WideFeedForward(
            n_embd=n_embd,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present = self.attention(self.norm1(x), past_key_value=past_key_value, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, present


class V1DeepLegalLM(nn.Module):
    """Depth-Expanded Causal Language Model (16 layers, ~158M parameters)."""

    def __init__(
        self,
        vocab_size: int = 25000,
        n_embd: int = 768,
        n_layer: int = 16,
        n_head: int = 12,
        n_kv_head: int = 4,
        max_seq_len: int = 2048,
        intermediate_dim: int = 4352,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.intermediate_dim = intermediate_dim
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            V1DeepTransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_kv_head=n_kv_head,
                max_seq_len=max_seq_len,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
            )
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
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        x = self.token_embedding(input_ids)
        presents = [] if use_cache else None

        for layer_idx, block in enumerate(self.blocks):
            past = past_key_values[layer_idx] if past_key_values is not None else None
            x, present = block(x, past_key_value=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        if return_hidden_states:
            return logits, presents, x
        return logits, presents


def morph_wide_to_deep(
    wide_model: nn.Module,
    new_n_layer: int = 16,
    identity_insert_indices: List[int] = [3, 7, 11, 15],
) -> V1DeepLegalLM:
    """Performs function-preserving Net2Net depth growth from 12 layers to 16 layers."""
    device = next(wide_model.parameters()).device
    orig_n_layer = len(wide_model.blocks)
    assert orig_n_layer == 12, f"Expected 12 original layers, found {orig_n_layer}"
    assert new_n_layer == 16, f"Expected 16 new layers, found {new_n_layer}"
    assert len(identity_insert_indices) == 4, "Must specify exactly 4 insertion positions"

    deep_model = V1DeepLegalLM(
        vocab_size=getattr(wide_model, "vocab_size", 25000),
        n_embd=getattr(wide_model, "n_embd", 768),
        n_layer=new_n_layer,
        intermediate_dim=getattr(wide_model, "intermediate_dim", 4352),
    ).to(device)

    with torch.no_grad():
        deep_model.token_embedding.weight.copy_(wide_model.token_embedding.weight)
        deep_model.final_norm.weight.copy_(wide_model.final_norm.weight)

        orig_ptr = 0
        for deep_idx in range(new_n_layer):
            deep_block = deep_model.blocks[deep_idx]

            if deep_idx in identity_insert_indices:
                source_orig_idx = max(0, orig_ptr - 1)
                source_block = wide_model.blocks[source_orig_idx]

                deep_block.norm1.weight.copy_(source_block.norm1.weight)
                deep_block.norm2.weight.copy_(source_block.norm2.weight)

                deep_block.attention.q_proj.weight.copy_(source_block.attention.q_proj.weight)
                deep_block.attention.k_proj.weight.copy_(source_block.attention.k_proj.weight)
                deep_block.attention.v_proj.weight.copy_(source_block.attention.v_proj.weight)
                deep_block.attention.out_proj.weight.zero_()

                deep_block.ffn.fc1.weight.copy_(source_block.ffn.fc1.weight)
                deep_block.ffn.fc1.bias.copy_(source_block.ffn.fc1.bias)
                deep_block.ffn.fc2.weight.zero_()
                deep_block.ffn.fc2.bias.zero_()
            else:
                source_block = wide_model.blocks[orig_ptr]

                deep_block.norm1.weight.copy_(source_block.norm1.weight)
                deep_block.norm2.weight.copy_(source_block.norm2.weight)

                deep_block.attention.q_proj.weight.copy_(source_block.attention.q_proj.weight)
                deep_block.attention.k_proj.weight.copy_(source_block.attention.k_proj.weight)
                deep_block.attention.v_proj.weight.copy_(source_block.attention.v_proj.weight)
                deep_block.attention.out_proj.weight.copy_(source_block.attention.out_proj.weight)

                deep_block.ffn.fc1.weight.copy_(source_block.ffn.fc1.weight)
                deep_block.ffn.fc1.bias.copy_(source_block.ffn.fc1.bias)
                deep_block.ffn.fc2.weight.copy_(source_block.ffn.fc2.weight)
                deep_block.ffn.fc2.bias.copy_(source_block.ffn.fc2.bias)

                orig_ptr += 1

    return deep_model
