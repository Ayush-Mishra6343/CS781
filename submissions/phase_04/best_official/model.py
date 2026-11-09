"""
Model architecture definition for Phase 4 CJPE - V4 Hybrid Decoder with LoRA.
Preserves the exact Phase 1 pretrained architecture and LoRA adapter configuration.
"""

import math
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

GLOBAL_ATTENTION_LAYERS = [0, 4, 8, 11]
LOCAL_ATTN_WINDOW = 256
V4_CONTEXT = 1024
N_EMBD = 768
N_LAYER = 12
N_HEAD = 12
N_KV_HEAD = 4
FFN_MULTIPLIER = 4


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(rms + self.eps))


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)."""
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        required_len = position_offset + seq_len
        if required_len > self.max_seq_len:
            self._build_cache(required_len)
            self.max_seq_len = required_len

        cos = self.cos_cached[position_offset : position_offset + seq_len]
        sin = self.sin_cached[position_offset : position_offset + seq_len]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

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
    """
    Hybrid Grouped Query Attention (GQA).
    Supports both global causal attention and local sliding-window causal attention.
    """
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        max_seq_len: int,
        attention_type: str = "global",
        local_window: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0
        assert attention_type in {"global", "local"}
        assert local_window > 0

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.kv_repeat = n_head // n_kv_head
        self.attention_type = attention_type
        self.local_window = local_window

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.rope = RotaryEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len)
        self.dropout = dropout

    def _build_attention_mask(
        self,
        query_length: int,
        key_length: int,
        query_position_offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = query_position_offset + torch.arange(query_length, device=device)
        key_positions = torch.arange(key_length, device=device)

        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if self.attention_type == "global":
            return causal_mask

        local_mask = key_positions.unsqueeze(0) >= (
            query_positions.unsqueeze(1) - self.local_window + 1
        )
        return causal_mask & local_mask

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

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(-2)

        q, k = self.rope(q, k, position_offset=past_len)

        if past_key_value is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present = (k, v) if use_cache else None

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        key_length = k.size(-2)
        attn_mask = self._build_attention_mask(
            query_length=T,
            key_length=key_length,
            query_position_offset=past_len,
            device=x.device,
        )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=(self.dropout if self.training else 0.0),
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y), present


class FeedForward(nn.Module):
    """Standard GELU Feed-Forward Network."""
    def __init__(self, n_embd: int, multiplier: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(n_embd * multiplier)
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Hybrid Transformer block with pre-norm RMSNorm and residual connections."""
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int,
        max_seq_len: int,
        ffn_multiplier: int,
        dropout: float,
        layer_index: int,
        global_attention_layers: List[int],
        local_window: int,
    ):
        super().__init__()
        self.layer_index = layer_index
        attention_type = "global" if layer_index in global_attention_layers else "local"
        self.attention_type = attention_type

        self.norm1 = RMSNorm(n_embd)
        self.attention = GQAAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            max_seq_len=max_seq_len,
            attention_type=attention_type,
            local_window=local_window,
            dropout=dropout,
        )
        self.norm2 = RMSNorm(n_embd)
        self.ffn = FeedForward(
            n_embd=n_embd,
            multiplier=ffn_multiplier,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_output, present = self.attention(
            self.norm1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x, present


class HybridLegalLM(nn.Module):
    """Pretrained Phase 1 V4 Hybrid Legal Language Model."""
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        n_kv_head: int,
        max_seq_len: int,
        ffn_multiplier: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_kv_head=n_kv_head,
                max_seq_len=max_seq_len,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
                layer_index=layer_index,
                global_attention_layers=GLOBAL_ATTENTION_LAYERS,
                local_window=LOCAL_ATTN_WINDOW,
            )
            for layer_index in range(n_layer)
        ])
        self.final_norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        x = self.token_embedding(input_ids)
        presents = [] if use_cache else None

        for layer_index, block in enumerate(self.blocks):
            past = past_key_values[layer_index] if past_key_values is not None else None
            x, present = block(x, past_key_value=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, presents


class LoRALinear(nn.Module):
    """
    LoRA low-rank adapter wrapping a Linear layer.
    W' = W + (B * A) * (alpha / rank).
    Matrix B is initialized to zeros, ensuring initial output equals the base layer.
    """
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.A), self.B) * self.scale


class CaseClassifier(nn.Module):
    """
    Binary case outcome classifier wrapping the adapted hybrid transformer backbone.
    Pools the hidden representation at the last non-pad token of the document window.
    """
    def __init__(self, backbone: HybridLegalLM, head_dropout: float = 0.10, n_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(backbone.n_embd, n_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.backbone.token_embedding(input_ids)
        for block in self.backbone.blocks:
            x, _ = block(x, past_key_value=None, use_cache=False)
        x = self.backbone.final_norm(x)
        last = (attention_mask.sum(1) - 1).clamp(min=0)
        pooled = x[torch.arange(x.size(0), device=x.device), last]
        return self.head(pooled)


def build_v4_backbone(checkpoint_path: str, vocab_size: int = 25000) -> HybridLegalLM:
    """
    Rebuilds the exact V4 hybrid architecture and strictly verifies and loads Phase 1 weights.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError(f"{checkpoint_path} must contain model_state_dict and config")

    v4_config = ckpt["config"]
    expected = {
        "vocab_size": 25000,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
        "n_kv_head": N_KV_HEAD,
        "ffn_multiplier": FFN_MULTIPLIER,
        "context_length": V4_CONTEXT,
        "local_attention_window": 256,
    }
    for key, exp_val in expected.items():
        if key not in v4_config or int(v4_config[key]) != exp_val:
            raise ValueError(f"V4 checkpoint config mismatch for {key}: {v4_config.get(key)} vs {exp_val}")

    if list(map(int, v4_config.get("global_attention_layers", []))) != GLOBAL_ATTENTION_LAYERS:
        raise ValueError(f"Unexpected V4 global-attention pattern: {v4_config.get('global_attention_layers')}")

    model = HybridLegalLM(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_kv_head=N_KV_HEAD,
        max_seq_len=V4_CONTEXT,
        ffn_multiplier=FFN_MULTIPLIER,
        dropout=0.0,
    )

    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State dict loading mismatch: missing={missing}, unexpected={unexpected}")

    param_count = sum(p.numel() for p in model.parameters())
    if param_count != 94_762_752:
        raise ValueError(f"Unexpected V4 parameter count: {param_count:,} (expected 94,762,752)")

    return model


def apply_lora_and_head(
    backbone: HybridLegalLM,
    rank: int = 8,
    alpha: float = 16.0,
    lora_dropout: float = 0.05,
    head_dropout: float = 0.10,
) -> CaseClassifier:
    """
    Freezes the backbone, wraps attention q_proj and v_proj with LoRALinear, and adds classification head.
    """
    for p in backbone.parameters():
        p.requires_grad_(False)

    for block in backbone.blocks:
        block.attention.q_proj = LoRALinear(block.attention.q_proj, rank, alpha, lora_dropout)
        block.attention.v_proj = LoRALinear(block.attention.v_proj, rank, alpha, lora_dropout)

    classifier = CaseClassifier(backbone, head_dropout=head_dropout)
    return classifier


def build_classifier(
    checkpoint_path: str,
    vocab_size: int = 25000,
    rank: int = 8,
    alpha: float = 16.0,
    lora_dropout: float = 0.05,
    head_dropout: float = 0.10,
    device: Optional[torch.device] = None,
) -> CaseClassifier:
    """Convenience function to build and initialize the complete Phase 4 CaseClassifier."""
    backbone = build_v4_backbone(checkpoint_path, vocab_size=vocab_size)
    model = apply_lora_and_head(
        backbone,
        rank=rank,
        alpha=alpha,
        lora_dropout=lora_dropout,
        head_dropout=head_dropout,
    )
    if device is not None:
        model = model.to(device)
    return model


def save_adapter_checkpoint(
    model: CaseClassifier,
    save_path: str,
    epoch: int,
    dev_macro_f1: float,
    rank: int = 8,
    alpha: float = 16.0,
    chunk_tokens: int = 512,
    max_chunks: int = 8,
):
    """Saves only the trained LoRA adapter and classification head weights."""
    adapter_state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if (k.startswith("backbone.blocks.") and (".q_proj.A" in k or ".q_proj.B" in k or ".v_proj.A" in k or ".v_proj.B" in k))
        or k.startswith("head.")
    }
    ckpt = {
        "adapter_state_dict": adapter_state,
        "epoch": epoch,
        "dev_macro_f1": dev_macro_f1,
        "config": {
            "rank": rank,
            "alpha": alpha,
            "chunk_tokens": chunk_tokens,
            "max_chunks": max_chunks,
        },
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)


def load_adapter_checkpoint(
    model: CaseClassifier,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Loads trained LoRA adapter and classification head weights into an initialized model."""
    target_device = device if device is not None else next(model.parameters()).device
    ckpt = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    if "adapter_state_dict" not in ckpt:
        raise ValueError(f"{checkpoint_path} does not contain 'adapter_state_dict'")
    own = model.state_dict()
    own.update(ckpt["adapter_state_dict"])
    model.load_state_dict(own, strict=True)
    return ckpt
