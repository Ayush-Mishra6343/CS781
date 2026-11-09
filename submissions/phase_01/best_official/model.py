"""
Phase 1: In-House Legal Language Model Architecture (V4 Winning Model)
Extracted directly from the author's winning notebook `v4.ipynb`.

Architecture: HybridLegalLM (94,762,752 Parameters)
- 12 Decoder Layers, 768 Hidden Dimension
- Grouped Query Attention (GQA): 12 Query Heads, 4 Key/Value Heads
- Hybrid Attention: Global layers [0, 4, 8, 11] + Local sliding-window (256) layers
- Rotary Position Embeddings (RoPE)
- RMSNorm pre-normalization
- GELU Feed-Forward Network (multiplier 4)
- Weight Tying between Embedding and LM Head
- KV Cache supported for fast autoregressive generation
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

GLOBAL_ATTENTION_LAYERS = [0, 4, 8, 11]
LOCAL_ATTN_WINDOW = 256

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return self.weight * x

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, q, k, position_offset=0):
        seq_len = q.size(-2)
        required_len = position_offset + seq_len
        if required_len > self.max_seq_len:
            self._build_cache(required_len)
            self.max_seq_len = required_len

        cos = self.cos_cached[position_offset : position_offset + seq_len]
        sin = self.sin_cached[position_offset : position_offset + seq_len]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = (q * cos) + (torch.cat((-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]), dim=-1) * sin)
        k = (k * cos) + (torch.cat((-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]), dim=-1) * sin)
        return q, k

class Attention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, max_seq_len, dropout, layer_index, global_attention_layers, local_window):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.num_queries_per_kv = n_head // n_kv_head
        self.layer_index = layer_index
        self.is_global = layer_index in global_attention_layers
        self.local_window = local_window

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x, past_key_value=None, use_cache=False):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_head, self.head_dim).transpose(1, 2)

        offset = past_key_value[0].size(-2) if past_key_value is not None else 0
        q, k = self.rotary(q, k, position_offset=offset)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)
        present = (k, v) if use_cache else None

        # Repeat KV for GQA
        k_rep = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v_rep = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        total_len = k_rep.size(-2)

        # Causal & local/global masking
        q_pos = torch.arange(offset, offset + S, device=x.device).unsqueeze(1)
        k_pos = torch.arange(total_len, device=x.device).unsqueeze(0)
        causal_mask = k_pos <= q_pos

        if not self.is_global:
            local_mask = (q_pos - k_pos) <= self.local_window
            valid_mask = causal_mask & local_mask
        else:
            valid_mask = causal_mask

        scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v_rep).transpose(1, 2).contiguous().view(B, S, self.n_embd)
        return self.out_proj(context), present

class FeedForward(nn.Module):
    def __init__(self, n_embd, multiplier=4, dropout=0.0):
        super().__init__()
        hidden_dim = n_embd * multiplier
        self.fc1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, max_seq_len, ffn_multiplier, dropout, layer_index):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attention = Attention(n_embd, n_head, n_kv_head, max_seq_len, dropout, layer_index, GLOBAL_ATTENTION_LAYERS, LOCAL_ATTN_WINDOW)
        self.norm2 = RMSNorm(n_embd)
        self.ffn = FeedForward(n_embd, ffn_multiplier, dropout)

    def forward(self, x, past_key_value=None, use_cache=False):
        attn_out, present = self.attention(self.norm1(x), past_key_value=past_key_value, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, present

class HybridLegalLM(nn.Module):
    def __init__(self, vocab_size=25000, n_embd=768, n_layer=12, n_head=12, n_kv_head=4, max_seq_len=1024, ffn_multiplier=4, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, n_kv_head, max_seq_len, ffn_multiplier, dropout, i)
            for i in range(n_layer)
        ])
        self.final_norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight # Weight tying

    def forward(self, input_ids, targets=None, past_key_values=None, use_cache=False):
        x = self.token_embedding(input_ids)
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, past_key_value=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100)
        return (logits, presents, loss) if targets is not None else (logits, presents)
