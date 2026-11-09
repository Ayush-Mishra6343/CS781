"""
Phase 4 & 5: SwiGLU Gated Feed-Forward + Local-Global Window Attention.
Provides high expressive capacity for long-context Indian court cases.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class LocalGlobalDualAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=8, window_size=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.local_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.global_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.swiglu = SwiGLU(d_model, d_model * 2)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, chunk_mask=None):
        # x: [B, Num_Chunks, Window_Size, D]
        B, C, W, D = x.shape
        x_flat = x.view(B * C, W, D)
        loc_out, _ = self.local_attn(x_flat, x_flat, x_flat)
        x_loc = self.ln1(x_flat + loc_out)

        # Chunk-level summary
        chunk_rep = x_loc.mean(dim=1).view(B, C, D)
        glob_out, _ = self.global_attn(chunk_rep, chunk_rep, chunk_rep)
        chunk_glob = self.ln2(chunk_rep + glob_out)

        # SwiGLU projection
        out = self.swiglu(chunk_glob)
        return out
