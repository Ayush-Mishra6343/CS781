"""LoRMA and SwiGLU Adapters for Phase 5 Legal Statute Identification.

Implements activation-space LoRMA with SwiGLU gating and global-window
conditioning based on Phase-4 HLG-SwiGLU-LoRMA research.
"""

import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLULoRMALinear(nn.Module):
    """Activation-space LoRMA layer with SwiGLU gating and global window summary.

    Given base projection z = W_base @ x:
    g = masked_mean(z) (global summary)
    u = W_A_gate @ drop(z) + W_A_glob_gate @ g
    v = W_A_lin  @ drop(z) + W_A_glob_lin  @ g
    delta = W_B @ (SiLU(u) * v) * scale
    output = z + delta

    At step zero, W_B is initialized to 0, guaranteeing strict mathematical identity
    to the base projection (delta = 0).
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base = base
        # Freeze base linear layer permanently
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        d_out = base.out_features

        # Low-rank gate and linear branches
        self.A_gate = nn.Linear(d_out, rank, bias=False)
        self.A_lin = nn.Linear(d_out, rank, bias=False)
        self.A_glob_gate = nn.Linear(d_out, rank, bias=False)
        self.A_glob_lin = nn.Linear(d_out, rank, bias=False)
        self.B = nn.Linear(rank, d_out, bias=False)

        # Initialization
        nn.init.kaiming_uniform_(self.A_gate.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_lin.weight, a=math.sqrt(5))
        nn.init.zeros_(self.A_glob_gate.weight)
        nn.init.zeros_(self.A_glob_lin.weight)
        # Strict zero-initialization on B guarantees step-zero identity
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.base(x)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(z.dtype)
            g = (z * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        else:
            g = z.mean(dim=1, keepdim=True)

        z_drop = self.dropout(z)
        u = self.A_gate(z_drop) + self.A_glob_gate(g)
        v = self.A_lin(z_drop) + self.A_glob_lin(g)
        delta = self.B(F.silu(u) * v) * self.scale
        return z + delta


def apply_lorma_to_v1(
    model: nn.Module,
    target_modules: List[str] = ["q_proj", "v_proj"],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> int:
    """Replaces target linear layers in V1LegalLM blocks with SwiGLULoRMALinear.

    Freezes all base parameters and returns the number of trainable adapter parameters added.
    """
    # First, freeze all existing parameters in model
    for p in model.parameters():
        p.requires_grad_(False)

    num_replaced = 0
    # Walk through transformer blocks
    for block_idx, block in enumerate(model.blocks):
        attn = block.attention
        for mod_name in target_modules:
            if hasattr(attn, mod_name):
                base_linear = getattr(attn, mod_name)
                if isinstance(base_linear, nn.Linear):
                    adapter = SwiGLULoRMALinear(
                        base=base_linear, rank=rank, alpha=alpha, dropout=dropout
                    )
                    setattr(attn, mod_name, adapter)
                    num_replaced += 1

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"LoRMA injected: replaced {num_replaced} modules across {len(model.blocks)} blocks. "
        f"Trainable adapter parameters: {trainable_params:,}"
    )
    return trainable_params
