"""
Phase 3: Parameter-Efficient Fine-Tuning (PEFT) for Rhetorical Role Labeling
Ensemble of LoRA, LoRMA, and BitFit Adapters with Asymmetric Focal Loss.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.scaling = alpha / rank
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        self.linear.weight.requires_grad = False

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out
