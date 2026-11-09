"""
Parameter-Efficient Fine-Tuning (PEFT) library:
- LoRA: Low-Rank Adaptation
- LoRMA: Low-Rank Moving Average Adapter
- BitFit: Bias-Term Only Tuning
"""
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        in_dim = linear_layer.in_features
        out_dim = linear_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        self.linear.weight.requires_grad = False

    def forward(self, x):
        base = self.linear(x)
        lora = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora

def apply_bitfit(model):
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if "bias" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    return trainable_params, all_params
