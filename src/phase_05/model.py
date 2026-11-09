"""
Phase 5: Legal Statute Identification (LSI)
Architecture: Hierarchical Local-Global SwiGLU Network + Sparse Lexical Fusion.
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

class HLG_LSI_Classifier(nn.Module):
    def __init__(self, d_model=768, num_statutes=100, num_heads=8):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_model * 2)
        self.global_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.classifier = nn.Linear(d_model, num_statutes)

    def forward(self, chunk_embeddings):
        # chunk_embeddings: [batch_size, num_chunks, d_model]
        g_out, _ = self.global_attn(chunk_embeddings, chunk_embeddings, chunk_embeddings)
        pooled = g_out.mean(dim=1)
        swiglu_out = self.swiglu(pooled)
        logits = self.classifier(swiglu_out)
        return logits
