"""
Specialized Loss Functions for Legal NLP:
- AsymmetricFocalLoss: Handles extreme class imbalance in Rhetorical Roles
- DistillationLoss: Student-Teacher Knowledge Distillation (KL + MSE)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # targets: one-hot or binary [B, C]
        p = torch.sigmoid(logits)
        targets = targets.float()

        # Positive loss
        pos_p = p.clamp(min=self.eps, max=1.0)
        pos_loss = targets * ((1.0 - pos_p) ** self.gamma_pos) * torch.log(pos_p)

        # Negative loss with clipping
        neg_p = (1.0 - p).clamp(min=self.eps, max=1.0)
        if self.clip > 0:
            p_clip = (p - self.clip).clamp(min=0.0)
            neg_loss = (1.0 - targets) * (p_clip ** self.gamma_neg) * torch.log((1.0 - p_clip).clamp(min=self.eps))
        else:
            neg_loss = (1.0 - targets) * (p ** self.gamma_neg) * torch.log(neg_p)

        return -(pos_loss + neg_loss).sum(dim=-1).mean()

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, student_hidden=None, teacher_hidden=None):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        kd_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)

        if student_hidden is not None and teacher_hidden is not None:
            hidden_loss = self.mse(student_hidden, teacher_hidden)
            return self.alpha * kd_loss + (1.0 - self.alpha) * hidden_loss
        return kd_loss
