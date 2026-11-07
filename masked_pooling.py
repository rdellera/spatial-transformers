from typing import Tuple

import torch
import torch.nn as nn


class MaskedAttentionPooling(nn.Module):
    """Learned attention pooling over valid vertices."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(x).squeeze(-1)  # [B, V]
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Simple alternative to attention pooling."""
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)