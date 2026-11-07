import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Fourier features for the last dimension of an input tensor."""

    def __init__(self, num_bands: int = 8, max_frequency: float = 16.0, include_input: bool = True):
        super().__init__()
        self.num_bands = num_bands
        self.max_frequency = max_frequency
        self.include_input = include_input

        # Frequencies are fixed, not learned.
        frequencies = torch.logspace(
            start=0.0,
            end=math.log2(max_frequency),
            steps=num_bands,
            base=2.0,
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_multiplier(self) -> int:
        return (1 if self.include_input else 0) + 2 * self.num_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        parts = []
        if self.include_input:
            parts.append(x)

        # [..., D, K]
        xb = x.unsqueeze(-1) * self.frequencies * math.pi
        sin_cos = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
        # [..., D * 2K]
        sin_cos = sin_cos.flatten(start_dim=-2)
        parts.append(sin_cos)
        return torch.cat(parts, dim=-1)

