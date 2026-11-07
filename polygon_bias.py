import math
import torch
import torch.nn as nn

from fourier_features import FourierFeatures

class RelativeSpatialCyclicBias(nn.Module):
    """Attention bias from relative spatial geometry and cyclic boundary distance."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 128,
        spatial_bands: int = 8,
        cyclic_bands: int = 6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.spatial_fourier = FourierFeatures(num_bands=spatial_bands, max_frequency=16.0, include_input=True)
        self.cyclic_fourier = FourierFeatures(num_bands=cyclic_bands, max_frequency=16.0, include_input=True)

        # Spatial raw dimension: delta(2), distance(1), direction(2) = 5
        spatial_in_dim = 5 * self.spatial_fourier.output_multiplier

        # Cyclic raw dimension: cyclic_distance(1), sin phase(1), cos phase(1) = 3
        cyclic_in_dim = 3 * self.cyclic_fourier.output_multiplier

        in_dim = spatial_in_dim + cyclic_in_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, vertices: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute per-head attention bias.

        Args:
            vertices: [B, V, 2], normalized polygon vertices.
            mask:     [B, V], True for valid vertices.
            lengths:  [B], number of valid vertices in each polygon.

        Returns:
            bias: [B, H, V, V]
        """
        bsz, vmax, _ = vertices.shape
        device = vertices.device

        # delta[b, i, j] = p_j - p_i
        pi = vertices[:, :, None, :]  # [B, V, 1, 2]
        pj = vertices[:, None, :, :]  # [B, 1, V, 2]
        delta = pj - pi               # [B, V, V, 2]

        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        direction = delta / distance.clamp(min=1e-6)

        spatial_raw = torch.cat([delta, distance, direction], dim=-1)  # [B, V, V, 5]
        spatial_feat = self.spatial_fourier(spatial_raw)

        idx = torch.arange(vmax, device=device)
        i = idx[None, :, None]  # [1, V, 1]
        j = idx[None, None, :]  # [1, 1, V]
        n = lengths[:, None, None].clamp(min=1)  # [B, 1, 1]

        forward = (j - i) % n
        backward = (i - j) % n
        cyclic_distance = torch.minimum(forward, backward).float() / n.float()

        directed_phase = 2.0 * math.pi * (forward.float() / n.float())
        cyclic_raw = torch.stack(
            [
                cyclic_distance,
                torch.sin(directed_phase),
                torch.cos(directed_phase),
            ],
            dim=-1,
        )  # [B, V, V, 3]
        cyclic_feat = self.cyclic_fourier(cyclic_raw)

        features = torch.cat([spatial_feat, cyclic_feat], dim=-1)
        bias = self.mlp(features)  # [B, V, V, H]
        bias = bias.permute(0, 3, 1, 2).contiguous()  # [B, H, V, V]

        # The actual masking is applied in the attention layer.
        # We do not need to modify bias here.
        return bias

    def forward_sparse(
        self,
        vertices: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-head attention bias only for sparse neighbor pairs.

        Args:
            vertices:     [B, V, 2]
            mask:         [B, V]
            lengths:      [B]
            neighbor_idx: [B, V, K]

        Returns:
            bias: [B, H, V, K]
        """
        del mask
        bsz, vmax, _ = vertices.shape
        k_neighbors = neighbor_idx.shape[-1]
        device = vertices.device

        pi = vertices[:, :, None, :]  # [B, V, 1, 2]

        # Gather only requested neighbor coordinates. Avoid expanding to
        # [B,V,V,2], which would make sparse mode quadratic in memory.
        batch_idx = torch.arange(bsz, device=device)[:, None, None].expand(
            bsz, vmax, k_neighbors
        )
        pj = vertices[batch_idx, neighbor_idx]  # [B, V, K, 2]
        delta = pj - pi  # [B, V, K, 2]

        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        direction = delta / distance.clamp(min=1e-6)
        spatial_raw = torch.cat([delta, distance, direction], dim=-1)
        spatial_feat = self.spatial_fourier(spatial_raw)

        i = torch.arange(vmax, device=device)[None, :, None]
        j = neighbor_idx
        n = lengths[:, None, None].clamp(min=1)
        forward = (j - i) % n
        backward = (i - j) % n
        cyclic_distance = torch.minimum(forward, backward).float() / n.float()
        directed_phase = 2.0 * math.pi * (forward.float() / n.float())
        cyclic_raw = torch.stack(
            [
                cyclic_distance,
                torch.sin(directed_phase),
                torch.cos(directed_phase),
            ],
            dim=-1,
        )
        cyclic_feat = self.cyclic_fourier(cyclic_raw)

        features = torch.cat([spatial_feat, cyclic_feat], dim=-1)
        bias = self.mlp(features)  # [B, V, K, H]
        bias = bias.permute(0, 3, 1, 2).contiguous()  # [B, H, V, K]
        return bias
