import torch
import torch.nn as nn

from fourier_features import FourierFeatures

class RelativeGeometrySpatialBias(nn.Module):
    """Attention bias from relative geometry-to-geometry centroid displacement."""

    def __init__(self, num_heads: int, hidden_dim: int = 128, spatial_bands: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.spatial_fourier = FourierFeatures(num_bands=spatial_bands, max_frequency=16.0, include_input=True)

        # Raw spatial features: delta(2), distance(1), direction(2) = 5
        in_dim = 5 * self.spatial_fourier.output_multiplier

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, centroids: torch.Tensor, geom_mask: torch.Tensor) -> torch.Tensor:
        """Build attention bias.

        Args:
            centroids: [B, G, 2], normalized geometry centroids
            geom_mask: [B, G], True for real geometries

        Returns:
            bias: [B, H, G, G]
        """
        del geom_mask  # Masking is handled by the attention layer; kept for API symmetry.
        delta = centroids.unsqueeze(1) - centroids.unsqueeze(2)  # [B, G, G, 2] = c_j - c_i
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        direction = delta / distance.clamp(min=1e-6)
        spatial_raw = torch.cat([delta, distance, direction], dim=-1)
        spatial_feat = self.spatial_fourier(spatial_raw)
        bias = self.mlp(spatial_feat)                 # [B, G, G, H]
        bias = bias.permute(0, 3, 1, 2).contiguous()  # [B, H, G, G]
        return bias

    def forward_sparse(
        self,
        centroids: torch.Tensor,
        geom_mask: torch.Tensor,
        neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Build geometry relative bias only for sparse neighbor pairs.

        Args:
            centroids:    [B, G, 2]
            geom_mask:    [B, G]
            neighbor_idx: [B, G, K]

        Returns:
            bias: [B, H, G, K]
        """
        del geom_mask
        bsz, gmax, _ = centroids.shape
        k_neighbors = neighbor_idx.shape[-1]
        ci = centroids[:, :, None, :]

        # Gather only requested neighbor centroids. Avoid expanding to
        # [B,G,G,2], which would make sparse mode quadratic in memory.
        batch_idx = torch.arange(bsz, device=centroids.device)[:, None, None].expand(
            bsz, gmax, k_neighbors
        )
        cj = centroids[batch_idx, neighbor_idx]  # [B, G, K, 2]
        delta = cj - ci
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        direction = delta / distance.clamp(min=1e-6)
        spatial_raw = torch.cat([delta, distance, direction], dim=-1)
        spatial_feat = self.spatial_fourier(spatial_raw)
        bias = self.mlp(spatial_feat)                 # [B, G, K, H]
        bias = bias.permute(0, 3, 1, 2).contiguous()  # [B, H, G, K]
        return bias
