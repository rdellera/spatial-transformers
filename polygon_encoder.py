from dataclasses import dataclass

import torch
import torch.nn as nn

from polygon_bias import RelativeSpatialCyclicBias
from transformer_utils import PolygonTransformerBlock, build_polygon_sparse_neighbors
from masked_pooling import MaskedAttentionPooling

@dataclass
class PolygonEncoderConfig:
    d_model: int = 128
    embedding_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    bias_hidden_dim: int = 128
    spatial_bands: int = 8
    cyclic_bands: int = 6
    mlp_ratio: float = 4.0
    attention_mode: str = "full"  # "full" or "sparse"
    sparse_boundary_window: int = 8
    sparse_spatial_k: int = 16


class PolygonEncoder(nn.Module):
    """Transformer encoder for simple polygons with variable vertex count."""

    def __init__(self, config: PolygonEncoderConfig):
        super().__init__()
        self.config = config

        # Vertex features are normalized coordinates only in this minimal version.
        # You can extend this to include edge length, turn angle, vertex class, etc.
        self.vertex_proj = nn.Sequential(
            nn.Linear(2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.relative_bias = RelativeSpatialCyclicBias(
            num_heads=config.num_heads,
            hidden_dim=config.bias_hidden_dim,
            spatial_bands=config.spatial_bands,
            cyclic_bands=config.cyclic_bands,
        )

        self.blocks = nn.ModuleList([
            PolygonTransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)
        self.pool = MaskedAttentionPooling(config.d_model)
        self.out = nn.Linear(config.d_model, config.embedding_dim)

    def forward(
        self,
        vertices: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        return_aux: bool = False,
    ):
        """Encode a batch of polygons.

        Args:
            vertices: [B, V, 2], normalized coordinates.
            mask:     [B, V], True for valid vertices.
            lengths:  [B], number of valid vertices per polygon.
            return_aux: if True, also return vertex embeddings and pooling weights.
        """
        x = self.vertex_proj(vertices)
        x = x * mask.unsqueeze(-1)

        # The same relative bias is reused in every layer.
        # You could also compute a separate bias per layer, but this is simpler.
        neighbor_idx = None
        if self.config.attention_mode == "full":
            attention_bias = self.relative_bias(vertices=vertices, mask=mask, lengths=lengths)
        elif self.config.attention_mode == "sparse":
            neighbor_idx = build_polygon_sparse_neighbors(
                vertices=vertices,
                mask=mask,
                lengths=lengths,
                boundary_window=self.config.sparse_boundary_window,
                spatial_k=self.config.sparse_spatial_k,
            )
            attention_bias = self.relative_bias.forward_sparse(
                vertices=vertices,
                mask=mask,
                lengths=lengths,
                neighbor_idx=neighbor_idx,
            )
        else:
            raise ValueError(f"Unknown polygon attention_mode: {self.config.attention_mode!r}")

        for block in self.blocks:
            x = block(x, mask=mask, attention_bias=attention_bias, neighbor_idx=neighbor_idx)

        x = self.norm(x)
        x = x * mask.unsqueeze(-1)

        pooled, pool_weights = self.pool(x, mask)
        embedding = self.out(pooled)

        if return_aux:
            return {
                "embedding": embedding,
                "vertex_embeddings": x,
                "pool_weights": pool_weights,
                "attention_bias": attention_bias,
            }
        return embedding