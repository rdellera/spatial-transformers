from dataclasses import dataclass

import torch
import torch.nn as nn

from dataset_bias import RelativeGeometrySpatialBias
from masked_pooling import MaskedAttentionPooling
from transformer_utils import RelativeBiasSelfAttention, build_geometry_sparse_neighbors

@dataclass
class DatasetEncoderConfig:
    polygon_embedding_dim: int = 128
    geom_feature_dim: int = 6
    d_model: int = 128
    embedding_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    bias_hidden_dim: int = 128
    spatial_bands: int = 8
    mlp_ratio: float = 4.0
    attention_mode: str = "full"  # "full" or "sparse"
    sparse_spatial_k: int = 64


class DatasetTransformerBlock(nn.Module):
    """Transformer block for geometry tokens with relative spatial attention bias."""

    def __init__(self, config: DatasetEncoderConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = RelativeBiasSelfAttention(config.d_model, config.num_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)
        hidden_dim = int(config.d_model * config.mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, attention_bias: torch.Tensor, neighbor_idx: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask, attention_bias, neighbor_idx=neighbor_idx)
        x = x * mask.unsqueeze(-1).float()
        x = x + self.ffn(self.norm2(x))
        x = x * mask.unsqueeze(-1).float()
        return x


class DatasetEncoder(nn.Module):
    """Second-stage encoder: variable-size set of geometries -> one dataset embedding."""

    def __init__(self, config: DatasetEncoderConfig):
        super().__init__()
        self.config = config
        input_dim = config.polygon_embedding_dim + config.geom_feature_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.relative_bias = RelativeGeometrySpatialBias(
            num_heads=config.num_heads,
            hidden_dim=config.bias_hidden_dim,
            spatial_bands=config.spatial_bands,
        )

        self.blocks = nn.ModuleList([DatasetTransformerBlock(config) for _ in range(config.num_layers)])
        self.pool = MaskedAttentionPooling(config.d_model)
        self.out = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.embedding_dim),
        )

    def forward(
        self,
        geom_embeddings: torch.Tensor,
        geom_features: torch.Tensor,
        geom_centroids: torch.Tensor,
        geom_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        x = torch.cat([geom_embeddings, geom_features], dim=-1)
        x = self.input_proj(x)
        x = x * geom_mask.unsqueeze(-1).float()

        neighbor_idx = None
        if self.config.attention_mode == "full":
            attention_bias = self.relative_bias(geom_centroids, geom_mask)
        elif self.config.attention_mode == "sparse":
            neighbor_idx = build_geometry_sparse_neighbors(
                centroids=geom_centroids,
                geom_mask=geom_mask,
                spatial_k=self.config.sparse_spatial_k,
            )
            attention_bias = self.relative_bias.forward_sparse(
                centroids=geom_centroids,
                geom_mask=geom_mask,
                neighbor_idx=neighbor_idx,
            )
        else:
            raise ValueError(f"Unknown dataset attention_mode: {self.config.attention_mode!r}")

        for block in self.blocks:
            x = block(x, geom_mask, attention_bias, neighbor_idx=neighbor_idx)

        pooled, pool_weights = self.pool(x, geom_mask)
        embedding = self.out(pooled)

        if return_aux:
            return {
                "embedding": embedding,
                "geometry_token_embeddings": x,
                "pool_weights": pool_weights,
                "attention_bias": attention_bias,
            }
        return embedding