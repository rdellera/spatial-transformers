import torch
import torch.nn as nn


def _pad_indices(indices: torch.Tensor, target_k: int) -> torch.Tensor:
    """Pad a 1D LongTensor of indices to target_k by repeating the first value."""
    if indices.numel() == 0:
        return torch.zeros(target_k, dtype=torch.long, device=indices.device)
    if indices.numel() >= target_k:
        return indices[:target_k]
    pad = indices[:1].expand(target_k - indices.numel())
    return torch.cat([indices, pad], dim=0)


def build_polygon_sparse_neighbors(
    vertices: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    boundary_window: int = 8,
    spatial_k: int = 16,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Build sparse neighbor indices for polygon vertices.

    Each valid vertex attends to local cyclic-boundary neighbors and to spatial
    k-nearest neighbors in normalized coordinate space. Invalid query positions
    are filled with index 0 and later zeroed by the attention block.

    Returns:
        neighbor_idx: [B, V, K], where K = (2 * boundary_window + 1) + spatial_k.
    """
    bsz, vmax, _ = vertices.shape
    device = vertices.device
    boundary_count = max(0, 2 * int(boundary_window) + 1)
    spatial_count = max(0, int(spatial_k))
    total_k = boundary_count + spatial_count
    if total_k <= 0:
        total_k = 1
        boundary_count = 1

    neighbor_idx = torch.zeros((bsz, vmax, total_k), dtype=torch.long, device=device)

    for b in range(bsz):
        n = int(lengths[b].item()) if lengths is not None else int(mask[b].sum().item())
        n = max(0, min(n, vmax))
        if n <= 0:
            continue

        row_parts = []
        if boundary_count > 0:
            idx = torch.arange(n, device=device)
            offsets = torch.arange(-boundary_window, boundary_window + 1, device=device)
            if offsets.numel() == 0:
                offsets = torch.zeros(1, dtype=torch.long, device=device)
            cyclic = (idx[:, None] + offsets[None, :]) % n  # [n, boundary_count]
            row_parts.append(cyclic.long())

        if spatial_count > 0:
            coords = vertices[b, :n, :]
            k_eff = min(spatial_count, n)
            knn_chunks = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                dist = torch.cdist(coords[start:end], coords)  # [chunk, n]
                idx = torch.topk(dist, k=k_eff, largest=False).indices.long()
                if k_eff < spatial_count:
                    idx = torch.cat([idx, idx[:, :1].expand(idx.shape[0], spatial_count - k_eff)], dim=1)
                knn_chunks.append(idx)
            spatial = torch.cat(knn_chunks, dim=0)
            row_parts.append(spatial)

        neigh = torch.cat(row_parts, dim=1) if row_parts else torch.arange(n, device=device)[:, None]
        if neigh.shape[1] < total_k:
            neigh = torch.cat([neigh, neigh[:, :1].expand(n, total_k - neigh.shape[1])], dim=1)
        neighbor_idx[b, :n, :] = neigh[:, :total_k]

    return neighbor_idx


def build_geometry_sparse_neighbors(
    centroids: torch.Tensor,
    geom_mask: torch.Tensor,
    spatial_k: int = 64,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Build k-nearest-neighbor sparse indices for dataset geometries.

    Returns:
        neighbor_idx: [B, G, K], where K = spatial_k.
    """
    bsz, gmax, _ = centroids.shape
    device = centroids.device
    total_k = max(1, int(spatial_k))
    neighbor_idx = torch.zeros((bsz, gmax, total_k), dtype=torch.long, device=device)

    for b in range(bsz):
        valid = torch.nonzero(geom_mask[b], as_tuple=False).squeeze(-1)
        n = int(valid.numel())
        if n <= 0:
            continue
        coords = centroids[b, valid, :]
        k_eff = min(total_k, n)
        out_chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dist = torch.cdist(coords[start:end], coords)
            local_idx = torch.topk(dist, k=k_eff, largest=False).indices.long()
            if k_eff < total_k:
                local_idx = torch.cat([local_idx, local_idx[:, :1].expand(local_idx.shape[0], total_k - k_eff)], dim=1)
            out_chunks.append(valid[local_idx])
        neigh = torch.cat(out_chunks, dim=0)
        neighbor_idx[b, valid, :] = neigh[:, :total_k]
    return neighbor_idx


class RelativeBiasSelfAttention(nn.Module):
    """Multi-head self-attention with additive relative attention bias.

    Supports both full attention (attention_bias [B, H, N, N]) and true sparse
    attention (neighbor_idx [B, N, K], attention_bias [B, H, N, K]). Sparse
    attention avoids constructing the quadratic [N, N] score tensor.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attention_bias: torch.Tensor,
        neighbor_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if neighbor_idx is None:
            return self._forward_full(x=x, mask=mask, attention_bias=attention_bias)
        return self._forward_sparse(x=x, mask=mask, attention_bias=attention_bias, neighbor_idx=neighbor_idx)

    def _forward_full(self, x: torch.Tensor, mask: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(bsz, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        scores = scores + attention_bias
        key_mask = mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v)  # [B, H, N, Dh]
        y = y.transpose(1, 2).contiguous().view(bsz, n_tokens, self.d_model)
        y = self.out(y)
        y = y * mask.unsqueeze(-1)
        return y

    def _gather_token_neighbors(self, tokens: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
        """Gather sparse neighbors without constructing a dense [B,H,N,N,D] tensor.

        Args:
            tokens:       [B, H, N, D]
            neighbor_idx: [B, N, K]

        Returns:
            gathered: [B, H, N, K, D]
        """
        bsz, num_heads, n_tokens, head_dim = tokens.shape
        k_neighbors = neighbor_idx.shape[-1]

        # Advanced indexing selects only the requested neighbors. This is the
        # crucial sparse operation: it allocates [B,N,K,H,D], not [B,H,N,N,D].
        tokens_bnhd = tokens.permute(0, 2, 1, 3)  # [B, N, H, D]
        batch_idx = torch.arange(bsz, device=tokens.device)[:, None, None].expand(
            bsz, n_tokens, k_neighbors
        )
        gathered = tokens_bnhd[batch_idx, neighbor_idx]  # [B, N, K, H, D]
        return gathered.permute(0, 3, 1, 2, 4).contiguous()  # [B, H, N, K, D]

    def _forward_sparse(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attention_bias: torch.Tensor,
        neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_tokens, _ = x.shape
        k_neighbors = neighbor_idx.shape[-1]

        qkv = self.qkv(x)
        qkv = qkv.view(bsz, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_neigh = self._gather_token_neighbors(k, neighbor_idx)  # [B, H, N, K, Dh]
        v_neigh = self._gather_token_neighbors(v, neighbor_idx)  # [B, H, N, K, Dh]

        scores = (q.unsqueeze(-2) * k_neigh).sum(dim=-1) * self.scale  # [B, H, N, K]
        scores = scores + attention_bias

        key_valid = torch.gather(mask, dim=1, index=neighbor_idx.reshape(bsz, -1)).view(
            bsz, n_tokens, k_neighbors
        )
        scores = scores.masked_fill(~key_valid[:, None, :, :], torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = (attn.unsqueeze(-1) * v_neigh).sum(dim=-2)  # [B, H, N, Dh]
        y = y.transpose(1, 2).contiguous().view(bsz, n_tokens, self.d_model)
        y = self.out(y)
        y = y * mask.unsqueeze(-1)
        return y


class PolygonTransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelativeBiasSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attention_bias: torch.Tensor,
        neighbor_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, attention_bias=attention_bias, neighbor_idx=neighbor_idx)
        x = x * mask.unsqueeze(-1)
        x = x + self.mlp(self.norm2(x))
        x = x * mask.unsqueeze(-1)
        return x
