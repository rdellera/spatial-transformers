from typing import List, Tuple
import torch

def collate_polygons(polygons: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate a list of variable-length polygons into padded tensors.

    Returns:
        vertices: [B, V_max, 2]
        mask:     [B, V_max], True for valid vertices
        lengths:  [B]
    """
    lengths = torch.tensor([p.shape[0] for p in polygons], dtype=torch.long)
    batch_size = len(polygons)
    max_len = int(lengths.max().item())

    vertices = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for b, p in enumerate(polygons):
        v = p.shape[0]
        vertices[b, :v] = p
        mask[b, :v] = True

    return vertices, mask, lengths


def collate_encoded_datasets(items: List[dict]) -> dict:
    """Collate precomputed encoded datasets into padded tensors.

    Stage 2 receives one token per geometry. Because different datasets can
    contain different numbers of geometries, we pad to G_max inside the
    current batch and create a geometry mask.

    Returns:
        geom_embeddings: [B, G_max, D_polygon]
        geom_mask:       [B, G_max]
        geom_centroids:  [B, G_max, 2], dataset-normalized centroids
        geom_features:   [B, G_max, 6], dataset-normalized shape features
        geom_lengths:    [B]

    Raw geometry features are extracted before normalization. Then each
    feature is normalized according to its physical meaning:
        area        -> divide by dataset_scale^2
        perimeter   -> divide by dataset_scale
        bbox width  -> divide by dataset_scale
        bbox height -> divide by dataset_scale
        aspect ratio -> scale-invariant width / height
        vertex count -> log1p count
    """
    lengths = torch.tensor([item["geom_embeddings"].shape[0] for item in items], dtype=torch.long)
    batch_size = len(items)
    gmax = int(lengths.max().item())
    emb_dim = items[0]["geom_embeddings"].shape[-1]
    feat_dim = 6

    geom_embeddings = torch.zeros(batch_size, gmax, emb_dim, dtype=torch.float32)
    geom_centroids = torch.zeros(batch_size, gmax, 2, dtype=torch.float32)
    geom_features = torch.zeros(batch_size, gmax, feat_dim, dtype=torch.float32)
    geom_mask = torch.zeros(batch_size, gmax, dtype=torch.bool)

    for b, item in enumerate(items):
        g = item["geom_embeddings"].shape[0]
        z = item["geom_embeddings"].float()
        c = item["centroids"].float()
        f = item["raw_features"].float()

        # ------------------------------------------------------------------
        # Dataset-level normalization.
        #
        # c contains raw geometry centroids: [G, 2]
        # f contains raw shape features:
        #   f[:, 0] = area
        #   f[:, 1] = perimeter
        #   f[:, 2] = bbox_width
        #   f[:, 3] = bbox_height
        #   f[:, 4] = num_vertices
        #
        # Normalized centroids are used only by the relative geometry-to-geometry
        # attention bias. Normalized shape features are concatenated with the
        # polygon embeddings and passed to the Geometry MLP.
        # ------------------------------------------------------------------

        # Normalize centroids: remove absolute dataset position and scale.
        dataset_center = c.mean(dim=0, keepdim=True)
        centered_centroids = c - dataset_center

        # Dataset scale from centroid spread. If all centroids coincide, fall
        # back to a scale derived from geometry sizes, then to 1.
        centroid_scale = centered_centroids.abs().max()
        bbox_size_scale = f[:, 2:4].max() if f.numel() > 0 else torch.tensor(1.0)
        dataset_scale = torch.maximum(centroid_scale, bbox_size_scale).clamp(min=1e-6)

        c_norm = centered_centroids / dataset_scale

        # Normalize scalar shape features.
        area = f[:, 0:1]
        perimeter = f[:, 1:2]
        bbox_width = f[:, 2:3]
        bbox_height = f[:, 3:4]
        num_vertices = f[:, 4:5]

        area_norm = area / (dataset_scale ** 2)
        perimeter_norm = perimeter / dataset_scale
        bbox_width_norm = bbox_width / dataset_scale
        bbox_height_norm = bbox_height / dataset_scale
        bbox_aspect_ratio = bbox_width / bbox_height.clamp(min=1e-6)
        num_vertices_norm = torch.log1p(num_vertices)

        f_norm = torch.cat(
            [
                area_norm,
                perimeter_norm,
                bbox_width_norm,
                bbox_height_norm,
                bbox_aspect_ratio,
                num_vertices_norm,
            ],
            dim=-1,
        )

        geom_embeddings[b, :g] = z
        geom_centroids[b, :g] = c_norm
        geom_features[b, :g] = f_norm
        geom_mask[b, :g] = True

    return {
        "geom_embeddings": geom_embeddings,
        "geom_centroids": geom_centroids,
        "geom_features": geom_features,
        "geom_mask": geom_mask,
        "geom_lengths": lengths,
    }
