from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import json
from spatial_io import read_wkt_dataset_file
from spatial_utils import normalize_polygon,  polygon_summary
from collate_utils import collate_polygons, collate_encoded_datasets

from pathlib import Path
from typing import Optional, Union

import numpy as np


@torch.no_grad()
def precompute_encoded_datasets(
    polygon_encoder: nn.Module,
    spatial_datasets: List[List[torch.Tensor]],
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> List[dict]:
    """Precompute polygon embeddings for every geometry in every dataset.

    The polygon encoder itself is unchanged. We simply reuse it as the first-stage model.

    Important detail:
    - raw polygons are kept for dataset-level centroids and scalar features;
    - normalized copies are passed to the polygon encoder, because the polygon
      encoder expects centered/scaled vertex coordinates.
    """
    if device is None:
        device = next(polygon_encoder.parameters()).device

    was_training = polygon_encoder.training
    polygon_encoder.eval()

    encoded_items = []
    for polygons in spatial_datasets:
        embeddings = []
        centroids = []
        raw_features = []

        skipped_invalid_polygons = 0

        # Process polygons in small mini-batches for efficiency.  Invalid or
        # degenerate polygons are skipped here so a single bad geometry does not
        # stop preprocessing of the whole dataset.
        for start in range(0, len(polygons), batch_size):
            chunk = polygons[start:start + batch_size]
            normalized_chunk = []
            valid_raw_chunk = []

            for p in chunk:
                try:
                    normalized = normalize_polygon(p)
                except ValueError:
                    skipped_invalid_polygons += 1
                    continue

                if normalized.ndim != 2 or normalized.shape[0] < 3 or normalized.shape[-1] < 2:
                    skipped_invalid_polygons += 1
                    continue

                normalized_chunk.append(normalized)
                valid_raw_chunk.append(p)

            if len(normalized_chunk) == 0:
                continue

            vertices, mask, lengths = collate_polygons(normalized_chunk)
            vertices = vertices.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)

            z = polygon_encoder(vertices, mask, lengths)
            embeddings.append(z.cpu())

            for p in valid_raw_chunk:
                c, f = polygon_summary(p.cpu())
                centroids.append(c)
                raw_features.append(f)

        if len(embeddings) == 0:
            raise ValueError("No valid polygons remained after normalization/sampling.")

        if skipped_invalid_polygons > 0:
            print(f"Skipped {skipped_invalid_polygons} degenerate polygons after normalization/sampling")

        encoded_items.append({
            "geom_embeddings": torch.cat(embeddings, dim=0),
            "centroids": torch.stack(centroids, dim=0),
            "raw_features": torch.stack(raw_features, dim=0),
        })

    if was_training:
        polygon_encoder.train()

    return encoded_items


@torch.no_grad()
def encode_wkt_file_and_save_dataset_embedding(
    wkt_file: str | Path,
    polygon_encoder: nn.Module,
    dataset_encoder: nn.Module,
    output_prefix: str | Path,
    wkt_column: str | None = None,
    polygon_batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> dict:
    """Encode a full spatial dataset from a WKT file and save the resulting dataset embedding.

    Args:
        wkt_file:
            Input file. Plain text files should contain one WKT polygon per line.
            CSV/TSV files should contain a WKT column.
        polygon_encoder:
            Stage-1 polygon encoder. The architecture is exactly the same as above.
        dataset_encoder:
            Stage-2 dataset encoder.
        output_prefix:
            Output path without extension, for example './outputs/my_dataset'.
            The function writes './outputs/my_dataset.pt' and './outputs/my_dataset.npy'.
        wkt_column:
            Optional CSV/TSV column name containing WKT strings.
        polygon_batch_size:
            Mini-batch size used while precomputing polygon embeddings.
        device:
            Device for inference. If None, inferred from the polygon encoder.

    Returns:
        A dictionary with the dataset embedding and file paths.
    """
    wkt_file = Path(wkt_file)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(polygon_encoder.parameters()).device

    polygons = read_wkt_dataset_file(wkt_file, wkt_column=wkt_column)
    print(f"Read {len(polygons)} polygons from {wkt_file}")

    # Stage 1: precompute one polygon embedding per polygon.
    encoded_item = precompute_encoded_datasets(
        polygon_encoder=polygon_encoder,
        spatial_datasets=[polygons],
        batch_size=polygon_batch_size,
        device=device,
    )[0]

    # Stage 2 expects a batch of datasets. Here the batch has B = 1 dataset.
    dataset_batch = collate_encoded_datasets([encoded_item])
    dataset_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in dataset_batch.items()
    }

    polygon_encoder.eval()
    dataset_encoder.eval()

    dataset_embedding = dataset_encoder(
        geom_embeddings=dataset_batch["geom_embeddings"],
        geom_features=dataset_batch["geom_features"],
        geom_centroids=dataset_batch["geom_centroids"],
        geom_mask=dataset_batch["geom_mask"],
    )

    # Remove the batch dimension because B = 1.
    dataset_embedding_1d = dataset_embedding.squeeze(0).detach().cpu()

    pt_path = output_prefix.with_suffix(".pt")
    npy_path = output_prefix.with_suffix(".npy")
    json_path = output_prefix.with_suffix(".json")

    # Save a rich PyTorch artifact.
    torch.save(
        {
            "dataset_embedding": dataset_embedding_1d,
            "dataset_embedding_shape": tuple(dataset_embedding_1d.shape),
            "num_geometries": len(polygons),
            "input_file": str(wkt_file),
            "polygon_embeddings": encoded_item["geom_embeddings"],
            "raw_centroids": encoded_item["centroids"],
            "raw_features": encoded_item["raw_features"],
            "raw_feature_order": ["area", "perimeter", "bbox_width", "bbox_height", "num_vertices"],
            "normalized_feature_order": ["area_norm", "perimeter_norm", "bbox_width_norm", "bbox_height_norm", "bbox_aspect_ratio", "log1p_num_vertices"],
        },
        pt_path,
    )

    # Save a framework-independent NumPy vector.
    np.save(npy_path, dataset_embedding_1d.numpy())

    # Save small human-readable metadata.
    metadata = {
        "input_file": str(wkt_file),
        "num_geometries": len(polygons),
        "dataset_embedding_shape": list(dataset_embedding_1d.shape),
        "pt_output": str(pt_path),
        "npy_output": str(npy_path),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Dataset embedding shape: {tuple(dataset_embedding_1d.shape)}")
    print(f"Saved PyTorch artifact: {pt_path}")
    print(f"Saved NumPy vector:     {npy_path}")
    print(f"Saved metadata:         {json_path}")

    return {
        "embedding": dataset_embedding_1d,
        "pt_path": pt_path,
        "npy_path": npy_path,
        "json_path": json_path,
        "num_geometries": len(polygons),
    }