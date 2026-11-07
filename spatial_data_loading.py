"""Robust dataset-file resolution and WKT polygon loading utilities.

This module contains only reusable I/O helpers. It is used by the main OPT-1/OPT-2 training script.
"""

from __future__ import annotations

from pathlib import Path
import bz2
import random

import pandas as pd
import torch
import torch.nn as nn

try:
    from shapely import wkt as shapely_wkt
except ImportError:  # pragma: no cover - handled at runtime
    shapely_wkt = None

from spatial_io import read_wkt_dataset_file


DEFAULT_RANDOM_SEED = 42


def _strip_known_dataset_suffixes(name_path: Path) -> tuple[Path, str]:
    """Return parent and base name after removing common dataset suffixes.

    Examples:
        dataset.csv -> ("", "dataset")
        dataset.wkt.bz2 -> ("", "dataset")
        subdir/dataset.csv.bz2 -> ("subdir", "dataset")
    """
    parent = name_path.parent if name_path.parent != Path(".") else Path("")
    name = name_path.name
    for suffix in (
        ".wkt.bz2",
        ".csv.bz2",
        ".tsv.bz2",
        ".txt.bz2",
        ".wkt",
        ".csv",
        ".tsv",
        ".txt",
    ):
        if name.lower().endswith(suffix):
            return parent, name[: -len(suffix)]
    return parent, name_path.stem


def open_text_maybe_bz2(path: str | Path):
    """Open plain-text or bzip2-compressed text files transparently."""
    path = Path(path)
    if path.name.lower().endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _infer_table_separator(path: str | Path) -> str:
    """Infer separator for CSV/TSV files, including compressed names."""
    name = Path(path).name.lower()
    if name.endswith(".tsv") or name.endswith(".tsv.bz2"):
        return "\t"
    return ","


def resolve_dataset_file(
    datasets_dir: str | Path,
    dataset_name: str,
    dataset_file_name: str | None = None,
) -> Path:
    """Resolve the file associated with a dataset name.

    If the labels CSV contains ``dataset_01.csv``, this function also searches
    for ``dataset_01.wkt`` inside ``datasets_dir``. Relative subdirectories are
    preserved.
    """
    datasets_dir = Path(datasets_dir)
    name_path = Path(str(dataset_name).strip())
    parent, base_name = _strip_known_dataset_suffixes(name_path)

    candidates: list[Path] = []
    candidates.append(datasets_dir / name_path)

    for ext in (
        ".wkt",
        ".txt",
        ".csv",
        ".tsv",
        ".wkt.bz2",
        ".txt.bz2",
        ".csv.bz2",
        ".tsv.bz2",
    ):
        candidates.append(datasets_dir / parent / f"{base_name}{ext}")

    dataset_dir = datasets_dir / parent / base_name
    if dataset_file_name is not None:
        candidates.append(dataset_dir / dataset_file_name)

    for default_name in (
        "polygons.wkt",
        "geometries.wkt",
        "dataset.wkt",
        "polygons.wkt.bz2",
        "geometries.wkt.bz2",
        "dataset.wkt.bz2",
    ):
        candidates.append(dataset_dir / default_name)

    for path in candidates:
        if path.is_file():
            return path

    if dataset_dir.is_dir():
        for pattern in (
            "*.wkt",
            "*.txt",
            "*.csv",
            "*.tsv",
            "*.wkt.bz2",
            "*.txt.bz2",
            "*.csv.bz2",
            "*.tsv.bz2",
        ):
            matches = sorted(dataset_dir.glob(pattern))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"Could not resolve dataset file for dataset_name={dataset_name!r} "
        f"inside {datasets_dir}. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def _sample_tensor_rows_preserving_order(
    tensor: torch.Tensor,
    max_rows: int | None,
    rng: random.Random,
) -> tuple[torch.Tensor, bool]:
    """Randomly sample rows while preserving their original order."""
    if max_rows is None or tensor.shape[0] <= max_rows:
        return tensor, False
    if max_rows < 3:
        raise ValueError("max_rows must be at least 3 for polygon vertices")
    indices = sorted(rng.sample(range(tensor.shape[0]), max_rows))
    return tensor[indices], True


def _polygon_to_vertex_tensor(geom) -> torch.Tensor | None:
    """Convert a supported polygon representation to a [V, 2] tensor.

    Only simple polygons without holes are retained. MultiPolygons, Points,
    LineStrings, polygons with holes, and invalid coordinate sequences are
    skipped by returning ``None``.
    """
    if torch.is_tensor(geom):
        if geom.ndim == 2 and geom.shape[-1] >= 2 and geom.shape[0] >= 3:
            return geom[:, :2].float()
        return None

    geom_type = getattr(geom, "geom_type", None)
    if geom_type is not None:
        if geom_type != "Polygon":
            return None
        if len(getattr(geom, "interiors", [])) > 0:
            return None
        coords = list(geom.exterior.coords)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            return None
        return torch.tensor(coords, dtype=torch.float32)[:, :2]

    try:
        tensor = torch.as_tensor(geom, dtype=torch.float32)
    except Exception:
        return None
    if tensor.ndim == 2 and tensor.shape[-1] >= 2 and tensor.shape[0] >= 3:
        return tensor[:, :2]
    return None


def _read_wkt_lines_polygons_only(
    dataset_file: str | Path,
    max_vertices_per_polygon: int | None = None,
    rng: random.Random | None = None,
) -> list[torch.Tensor]:
    """Read one WKT geometry per line and keep only simple POLYGON geometries."""
    if shapely_wkt is None:
        raise ImportError("shapely is required for line-by-line polygon filtering")

    polygons: list[torch.Tensor] = []
    skipped = 0
    sampled_too_large = 0
    dataset_file = Path(dataset_file)
    rng = rng or random.Random(DEFAULT_RANDOM_SEED)

    with open_text_maybe_bz2(dataset_file) as fin:
        for line_no, line in enumerate(fin, start=1):
            text = line.strip()
            if not text:
                continue
            if not text.upper().startswith("POLYGON"):
                skipped += 1
                continue
            try:
                geom = shapely_wkt.loads(text)
            except Exception as exc:
                raise ValueError(
                    f"Failed parsing WKT at line {line_no} in {dataset_file}: {text[:200]}"
                ) from exc

            tensor = _polygon_to_vertex_tensor(geom)
            if tensor is None:
                skipped += 1
                continue

            tensor, was_sampled = _sample_tensor_rows_preserving_order(
                tensor,
                max_vertices_per_polygon,
                rng,
            )
            if was_sampled:
                sampled_too_large += 1
            polygons.append(tensor)

    if skipped:
        print(f"Skipped {skipped} non-supported geometries in {dataset_file.name}")
    if sampled_too_large:
        print(
            f"Randomly sampled vertices for {sampled_too_large} polygons with more than "
            f"{max_vertices_per_polygon} vertices in {dataset_file.name}"
        )
    return polygons


def _read_wkt_column_polygons_only(
    dataset_file: str | Path,
    wkt_column: str,
    max_vertices_per_polygon: int | None = None,
    rng: random.Random | None = None,
) -> list[torch.Tensor]:
    """Read a CSV/TSV file, parse the selected WKT column, and keep polygons."""
    if shapely_wkt is None:
        raise ImportError("shapely is required for WKT-column polygon filtering")

    dataset_file = Path(dataset_file)
    sep = _infer_table_separator(dataset_file)
    df = pd.read_csv(dataset_file, sep=sep, compression="infer")

    if wkt_column not in df.columns:
        raise ValueError(
            f"Column {wkt_column!r} not found in {dataset_file}. "
            f"Available columns: {list(df.columns)}"
        )

    polygons: list[torch.Tensor] = []
    skipped = 0
    sampled_too_large = 0
    rng = rng or random.Random(DEFAULT_RANDOM_SEED)
    for row_no, value in enumerate(df[wkt_column].dropna(), start=1):
        text = str(value).strip()
        if not text:
            continue
        if not text.upper().startswith("POLYGON"):
            skipped += 1
            continue
        try:
            geom = shapely_wkt.loads(text)
        except Exception as exc:
            raise ValueError(
                f"Failed parsing WKT in {dataset_file}, row {row_no}: {text[:200]}"
            ) from exc
        tensor = _polygon_to_vertex_tensor(geom)
        if tensor is None:
            skipped += 1
            continue

        tensor, was_sampled = _sample_tensor_rows_preserving_order(
            tensor,
            max_vertices_per_polygon,
            rng,
        )
        if was_sampled:
            sampled_too_large += 1
        polygons.append(tensor)

    if skipped:
        print(f"Skipped {skipped} non-supported geometries in {dataset_file.name}")
    if sampled_too_large:
        print(
            f"Randomly sampled vertices for {sampled_too_large} polygons with more than "
            f"{max_vertices_per_polygon} vertices in {dataset_file.name}"
        )
    return polygons


def read_polygons(
    dataset_file: str | Path,
    wkt_column: str | None = None,
    max_vertices_per_polygon: int | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> list[torch.Tensor]:
    """Read a dataset file and return only simple polygons as vertex tensors.

    If ``wkt_column`` is ``None``, the file is assumed to contain one WKT
    geometry per line. If no valid simple polygons are found, an empty list is
    returned and the caller decides whether to skip the dataset.
    """
    dataset_file = Path(dataset_file)
    rng = random.Random(f"{random_seed}:{dataset_file}")

    if shapely_wkt is not None:
        if wkt_column is None:
            return _read_wkt_lines_polygons_only(
                dataset_file,
                max_vertices_per_polygon=max_vertices_per_polygon,
                rng=rng,
            )
        return _read_wkt_column_polygons_only(
            dataset_file,
            wkt_column,
            max_vertices_per_polygon=max_vertices_per_polygon,
            rng=rng,
        )

    try:
        geometries = read_wkt_dataset_file(dataset_file, wkt_column=wkt_column)
    except TypeError:
        geometries = read_wkt_dataset_file(dataset_file)

    polygons: list[torch.Tensor] = []
    skipped = 0
    sampled_too_large = 0
    for geom in geometries:
        tensor = _polygon_to_vertex_tensor(geom)
        if tensor is None:
            skipped += 1
            continue
        tensor, was_sampled = _sample_tensor_rows_preserving_order(
            tensor,
            max_vertices_per_polygon,
            rng,
        )
        if was_sampled:
            sampled_too_large += 1
        polygons.append(tensor)

    if skipped:
        print(f"Skipped {skipped} non-supported geometries in {dataset_file.name}")
    if sampled_too_large:
        print(
            f"Randomly sampled vertices for {sampled_too_large} polygons with more than "
            f"{max_vertices_per_polygon} vertices in {dataset_file.name}"
        )
    return polygons


def run_dataset_encoder(dataset_encoder: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    """Call DatasetEncoder while supporting common forward argument names."""
    embedding_key = "geom_embeddings" if "geom_embeddings" in batch else "polygon_embeddings"
    common_kwargs = {
        "geom_features": batch["geom_features"],
        "geom_centroids": batch["geom_centroids"],
        "geom_mask": batch["geom_mask"],
    }
    try:
        return dataset_encoder(geom_embeddings=batch[embedding_key], **common_kwargs)
    except TypeError:
        return dataset_encoder(polygon_embeddings=batch[embedding_key], **common_kwargs)

