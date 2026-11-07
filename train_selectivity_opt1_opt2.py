"""
Train selectivity models that use the learned spatial dataset embedding.

OPT-1:
    End-to-end supervised model:
        raw dataset polygons -> PolygonEncoder -> DatasetEncoder -> z_D
        z_D + range query features -> SelectivityHead -> rq_sel

    The selectivity loss updates the SelectivityHead, DatasetEncoder, and PolygonEncoder.

OPT-2:
    Frozen trained embedding:
        use the encoders trained in OPT-1 to compute z_D once per dataset,
        freeze those embeddings, and train a new SelectivityHead on [z_D, query].

Assumptions:
    - This file is placed in the same project directory as:
        dataset_encoder.py, polygon_encoder.py, collate_utils.py, spatial_utils.py
    - spatial_data_loading.py is available in the same directory and provides
      robust dataset-file resolution and WKT/BZ2 polygon loading helpers.
    - Labels CSV contains:
        dataset,rq_minx,rq_miny,rq_maxx,rq_maxy,rq_sel
      Column names and separator are configurable; the default configuration
      reads AIDA range-query results (rqR_*.csv):
        datasetName;...;minX;minY;maxX;maxY;...;cardinality;...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import copy
from typing import Iterable
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_encoder import DatasetEncoderConfig, DatasetEncoder
from polygon_encoder import PolygonEncoderConfig, PolygonEncoder
from collate_utils import collate_polygons, collate_encoded_datasets
from spatial_utils import normalize_polygon, polygon_summary

# Reuse robust WKT resolution/reading helpers from the embedding-table script.
from spatial_data_loading import resolve_dataset_file, read_polygons, run_dataset_encoder


# ---------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------

# Datasets and labels generated with AIDA (one subfolder of ./data per size tier).
# Paper configuration: DATASETS_DIR = Path("./data/synthetic_datasets"),
# RANGE_QUERY_LABELS_CSV = Path("./data/rq_result_66087_balanced.csv"), separator ",".
DATA_TIER = "small"
DATASETS_DIR = Path("./data") / DATA_TIER
RANGE_QUERY_LABELS_CSV = DATASETS_DIR / f"rqR_{DATA_TIER}.csv"
RANGE_QUERY_LABELS_SEPARATOR = ";"

# AIDA column names (paper: "dataset", "rq_minx", ..., "rq_sel").
# AIDA's "cardinality" column holds the selectivity in [0, 1].
DATASET_NAME_COLUMN = "datasetName"
RANGE_QUERY_COLUMNS = ("minX", "minY", "maxX", "maxY")
TARGET_COLUMN = "cardinality"

WKT_COLUMN = None
DATASET_FILE_NAME = None

# Memory / attention controls.
# Use ATTENTION_STRATEGY = "sampling" to reproduce the previous full-attention
# experiments with random vertex/geometry sampling.
# Use ATTENTION_STRATEGY = "sparse" to use sparse attention and much larger
# safety limits. The safety limits are still random-sampling caps for pathological
# datasets, but sparse attention is the main scalability mechanism.
ATTENTION_STRATEGY = "sampling"  # "sampling" or "sparse"

# One output folder per data tier and attention strategy.
OUTPUT_DIR = Path("./outputs_selectivity") / f"{DATA_TIER}_{ATTENTION_STRATEGY}"

POLYGON_BATCH_SIZE = 1

SAMPLING_MAX_VERTICES_PER_POLYGON = 256
SAMPLING_MAX_GEOMETRIES_PER_DATASET = 512
# Phase-specific vertex caps for sampling mode. Dense attention builds [B,V,V]
# pairwise tensors, so complex real polygons require a much smaller training cap.
# The evaluation cap can be slightly larger, but reduce it too if validation/test
# still trigger CUDA out-of-memory errors.
SAMPLING_TRAIN_VERTICES_PER_POLYGON = 64
SAMPLING_EVAL_VERTICES_PER_POLYGON = 128

SPARSE_MAX_VERTICES_PER_POLYGON = 4096
# Upper cap used while reading/caching sparse-mode datasets. This can be high
# because cached polygon tensors do not store the training autograd graph.
SPARSE_MAX_GEOMETRIES_PER_DATASET = 20000
# Smaller caps used during end-to-end training/evaluation. Sparse attention
# reduces pairwise attention memory, but OPT-1 still backpropagates through
# all selected polygon encodings; these phase-specific caps avoid OOM.
SPARSE_TRAIN_GEOMETRIES_PER_DATASET = 512
SPARSE_EVAL_GEOMETRIES_PER_DATASET = 1024
# Phase-specific vertex caps. Large polygons are randomly vertex-sampled while
# preserving cyclic boundary order. This cap is applied in addition to the cache
# cap, both for sparse and sampling strategies.
SPARSE_TRAIN_VERTICES_PER_POLYGON = 256
SPARSE_EVAL_VERTICES_PER_POLYGON = 512
SPARSE_VERTEX_BOUNDARY_WINDOW = 8
SPARSE_VERTEX_SPATIAL_K = 16
SPARSE_GEOMETRY_SPATIAL_K = 64

MAX_QUERIES_PER_DATASET_PER_EPOCH = 256

# Model sizes.
D_MODEL = 128
POLYGON_EMBEDDING_DIM = 128
DATASET_EMBEDDING_DIM = 128
NUM_HEADS = 4
POLYGON_LAYERS = 3
DATASET_LAYERS = 2
DROPOUT = 0.1

# Training.
RANDOM_SEED = 42
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.15
END_TO_END_EPOCHS = 50
FROZEN_HEAD_EPOCHS = 200
LEARNING_RATE_D = 2e-4
LEARNING_RATE_E = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
FROZEN_BATCH_SIZE = 512
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4
VALIDATE_EVERY_EPOCHS = 1
WMAPE_LOSS_WEIGHT = 0.7
SMOOTH_L1_LOSS_WEIGHT = 0.3

# Debug/progress controls.
# Set DEBUG_MAX_DATASETS to a small integer, e.g. 5, to test the full
# pipeline on a small subset before running the full experiment.
DEBUG_MAX_DATASETS: int | None = None
VERBOSE = True



# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def log(message: str) -> None:
    """Print a progress message immediately, avoiding buffered output."""
    if VERBOSE:
        print(message, flush=True)


class TimingStats:
    """Collect wall-clock timings for the main execution phases."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, elapsed: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + float(elapsed)
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> list[tuple[str, float, int, float]]:
        rows = []
        for name, total in sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True):
            count = self.counts.get(name, 1)
            rows.append((name, total, count, total / max(count, 1)))
        return rows

    def print_summary(self) -> None:
        print("\nTiming summary", flush=True)
        print("phase,total_s,count,avg_s", flush=True)
        for name, total, count, avg in self.summary():
            print(f"{name},{total:.3f},{count},{avg:.3f}", flush=True)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"phase": name, "total_s": total, "count": count, "avg_s": avg}
                for name, total, count, avg in self.summary()
            ]
        )


TIMINGS = TimingStats()


def sync_if_cuda(device: torch.device | None = None) -> None:
    """Synchronize CUDA before reading a timer, so GPU time is measured correctly."""
    if device is not None and isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


class timed_phase:
    """Context manager that records a phase duration in TIMINGS."""

    def __init__(self, name: str, device: torch.device | None = None, log_start: bool = False) -> None:
        self.name = name
        self.device = device
        self.log_start = log_start
        self.t0 = 0.0

    def __enter__(self):
        if self.log_start:
            log(f"Starting phase: {self.name}")
        sync_if_cuda(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        sync_if_cuda(self.device)
        elapsed = time.perf_counter() - self.t0
        TIMINGS.add(self.name, elapsed)
        if self.log_start:
            log(f"Finished phase: {self.name} in {elapsed:.2f}s")
        return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def target_to_logit(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = torch.clamp(y, eps, 1.0 - eps)
    return torch.log(y / (1.0 - y))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = float(np.sum(np.abs(y_true)))
    wmape = float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "WMAPE": wmape}


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name}", flush=True)
    print(f"MAE   : {metrics['MAE']}", flush=True)
    print(f"RMSE  : {metrics['RMSE']}", flush=True)
    print(f"R2    : {metrics['R2']}", flush=True)
    print(f"WMAPE : {metrics['WMAPE']} %", flush=True)


def compute_dataset_spatial_stats(polygon_cache: dict[str, list[torch.Tensor]]) -> dict[str, dict[str, np.ndarray | float]]:
    """Compute per-dataset center and scale for query normalization.

    The query window is expressed in the same local coordinate system for each
    dataset by subtracting a dataset center and dividing by a dataset scale.
    This makes query coordinates comparable across datasets with different
    extents.
    """
    stats: dict[str, dict[str, np.ndarray | float]] = {}
    for dataset_name, polygons in polygon_cache.items():
        if not polygons:
            continue
        mins = []
        maxs = []
        for poly in polygons:
            xy = poly[:, :2].float()
            mins.append(torch.min(xy, dim=0).values)
            maxs.append(torch.max(xy, dim=0).values)
        min_xy = torch.stack(mins, dim=0).min(dim=0).values.cpu().numpy().astype(np.float32)
        max_xy = torch.stack(maxs, dim=0).max(dim=0).values.cpu().numpy().astype(np.float32)
        center = 0.5 * (min_xy + max_xy)
        width_height = max_xy - min_xy
        scale = float(max(float(width_height[0]), float(width_height[1]), 1e-6))
        stats[dataset_name] = {"center": center, "scale": scale, "min_xy": min_xy, "max_xy": max_xy}
    return stats


def make_query_features(
    df: pd.DataFrame,
    dataset_spatial_stats: dict[str, dict[str, np.ndarray | float]] | None = None,
) -> np.ndarray:
    """Return query features.

    Features include the range-query coordinates plus derived width, height,
    area and center. If dataset_spatial_stats is provided, coordinates are first
    normalized per dataset: (coord - dataset_center) / dataset_scale.
    """
    q = df.loc[:, RANGE_QUERY_COLUMNS].astype(float).to_numpy(dtype=np.float32)
    minx, miny, maxx, maxy = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    if dataset_spatial_stats is not None:
        dataset_names = df[DATASET_NAME_COLUMN].astype(str).to_numpy()
        minx_n = np.empty_like(minx, dtype=np.float32)
        miny_n = np.empty_like(miny, dtype=np.float32)
        maxx_n = np.empty_like(maxx, dtype=np.float32)
        maxy_n = np.empty_like(maxy, dtype=np.float32)
        missing = set()
        for i, name in enumerate(dataset_names):
            stat = dataset_spatial_stats.get(str(name))
            if stat is None:
                missing.add(str(name))
                center = np.array([0.0, 0.0], dtype=np.float32)
                scale = 1.0
            else:
                center = np.asarray(stat["center"], dtype=np.float32)
                scale = float(stat["scale"])
            minx_n[i] = (minx[i] - center[0]) / scale
            miny_n[i] = (miny[i] - center[1]) / scale
            maxx_n[i] = (maxx[i] - center[0]) / scale
            maxy_n[i] = (maxy[i] - center[1]) / scale
        if missing:
            log(f"Warning: missing dataset spatial stats for {len(missing)} datasets; using raw scale for those rows")
        minx, miny, maxx, maxy = minx_n, miny_n, maxx_n, maxy_n

    width = maxx - minx
    height = maxy - miny
    area = width * height
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    return np.column_stack([minx, miny, maxx, maxy, width, height, area, cx, cy]).astype(np.float32)


def wmape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sum(torch.abs(pred - target)) / (torch.sum(torch.abs(target)) + eps)


def selectivity_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    return (
        WMAPE_LOSS_WEIGHT * wmape_loss(pred, target)
        + SMOOTH_L1_LOSS_WEIGHT * F.smooth_l1_loss(pred, target)
    )


@dataclass
class QueryNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "QueryNormalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def split_by_dataset(df: pd.DataFrame, seed: int, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = sorted(df[DATASET_NAME_COLUMN].dropna().astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(names)
    n_test = max(1, int(round(len(names) * test_fraction)))
    test_names = set(names[:n_test])
    is_test = df[DATASET_NAME_COLUMN].astype(str).isin(test_names)
    return df.loc[~is_test].reset_index(drop=True), df.loc[is_test].reset_index(drop=True)


def random_sample_list(items: list, max_items: int | None, seed: int) -> list:
    if max_items is None or len(items) <= max_items:
        return items
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(items)), max_items))
    return [items[i] for i in idx]


def get_active_memory_limits() -> tuple[int | None, int | None]:
    """Return the vertex and geometry caps for the selected strategy."""
    if ATTENTION_STRATEGY == "sampling":
        return SAMPLING_MAX_VERTICES_PER_POLYGON, SAMPLING_MAX_GEOMETRIES_PER_DATASET
    if ATTENTION_STRATEGY == "sparse":
        return SPARSE_MAX_VERTICES_PER_POLYGON, SPARSE_MAX_GEOMETRIES_PER_DATASET
    raise ValueError(f"Unknown ATTENTION_STRATEGY: {ATTENTION_STRATEGY!r}")


def get_encoder_attention_mode() -> str:
    """Return the encoder attention mode corresponding to the selected strategy."""
    if ATTENTION_STRATEGY == "sampling":
        return "full"
    if ATTENTION_STRATEGY == "sparse":
        return "sparse"
    raise ValueError(f"Unknown ATTENTION_STRATEGY: {ATTENTION_STRATEGY!r}")


def get_phase_geometry_cap(phase: str) -> int | None:
    """Return the geometry cap used for a given computation phase.

    In sampling mode the same cap is used everywhere. In sparse mode the cache
    may keep many geometries, but OPT-1 training/evaluation uses smaller
    random subsets to keep the autograd graph within GPU memory.
    """
    if ATTENTION_STRATEGY == "sampling":
        return SAMPLING_MAX_GEOMETRIES_PER_DATASET
    if ATTENTION_STRATEGY != "sparse":
        raise ValueError(f"Unknown ATTENTION_STRATEGY: {ATTENTION_STRATEGY!r}")

    if phase == "train":
        return SPARSE_TRAIN_GEOMETRIES_PER_DATASET
    if phase in {"eval", "frozen"}:
        return SPARSE_EVAL_GEOMETRIES_PER_DATASET
    if phase == "cache":
        return SPARSE_MAX_GEOMETRIES_PER_DATASET
    raise ValueError(f"Unknown phase: {phase!r}")


def get_phase_vertex_cap(phase: str) -> int | None:
    """Return the vertex cap used for a given computation phase.

    The cap is applied by random vertex sampling while preserving boundary
    order. In sparse mode, the cache may keep polygons with many vertices, but
    OPT-1 training/evaluation uses smaller vertex subsets to avoid OOM.
    """
    if ATTENTION_STRATEGY == "sampling":
        if phase == "train":
            return SAMPLING_TRAIN_VERTICES_PER_POLYGON
        if phase in {"eval", "frozen"}:
            return SAMPLING_EVAL_VERTICES_PER_POLYGON
        if phase == "cache":
            return SAMPLING_MAX_VERTICES_PER_POLYGON
        raise ValueError(f"Unknown phase: {phase!r}")

    if ATTENTION_STRATEGY != "sparse":
        raise ValueError(f"Unknown ATTENTION_STRATEGY: {ATTENTION_STRATEGY!r}")

    if phase == "train":
        return SPARSE_TRAIN_VERTICES_PER_POLYGON
    if phase in {"eval", "frozen"}:
        return SPARSE_EVAL_VERTICES_PER_POLYGON
    if phase == "cache":
        return SPARSE_MAX_VERTICES_PER_POLYGON
    raise ValueError(f"Unknown phase: {phase!r}")


def sample_polygon_vertices_for_phase(
    vertices: torch.Tensor,
    max_vertices: int | None,
    seed: int,
) -> torch.Tensor:
    """Randomly sample polygon vertices while preserving cyclic order.

    The sampled indices are sorted before slicing so the polygon boundary order
    remains valid. If max_vertices is None or the polygon is already small
    enough, the input tensor is returned unchanged.
    """
    if max_vertices is None or vertices.shape[0] <= max_vertices:
        return vertices
    if max_vertices < 3:
        raise ValueError("max_vertices_per_polygon must be at least 3")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(vertices.shape[0]), max_vertices))
    return vertices[idx]


def sample_polygon_vertices_list_for_phase(
    polygons: list[torch.Tensor],
    max_vertices: int | None,
    seed: int,
) -> list[torch.Tensor]:
    """Apply phase-specific random vertex sampling to all polygons."""
    if max_vertices is None:
        return polygons
    sampled: list[torch.Tensor] = []
    changed = 0
    for poly_idx, polygon in enumerate(polygons):
        reduced = sample_polygon_vertices_for_phase(
            polygon,
            max_vertices=max_vertices,
            seed=seed + poly_idx,
        )
        if reduced.shape[0] != polygon.shape[0]:
            changed += 1
        sampled.append(reduced)
    if changed:
        TIMINGS.add("vertex_random_sampling_polygons", 0.0)
    return sampled


def sample_polygons_for_phase(
    polygons: list[torch.Tensor],
    max_geometries: int | None,
    seed: int,
) -> list[torch.Tensor]:
    """Return a reproducible random subset of polygons for a computation phase.

    The sampled indices are sorted so the selected geometries keep their original
    file order. This reduces OPT-1 memory while preserving deterministic runs.
    """
    if max_geometries is None or len(polygons) <= max_geometries:
        return polygons
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(polygons)), max_geometries))
    return [polygons[i] for i in idx]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_labels() -> pd.DataFrame:
    log(f"Loading labels from: {RANGE_QUERY_LABELS_CSV}")
    with timed_phase("load_labels", log_start=False):
        df = pd.read_csv(RANGE_QUERY_LABELS_CSV, sep=RANGE_QUERY_LABELS_SEPARATOR, compression="infer")
    log(f"Loaded labels: {len(df)} rows")
    required = [DATASET_NAME_COLUMN, *RANGE_QUERY_COLUMNS, TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {RANGE_QUERY_LABELS_CSV}: {missing}")
    df = df.dropna(subset=required).copy()
    df[DATASET_NAME_COLUMN] = df[DATASET_NAME_COLUMN].astype(str)
    log(f"Labels after dropping missing required fields: {len(df)} rows")
    log(f"Unique datasets in labels: {df[DATASET_NAME_COLUMN].nunique()}")
    return df


def load_polygon_cache(
    dataset_names: Iterable[str],
    max_vertices_per_polygon: int | None,
    max_geometries_per_dataset: int | None,
) -> dict[str, list[torch.Tensor]]:
    """Load and filter polygons for each dataset once.

    Datasets with missing files or no valid polygons are skipped. Progress is
    printed before and after each dataset so long WKT/BZ2 parsing phases are
    visible even before CUDA work starts.
    """
    all_names = sorted(set(map(str, dataset_names)))
    if DEBUG_MAX_DATASETS is not None:
        all_names = all_names[:DEBUG_MAX_DATASETS]
        log(f"DEBUG_MAX_DATASETS={DEBUG_MAX_DATASETS}: using only {len(all_names)} datasets")

    cache: dict[str, list[torch.Tensor]] = {}
    skipped: list[str] = []
    total = len(all_names)

    log(f"Starting dataset loading/filtering for {total} datasets")
    t_all = time.perf_counter()

    for idx, dataset_name in enumerate(all_names, start=1):
        t0 = time.perf_counter()
        try:
            log(f"[{idx}/{total}] Resolving dataset: {dataset_name}")
            with timed_phase("dataset_file_resolution"):
                dataset_file = resolve_dataset_file(
                    datasets_dir=DATASETS_DIR,
                    dataset_name=dataset_name,
                    dataset_file_name=DATASET_FILE_NAME,
                )
            log(f"[{idx}/{total}] Reading polygons from: {dataset_file}")
            with timed_phase("wkt_read_filter_vertex_sampling"):
                polygons = read_polygons(
                    dataset_file,
                    wkt_column=WKT_COLUMN,
                    max_vertices_per_polygon=max_vertices_per_polygon,
                )
            before_sampling = len(polygons)
            with timed_phase("geometry_random_sampling"):
                polygons = random_sample_list(
                    polygons,
                    max_items=max_geometries_per_dataset,
                    seed=RANDOM_SEED + abs(hash(dataset_name)) % 1_000_000,
                )
            if before_sampling != len(polygons):
                log(
                    f"[{idx}/{total}] Random-sampled geometries for {dataset_name}: "
                    f"{before_sampling} -> {len(polygons)}"
                )
            if len(polygons) == 0:
                log(f"[{idx}/{total}] Skipping {dataset_name!r}: no valid polygons after filtering/sampling")
                skipped.append(dataset_name)
                continue
            cache[dataset_name] = polygons
            elapsed = time.perf_counter() - t0
            log(f"[{idx}/{total}] Loaded {dataset_name}: {len(polygons)} polygons in {elapsed:.2f}s")
        except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
            elapsed = time.perf_counter() - t0
            log(f"[{idx}/{total}] Skipping {dataset_name!r} after {elapsed:.2f}s: {exc}")
            skipped.append(dataset_name)

    elapsed_all = time.perf_counter() - t_all
    TIMINGS.add("dataset_loading_total", elapsed_all)
    log(f"Finished dataset loading in {elapsed_all:.2f}s")
    log(f"Loaded datasets: {len(cache)}")
    log(f"Skipped datasets: {len(skipped)}")
    return cache



def _unpack_polygon_batch(polygon_batch):
    """Return (vertices, mask, lengths) for both tuple- and dict-style collate outputs."""
    if isinstance(polygon_batch, dict):
        return polygon_batch["vertices"], polygon_batch["mask"], polygon_batch["lengths"]
    if isinstance(polygon_batch, (tuple, list)) and len(polygon_batch) == 3:
        return polygon_batch[0], polygon_batch[1], polygon_batch[2]
    raise TypeError(
        "collate_polygons must return either a dict with keys "
        "'vertices', 'mask', 'lengths' or a tuple/list (vertices, mask, lengths). "
        f"Got: {type(polygon_batch)!r}"
    )

# ---------------------------------------------------------------------
# Differentiable embedding computation
# ---------------------------------------------------------------------


def encode_dataset_differentiable(
    polygons: list[torch.Tensor],
    polygon_encoder: nn.Module,
    dataset_encoder: nn.Module,
    device: torch.device,
    polygon_batch_size: int,
    max_vertices_per_polygon: int | None = None,
    vertex_sampling_seed: int = RANDOM_SEED,
) -> torch.Tensor:
    """Compute z_D with gradients through PolygonEncoder and DatasetEncoder.

    Large polygons are randomly vertex-sampled according to the phase-specific
    cap before batching. This is applied in both sampling and sparse strategies.
    """
    with timed_phase("phase_vertex_random_sampling"):
        original_vertex_counts = [int(p.shape[0]) for p in polygons]
        polygons = sample_polygon_vertices_list_for_phase(
            polygons,
            max_vertices=max_vertices_per_polygon,
            seed=vertex_sampling_seed,
        )
        sampled_vertex_counts = [int(p.shape[0]) for p in polygons]
    if any(a != b for a, b in zip(original_vertex_counts, sampled_vertex_counts)):
        changed = sum(a != b for a, b in zip(original_vertex_counts, sampled_vertex_counts))
        log(
            f"    vertex subset: sampled {changed} polygons; "
            f"max vertices {max(original_vertex_counts)} -> {max(sampled_vertex_counts)}"
        )

    polygon_embeddings: list[torch.Tensor] = []
    centroids: list[torch.Tensor] = []
    raw_features: list[torch.Tensor] = []
    skipped_invalid_polygons = 0

    for start in range(0, len(polygons), polygon_batch_size):
        chunk = polygons[start:start + polygon_batch_size]

        # Some real-world geometries can become degenerate after cleaning,
        # duplicate-closing-vertex removal, or vertex sampling.  Normalize only
        # polygons that still have a valid simple polygon representation.
        normalized_chunk: list[torch.Tensor] = []
        valid_raw_chunk: list[torch.Tensor] = []
        with timed_phase("stage1_normalize_polygons", device=device):
            for poly in chunk:
                try:
                    normalized = normalize_polygon(poly)
                except ValueError:
                    skipped_invalid_polygons += 1
                    continue

                if normalized.ndim != 2 or normalized.shape[0] < 3 or normalized.shape[-1] < 2:
                    skipped_invalid_polygons += 1
                    continue

                normalized_chunk.append(normalized)
                valid_raw_chunk.append(poly)

        if len(normalized_chunk) == 0:
            continue

        with timed_phase("stage1_collate_polygons", device=device):
            polygon_batch = collate_polygons(normalized_chunk)
            vertices, mask, lengths = _unpack_polygon_batch(polygon_batch)

            vertices = vertices.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)

        with timed_phase("stage1_polygon_encoder_forward", device=device):
            z = polygon_encoder(vertices, mask, lengths)
        polygon_embeddings.append(z)

        with timed_phase("stage2_polygon_summary_features", device=device):
            for poly in valid_raw_chunk:
                centroid, features = polygon_summary(poly)
                centroids.append(torch.as_tensor(centroid, dtype=torch.float32))
                raw_features.append(torch.as_tensor(features, dtype=torch.float32))

    if skipped_invalid_polygons > 0:
        log(f"    skipped {skipped_invalid_polygons} degenerate polygons after normalization/sampling")

    if len(polygon_embeddings) == 0:
        raise ValueError("No valid polygons remained after normalization/sampling.")

    geom_embeddings = torch.cat(polygon_embeddings, dim=0)

    # Use collate_encoded_datasets for feature normalization/masks, but overwrite
    # the embedding tensor afterwards so the gradient path from geom_embeddings is preserved.
    item = {
        "polygon_embeddings": geom_embeddings.detach().cpu(),
        "geom_embeddings": geom_embeddings.detach().cpu(),
        "centroids": torch.stack(centroids, dim=0),
        "raw_features": torch.stack(raw_features, dim=0),
    }
    with timed_phase("stage2_collate_encoded_dataset", device=device):
        batch = collate_encoded_datasets([item])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        emb_batched = geom_embeddings.unsqueeze(0)
        batch["polygon_embeddings"] = emb_batched
        batch["geom_embeddings"] = emb_batched

    with timed_phase("stage2_dataset_encoder_forward", device=device):
        z_dataset = run_dataset_encoder(dataset_encoder, batch, device)
    return z_dataset.squeeze(0)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


class SelectivityHead(nn.Module):
    """Prediction head that explicitly models interactions between z_D and query."""

    def __init__(self, dataset_dim: int, query_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.query_mlp = nn.Sequential(
            nn.Linear(query_dim, dataset_dim),
            nn.GELU(),
            nn.Linear(dataset_dim, dataset_dim),
        )
        self.pred_mlp = nn.Sequential(
            nn.Linear(dataset_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, dataset_embedding: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        q = self.query_mlp(query_features)
        x = torch.cat([
            dataset_embedding,
            q,
            dataset_embedding * q,
            torch.abs(dataset_embedding - q),
        ], dim=-1)
        return self.pred_mlp(x).squeeze(-1)


def build_encoders(device: torch.device) -> tuple[PolygonEncoder, DatasetEncoder]:
    polygon_config = PolygonEncoderConfig(
        d_model=D_MODEL,
        embedding_dim=POLYGON_EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=POLYGON_LAYERS,
        dropout=DROPOUT,
        attention_mode=get_encoder_attention_mode(),
        sparse_boundary_window=SPARSE_VERTEX_BOUNDARY_WINDOW,
        sparse_spatial_k=SPARSE_VERTEX_SPATIAL_K,
    )
    polygon_encoder = PolygonEncoder(polygon_config).to(device)

    dataset_config = DatasetEncoderConfig(
        polygon_embedding_dim=POLYGON_EMBEDDING_DIM,
        geom_feature_dim=6,
        d_model=D_MODEL,
        embedding_dim=DATASET_EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=DATASET_LAYERS,
        dropout=DROPOUT,
        attention_mode=get_encoder_attention_mode(),
        sparse_spatial_k=SPARSE_GEOMETRY_SPATIAL_K,
    )
    dataset_encoder = DatasetEncoder(dataset_config).to(device)
    return polygon_encoder, dataset_encoder


# ---------------------------------------------------------------------
# OPT-1: end-to-end training
# ---------------------------------------------------------------------


def train_opt1_end_to_end(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    polygon_cache: dict[str, list[torch.Tensor]],
    query_norm: QueryNormalizer,
    dataset_spatial_stats: dict[str, dict[str, np.ndarray | float]],
    device: torch.device,
) -> tuple[PolygonEncoder, DatasetEncoder, SelectivityHead]:
    polygon_encoder, dataset_encoder = build_encoders(device)
    head = SelectivityHead(
        dataset_dim=DATASET_EMBEDDING_DIM,
        query_dim=make_query_features(train_df, dataset_spatial_stats).shape[1],
        dropout=DROPOUT,
    ).to(device)

    params = list(polygon_encoder.parameters()) + list(dataset_encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE_D, weight_decay=WEIGHT_DECAY)

    train_groups = {name: g.reset_index(drop=True) for name, g in train_df.groupby(DATASET_NAME_COLUMN)}

    best_wmape = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, END_TO_END_EPOCHS + 1):
        epoch_t0 = time.perf_counter()
        polygon_encoder.train()
        dataset_encoder.train()
        head.train()

        names = list(train_groups.keys())
        random.shuffle(names)
        losses = []
        log(f"OPT-1 epoch {epoch:03d}: training on {len(names)} datasets")

        for ds_idx, dataset_name in enumerate(names, start=1):
            if dataset_name not in polygon_cache:
                continue
            if ds_idx == 1 or ds_idx % 10 == 0 or ds_idx == len(names):
                log(f"  epoch {epoch:03d} dataset {ds_idx}/{len(names)}: {dataset_name}")
            rows = train_groups[dataset_name]
            if len(rows) > MAX_QUERIES_PER_DATASET_PER_EPOCH:
                rows = rows.sample(n=MAX_QUERIES_PER_DATASET_PER_EPOCH, random_state=RANDOM_SEED + epoch)

            q_np = query_norm.transform(make_query_features(rows, dataset_spatial_stats))
            y_np = rows[TARGET_COLUMN].astype(float).to_numpy(dtype=np.float32)

            q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
            y = torch.as_tensor(y_np, dtype=torch.float32, device=device)

            with timed_phase("opt1_train_dataset_step", device=device):
                optimizer.zero_grad(set_to_none=True)
                polygons_for_train = sample_polygons_for_phase(
                    polygon_cache[dataset_name],
                    max_geometries=get_phase_geometry_cap("train"),
                    seed=RANDOM_SEED + epoch * 1_000_000 + ds_idx,
                )
                if len(polygons_for_train) < len(polygon_cache[dataset_name]) and (ds_idx == 1 or ds_idx % 10 == 0):
                    log(
                        f"    training geometry subset: {len(polygon_cache[dataset_name])} -> "
                        f"{len(polygons_for_train)}"
                    )
                z = encode_dataset_differentiable(
                    polygons_for_train,
                    polygon_encoder,
                    dataset_encoder,
                    device=device,
                    polygon_batch_size=POLYGON_BATCH_SIZE,
                    max_vertices_per_polygon=get_phase_vertex_cap("train"),
                    vertex_sampling_seed=RANDOM_SEED + epoch * 2_000_000 + ds_idx,
                )
                z_rep = z.unsqueeze(0).expand(q.shape[0], -1)
                pred_logit = head(z_rep, q)
                loss = selectivity_loss_from_logits(pred_logit, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))

        epoch_elapsed = time.perf_counter() - epoch_t0
        TIMINGS.add("opt1_epoch_total", epoch_elapsed)
        log(f"OPT-1 epoch {epoch:03d} | train loss {np.mean(losses):.6f} | time {epoch_elapsed:.2f}s")

        if epoch == END_TO_END_EPOCHS or epoch % VALIDATE_EVERY_EPOCHS == 0:
            metrics = evaluate_end_to_end(
                test_df, polygon_cache, polygon_encoder, dataset_encoder, head, query_norm, dataset_spatial_stats, device
            )
            print_metrics(f"OPT-1 validation epoch {epoch}", metrics)

            current_wmape = metrics["WMAPE"]
            if current_wmape + EARLY_STOPPING_MIN_DELTA < best_wmape:
                best_wmape = current_wmape
                patience_counter = 0
                best_state = {
                    "polygon_encoder": copy.deepcopy(polygon_encoder.state_dict()),
                    "dataset_encoder": copy.deepcopy(dataset_encoder.state_dict()),
                    "head": copy.deepcopy(head.state_dict()),
                }
                log(f"OPT-1 new best WMAPE: {best_wmape:.4f}%")
            else:
                patience_counter += 1
                log(f"OPT-1 early-stopping patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    log(f"OPT-1 early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        polygon_encoder.load_state_dict(best_state["polygon_encoder"])
        dataset_encoder.load_state_dict(best_state["dataset_encoder"])
        head.load_state_dict(best_state["head"])
        log(f"Restored OPT-1 best checkpoint with WMAPE {best_wmape:.4f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "polygon_encoder": polygon_encoder.state_dict(),
            "dataset_encoder": dataset_encoder.state_dict(),
            "head": head.state_dict(),
            "query_mean": query_norm.mean,
            "query_std": query_norm.std,
        },
        OUTPUT_DIR / "opt1_end_to_end_checkpoint.pt",
    )

    return polygon_encoder, dataset_encoder, head


@torch.no_grad()
def evaluate_end_to_end(
    df: pd.DataFrame,
    polygon_cache: dict[str, list[torch.Tensor]],
    polygon_encoder: nn.Module,
    dataset_encoder: nn.Module,
    head: nn.Module,
    query_norm: QueryNormalizer,
    dataset_spatial_stats: dict[str, dict[str, np.ndarray | float]],
    device: torch.device,
) -> dict[str, float]:
    polygon_encoder.eval()
    dataset_encoder.eval()
    head.eval()

    eval_t0 = time.perf_counter()
    y_true_all = []
    y_pred_all = []

    groups = list(df.groupby(DATASET_NAME_COLUMN))
    log(f"Evaluating end-to-end on {len(groups)} datasets")
    for eval_idx, (dataset_name, rows) in enumerate(groups, start=1):
        if eval_idx == 1 or eval_idx % 10 == 0 or eval_idx == len(groups):
            log(f"  eval dataset {eval_idx}/{len(groups)}: {dataset_name}")
        if dataset_name not in polygon_cache:
            continue
        polygons_for_eval = sample_polygons_for_phase(
            polygon_cache[dataset_name],
            max_geometries=get_phase_geometry_cap("eval"),
            seed=RANDOM_SEED + 9_000_000 + eval_idx,
        )
        z = encode_dataset_differentiable(
            polygons_for_eval,
            polygon_encoder,
            dataset_encoder,
            device,
            POLYGON_BATCH_SIZE,
            max_vertices_per_polygon=get_phase_vertex_cap("eval"),
            vertex_sampling_seed=RANDOM_SEED + 11_000_000 + eval_idx,
        )
        q_np = query_norm.transform(make_query_features(rows, dataset_spatial_stats))
        q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
        z_rep = z.unsqueeze(0).expand(q.shape[0], -1)
        pred = torch.sigmoid(head(z_rep, q)).detach().cpu().numpy()
        y = rows[TARGET_COLUMN].astype(float).to_numpy()
        y_true_all.append(y)
        y_pred_all.append(pred)

    metrics = compute_metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))
    TIMINGS.add("opt1_evaluation_total", time.perf_counter() - eval_t0)
    return metrics


# ---------------------------------------------------------------------
# OPT-2: freeze trained encoders, train a new head on frozen embeddings
# ---------------------------------------------------------------------


@torch.no_grad()
def compute_frozen_embeddings(
    dataset_names: Iterable[str],
    polygon_cache: dict[str, list[torch.Tensor]],
    polygon_encoder: nn.Module,
    dataset_encoder: nn.Module,
    device: torch.device,
) -> dict[str, np.ndarray]:
    polygon_encoder.eval()
    dataset_encoder.eval()
    embeddings = {}
    names = sorted(set(map(str, dataset_names)))
    frozen_t0 = time.perf_counter()
    log(f"Computing frozen embeddings for {len(names)} datasets")
    for idx, dataset_name in enumerate(names, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(names):
            log(f"  frozen embedding {idx}/{len(names)}: {dataset_name}")
        if dataset_name not in polygon_cache:
            continue
        polygons_for_frozen = sample_polygons_for_phase(
            polygon_cache[dataset_name],
            max_geometries=get_phase_geometry_cap("frozen"),
            seed=RANDOM_SEED + 10_000_000 + idx,
        )
        z = encode_dataset_differentiable(
            polygons_for_frozen,
            polygon_encoder,
            dataset_encoder,
            device,
            POLYGON_BATCH_SIZE,
            max_vertices_per_polygon=get_phase_vertex_cap("frozen"),
            vertex_sampling_seed=RANDOM_SEED + 12_000_000 + idx,
        )
        embeddings[dataset_name] = z.detach().cpu().numpy().astype(np.float32)
    TIMINGS.add("opt2_compute_frozen_embeddings_total", time.perf_counter() - frozen_t0)
    return embeddings


def build_frozen_matrix(
    df: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    query_norm: QueryNormalizer,
    dataset_spatial_stats: dict[str, dict[str, np.ndarray | float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    targets = []
    dataset_names = []
    for _, row in df.iterrows():
        name = str(row[DATASET_NAME_COLUMN])
        if name not in embeddings:
            continue
        q = query_norm.transform(make_query_features(pd.DataFrame([row]), dataset_spatial_stats))
        x = np.concatenate([embeddings[name], q.reshape(-1)], axis=0)
        rows.append(x)
        targets.append(float(row[TARGET_COLUMN]))
        dataset_names.append(name)
    return np.vstack(rows).astype(np.float32), np.asarray(targets, dtype=np.float32), np.asarray(dataset_names)


class FrozenEmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_opt2_frozen(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    polygon_cache: dict[str, list[torch.Tensor]],
    trained_polygon_encoder: nn.Module,
    trained_dataset_encoder: nn.Module,
    query_norm: QueryNormalizer,
    dataset_spatial_stats: dict[str, dict[str, np.ndarray | float]],
    device: torch.device,
) -> FrozenEmbeddingMLP:
    for p in trained_polygon_encoder.parameters():
        p.requires_grad_(False)
    for p in trained_dataset_encoder.parameters():
        p.requires_grad_(False)

    all_names = list(train_df[DATASET_NAME_COLUMN].astype(str).unique()) + list(test_df[DATASET_NAME_COLUMN].astype(str).unique())
    embeddings = compute_frozen_embeddings(all_names, polygon_cache, trained_polygon_encoder, trained_dataset_encoder, device)

    X_train, y_train, _ = build_frozen_matrix(train_df, embeddings, query_norm, dataset_spatial_stats)
    X_test, y_test, _ = build_frozen_matrix(test_df, embeddings, query_norm, dataset_spatial_stats)

    model = FrozenEmbeddingMLP(input_dim=X_train.shape[1], dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_E, weight_decay=WEIGHT_DECAY)

    x_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    n = x_train.shape[0]
    best_wmape = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, FROZEN_HEAD_EPOCHS + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, FROZEN_BATCH_SIZE):
            idx = perm[start:start + FROZEN_BATCH_SIZE]
            xb = x_train[idx]
            yb = y_train_t[idx]
            optimizer.zero_grad(set_to_none=True)
            pred_logit = model(xb)
            loss = selectivity_loss_from_logits(pred_logit, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        epoch_elapsed = time.perf_counter() - epoch_t0
        TIMINGS.add("opt2_epoch_total", epoch_elapsed)
        if epoch == FROZEN_HEAD_EPOCHS or epoch % VALIDATE_EVERY_EPOCHS == 0:
            metrics = evaluate_frozen_model(model, X_test, y_test, device)
            log(f"OPT-2 epoch {epoch:03d} | train loss {np.mean(losses):.6f} | time {epoch_elapsed:.2f}s")
            print_metrics(f"OPT-2 validation epoch {epoch}", metrics)

            current_wmape = metrics["WMAPE"]
            if current_wmape + EARLY_STOPPING_MIN_DELTA < best_wmape:
                best_wmape = current_wmape
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
                log(f"OPT-2 new best WMAPE: {best_wmape:.4f}%")
            else:
                patience_counter += 1
                log(f"OPT-2 early-stopping patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    log(f"OPT-2 early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"Restored OPT-2 best checkpoint with WMAPE {best_wmape:.4f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"head": model.state_dict()}, OUTPUT_DIR / "opt2_frozen_head.pt")
    return model


@torch.no_grad()
def evaluate_frozen_model(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    preds = []
    for start in range(0, x.shape[0], FROZEN_BATCH_SIZE):
        pred = torch.sigmoid(model(x[start:start + FROZEN_BATCH_SIZE])).detach().cpu().numpy()
        preds.append(pred)
    y_pred = np.concatenate(preds)
    return compute_metrics(y, y_pred)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    main_t0 = time.perf_counter()
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("Starting selectivity training script")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"Using device: {device}")
    active_max_vertices, active_max_geometries = get_active_memory_limits()
    cache_max_vertices = get_phase_vertex_cap("cache")
    cache_max_geometries = get_phase_geometry_cap("cache")
    log(f"ATTENTION_STRATEGY={ATTENTION_STRATEGY}")
    log(f"Encoder attention_mode={get_encoder_attention_mode()}")
    log(f"Memory controls: POLYGON_BATCH_SIZE={POLYGON_BATCH_SIZE}, "
        f"active_MAX_VERTICES_PER_POLYGON={active_max_vertices}, "
        f"active_MAX_GEOMETRIES_PER_DATASET={active_max_geometries}")
    if ATTENTION_STRATEGY == "sparse":
        log(
            f"Sparse params: boundary_window={SPARSE_VERTEX_BOUNDARY_WINDOW}, "
            f"vertex_spatial_k={SPARSE_VERTEX_SPATIAL_K}, "
            f"geometry_spatial_k={SPARSE_GEOMETRY_SPATIAL_K}")
        log(
            f"Sparse phase caps: train_geometries={SPARSE_TRAIN_GEOMETRIES_PER_DATASET}, "
            f"eval_geometries={SPARSE_EVAL_GEOMETRIES_PER_DATASET}, "
            f"cache_geometries={SPARSE_MAX_GEOMETRIES_PER_DATASET}")
    if torch.cuda.is_available():
        log(f"CUDA device count: {torch.cuda.device_count()}")
        log(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    with timed_phase("main_load_labels", log_start=True):
        labels = load_labels()

    dataset_names = labels[DATASET_NAME_COLUMN].astype(str).unique()
    log("Loading polygon cache. This phase is CPU-bound; nvidia-smi may show no process yet.")

    with timed_phase("main_load_polygon_cache", log_start=True):
        polygon_cache = load_polygon_cache(
            dataset_names,
            max_vertices_per_polygon=cache_max_vertices,
            max_geometries_per_dataset=cache_max_geometries,
        )

    labels = labels[labels[DATASET_NAME_COLUMN].astype(str).isin(polygon_cache.keys())].reset_index(drop=True)
    log(f"Rows remaining after skipping unavailable/invalid datasets: {len(labels)}")

    with timed_phase("main_compute_dataset_spatial_stats", log_start=True):
        dataset_spatial_stats = compute_dataset_spatial_stats(polygon_cache)
    log(f"Computed dataset spatial stats for {len(dataset_spatial_stats)} datasets")

    with timed_phase("main_split_and_query_normalization"):
        train_full_df, test_df = split_by_dataset(labels, seed=RANDOM_SEED, test_fraction=TEST_FRACTION)
        train_df, val_df = split_by_dataset(train_full_df, seed=RANDOM_SEED + 1, test_fraction=VALIDATION_FRACTION)
        query_norm = QueryNormalizer.fit(make_query_features(train_df, dataset_spatial_stats))

    log(f"Rows train: {len(train_df)}")
    log(f"Rows validation: {len(val_df)}")
    log(f"Rows test : {len(test_df)}")
    log(f"Datasets train: {train_df[DATASET_NAME_COLUMN].nunique()}")
    log(f"Datasets validation: {val_df[DATASET_NAME_COLUMN].nunique()}")
    log(f"Datasets test : {test_df[DATASET_NAME_COLUMN].nunique()}")

    with timed_phase("main_opt1_training", device=device, log_start=True):
        polygon_encoder, dataset_encoder, end_to_end_head = train_opt1_end_to_end(
            train_df=train_df,
            test_df=val_df,
            polygon_cache=polygon_cache,
            query_norm=query_norm,
            dataset_spatial_stats=dataset_spatial_stats,
            device=device,
        )

    with timed_phase("main_opt1_final_evaluation", device=device, log_start=True):
        final_opt1_metrics = evaluate_end_to_end(
            test_df,
            polygon_cache,
            polygon_encoder,
            dataset_encoder,
            end_to_end_head,
            query_norm,
            dataset_spatial_stats,
            device,
        )
    print_metrics("FINAL OPT-1: end-to-end trained embedding", final_opt1_metrics)

    with timed_phase("main_opt2_training", device=device, log_start=True):
        frozen_head = train_opt2_frozen(
            train_df=train_df,
            test_df=val_df,
            polygon_cache=polygon_cache,
            trained_polygon_encoder=polygon_encoder,
            trained_dataset_encoder=dataset_encoder,
            query_norm=query_norm,
            dataset_spatial_stats=dataset_spatial_stats,
            device=device,
        )

    with timed_phase("main_opt2_final_evaluation", device=device, log_start=True):
        embeddings = compute_frozen_embeddings(
            test_df[DATASET_NAME_COLUMN].astype(str).unique(),
            polygon_cache,
            polygon_encoder,
            dataset_encoder,
            device,
        )
        X_test, y_test, _ = build_frozen_matrix(test_df, embeddings, query_norm, dataset_spatial_stats)
        final_opt2_metrics = evaluate_frozen_model(frozen_head, X_test, y_test, device)
    print_metrics("FINAL OPT-2: frozen trained embedding + new head", final_opt2_metrics)

    TIMINGS.add("total_script_wall_time", time.perf_counter() - main_t0)
    TIMINGS.print_summary()
    timing_csv = OUTPUT_DIR / "timing_summary.csv"
    TIMINGS.to_dataframe().to_csv(timing_csv, index=False)
    log(f"Saved timing summary to: {timing_csv}")


if __name__ == "__main__":
    main()
