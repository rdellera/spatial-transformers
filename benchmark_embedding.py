"""
Measure time and memory needed to build dataset embeddings (z_D).

For every AIDA dataset under BENCHMARK_DIR and every configuration
(attention strategy x sampling caps), the script runs the same path used by
compute_frozen_embeddings in train_selectivity_opt1_opt2.py:

    read_polygons -> geometry/vertex caps -> PolygonEncoder -> DatasetEncoder -> z_D

Encoders keep the paper configuration and random weights (the cost does not
depend on the weights). Each measurement runs in its own process, so the RAM
peak belongs to that measurement only and an out-of-memory error does not stop
the benchmark.

Dataset layout (generated with AIDA):
    BENCHMARK_DIR/g<geometries>_v<maxseg>/uniform_<replica>.wkt
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import queue
import time
from pathlib import Path

import psutil
import torch

import train_selectivity_opt1_opt2 as paper
from spatial_data_loading import read_polygons


# ---------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------

BENCHMARK_DIR = Path("./data/benchmark")
OUTPUT_CSV = Path("./outputs_benchmark/embedding_benchmark.csv")

GEOMETRY_COUNTS = (100, 1000, 10000, 100000)
VERTEX_BANDS = (5, 10, 20, 200, 2000)  # AIDA maxseg: each polygon has 4..maxseg vertices

# (attention strategy, use the paper sampling caps)
CONFIGURATIONS = (
    ("sampling", True),
    ("sparse", True),
    ("sampling", False),
    ("sparse", False),
)

MAX_SECONDS_PER_RUN = 1800

CSV_FIELDS = [
    "group", "dataset", "geometries", "max_vertices", "strategy", "caps", "status",
    "polygons_read", "polygons_used", "vertices_used", "max_vertices_used",
    "read_s", "stage1_s", "stage2_s", "total_s", "gpu_peak_mb", "ram_peak_mb",
]


# ---------------------------------------------------------------------
# Single measurement (runs in a child process)
# ---------------------------------------------------------------------


def measure_embedding(dataset_file: Path, strategy: str, use_paper_caps: bool, results: mp.Queue) -> None:
    """Build z_D for one dataset and put the measured resources in ``results``."""
    # Same effect as editing ATTENTION_STRATEGY in the paper configuration block:
    # build_encoders and the phase caps read this module constant.
    paper.ATTENTION_STRATEGY = strategy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paper.set_seed(paper.RANDOM_SEED)
    polygon_encoder, dataset_encoder = paper.build_encoders(device)
    polygon_encoder.eval()
    dataset_encoder.eval()

    def caps(phase: str) -> tuple[int | None, int | None]:
        if not use_paper_caps:
            return None, None
        return paper.get_phase_vertex_cap(phase), paper.get_phase_geometry_cap(phase)

    def encode(polygons: list[torch.Tensor], max_vertices: int | None) -> torch.Tensor:
        return paper.encode_dataset_differentiable(
            polygons,
            polygon_encoder,
            dataset_encoder,
            device=device,
            polygon_batch_size=paper.POLYGON_BATCH_SIZE,
            max_vertices_per_polygon=max_vertices,
            vertex_sampling_seed=paper.RANDOM_SEED,
        )

    cache_vertices, cache_geometries = caps("cache")
    frozen_vertices, frozen_geometries = caps("frozen")
    result: dict = {"status": "ok"}
    try:
        with torch.no_grad():
            # Warm-up: CUDA context and kernels are initialized outside the measurement.
            encode([torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])] * 4, None)
            paper.TIMINGS.totals.clear()
            paper.TIMINGS.counts.clear()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            t0 = time.perf_counter()
            polygons = read_polygons(dataset_file, max_vertices_per_polygon=cache_vertices)
            result["polygons_read"] = len(polygons)
            polygons = paper.random_sample_list(polygons, cache_geometries, seed=paper.RANDOM_SEED)
            polygons = paper.sample_polygons_for_phase(polygons, frozen_geometries, seed=paper.RANDOM_SEED)
            result["read_s"] = time.perf_counter() - t0

            z_dataset = encode(polygons, frozen_vertices)
            paper.sync_if_cuda(device)
            result["total_s"] = time.perf_counter() - t0
            assert z_dataset.shape == (paper.DATASET_EMBEDDING_DIM,)

        used = [min(p.shape[0], frozen_vertices or p.shape[0]) for p in polygons]
        result.update(polygons_used=len(polygons), vertices_used=sum(used), max_vertices_used=max(used))
        result["stage1_s"] = sum(t for name, t in paper.TIMINGS.totals.items() if name.startswith("stage1"))
        result["stage2_s"] = sum(t for name, t in paper.TIMINGS.totals.items() if name.startswith("stage2"))
    except torch.OutOfMemoryError:
        result["status"] = "oom_gpu"
    except MemoryError:
        result["status"] = "oom_ram"

    if device.type == "cuda":
        result["gpu_peak_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    memory = psutil.Process().memory_info()
    result["ram_peak_mb"] = getattr(memory, "peak_wset", memory.rss) / 2**20  # peak_wset exists on Windows
    results.put(result)


def run_isolated(dataset_file: Path, strategy: str, use_paper_caps: bool) -> dict:
    """Run measure_embedding in a fresh process with a time limit."""
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    process = ctx.Process(target=measure_embedding, args=(dataset_file, strategy, use_paper_caps, results))
    process.start()
    process.join(MAX_SECONDS_PER_RUN)
    if process.is_alive():
        process.kill()
        process.join()
        return {"status": "timeout"}
    try:
        return results.get(timeout=5)
    except queue.Empty:
        # The process died without reporting (e.g. killed by the OS when RAM runs out).
        paper.log(f"    process exited with code {process.exitcode} without results")
        return {"status": "crash"}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, bool, int, int]] = []  # (strategy, caps, geometries, max_vertices)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for geometries in GEOMETRY_COUNTS:
            for max_vertices in VERTEX_BANDS:
                group = f"g{geometries}_v{max_vertices}"
                dataset_files = sorted((BENCHMARK_DIR / group).glob("*.wkt"))
                if not dataset_files:
                    paper.log(f"{group}: no datasets, skipped")
                    continue

                for strategy, use_paper_caps in CONFIGURATIONS:
                    # A configuration that failed on a smaller input cannot succeed on a larger one.
                    already_failed = any(
                        s == strategy and c == use_paper_caps and g <= geometries and v <= max_vertices
                        for s, c, g, v in failures
                    )
                    for dataset_file in dataset_files:
                        if already_failed:
                            result = {"status": "skipped"}
                        else:
                            result = run_isolated(dataset_file, strategy, use_paper_caps)
                        row = {
                            "group": group,
                            "dataset": dataset_file.stem,
                            "geometries": geometries,
                            "max_vertices": max_vertices,
                            "strategy": strategy,
                            "caps": "paper" if use_paper_caps else "none",
                            **result,
                        }
                        writer.writerow(row)
                        fout.flush()
                        paper.log(
                            f"{group} {dataset_file.stem} {strategy}/{row['caps']}: {row['status']} "
                            f"total={row.get('total_s', float('nan')):.2f}s "
                            f"gpu={row.get('gpu_peak_mb', float('nan')):.0f}MB "
                            f"ram={row.get('ram_peak_mb', float('nan')):.0f}MB"
                        )
                        if result["status"] not in ("ok", "skipped"):
                            failures.append((strategy, use_paper_caps, geometries, max_vertices))
                            already_failed = True

    paper.log(f"Saved benchmark results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
