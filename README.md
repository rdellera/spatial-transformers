# Spatial Embedding Transformers

Transformer-based spatial embeddings for variable-size polygon datasets, with supervised range-query selectivity experiments.

The repository implements a two-stage architecture:

1. **Polygon encoder**: encodes each simple polygon into a fixed-size polygon embedding without requiring a fixed number of vertices.
2. **Dataset encoder**: aggregates polygon embeddings into a fixed-size dataset embedding without requiring a fixed number of geometries.

The training script supports two experimental prediction settings:

- **OPT-1**: end-to-end supervised training. The polygon encoder, dataset encoder, query MLP, and prediction head are trained together for range-query selectivity estimation.
- **OPT-2**: frozen trained embedding. The encoders trained in OPT-1 are frozen, dataset embeddings are precomputed, and a new prediction head is trained on top of the frozen embeddings and query features.

The code also supports two scalability strategies:

- **sampling**: randomly samples vertices/geometries before using dense attention.
- **sparse**: uses sparse attention neighborhoods while keeping phase-specific safety caps for vertices and geometries.

## Repository structure

```text
train_selectivity_opt1_opt2.py   Main experiment script for OPT-1 and OPT-2
spatial_data_loading.py          Dataset-file resolution and WKT/BZ2 polygon loading helpers
polygon_encoder.py               Stage 1 polygon encoder
dataset_encoder.py               Stage 2 dataset encoder
polygon_bias.py                  Relative spatial + cyclic boundary bias for polygon vertices
dataset_bias.py                  Relative spatial bias for dataset geometries
transformer_utils.py             Dense and sparse relative-bias attention utilities
fourier_features.py              Fourier feature expansion
masked_pooling.py                Masked attention pooling
collate_utils.py                 Polygon and dataset collation utilities
spatial_utils.py                 Polygon normalization and geometry feature extraction
spatial_io.py                    Basic WKT I/O helpers
embedding.py                     Precomputation helpers
benchmark_embedding.py           Time/memory benchmark for building dataset embeddings
convert_csv_to_wkt.py            Utility: convert MBR CSV files to WKT polygons
process_osm_file.py              Utility: extract first-column WKT geometries from tab-separated files
```

Legacy experiment files whose names started with `test_` were removed. The reusable loading functions are now in `spatial_data_loading.py`.

## Input data

The main training script expects:

1. A directory containing one spatial dataset per file.
2. A CSV file containing range-query labels.

Each dataset file may be plain text or BZ2-compressed. The loader supports files such as:

```text
.wkt
.txt
.csv
.tsv
.wkt.bz2
.txt.bz2
.csv.bz2
.tsv.bz2
```

When `WKT_COLUMN = None`, each dataset file is assumed to contain one WKT geometry per line.

The current implementation keeps only simple `POLYGON` geometries without holes. Other geometry types, such as `POINT`, `LINESTRING`, `MULTIPOLYGON`, and polygons with holes, are skipped.

The range-query CSV must contain at least:

```text
dataset,rq_minx,rq_miny,rq_maxx,rq_maxy,rq_sel
```

where:

- `dataset` identifies the dataset file;
- `rq_minx`, `rq_miny`, `rq_maxx`, `rq_maxy` define the range-query window;
- `rq_sel` is the selectivity target.

Column names and separator are configurable through `DATASET_NAME_COLUMN`, `RANGE_QUERY_COLUMNS`, `TARGET_COLUMN` and `RANGE_QUERY_LABELS_SEPARATOR`. The current configuration reads range-query results produced by the AIDA tool (`rqR_<tier>.csv`, separator `;`):

```text
datasetName;numQuery;queryArea;minX;minY;maxX;maxY;areaint;cardinality;...
```

where `cardinality` is the selectivity (fraction of geometries intersecting the query window).

## Installation

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the PyTorch build matching your CUDA version from the official PyTorch installation selector, then install the remaining dependencies.

## Running the main experiment

Edit the configuration block at the top of `train_selectivity_opt1_opt2.py`:

```python
DATA_TIER = "small"
DATASETS_DIR = Path("./data") / DATA_TIER
RANGE_QUERY_LABELS_CSV = DATASETS_DIR / f"rqR_{DATA_TIER}.csv"
RANGE_QUERY_LABELS_SEPARATOR = ";"

ATTENTION_STRATEGY = "sampling"  # "sampling" or "sparse"
OUTPUT_DIR = Path("./outputs_selectivity") / f"{DATA_TIER}_{ATTENTION_STRATEGY}"
```

The paper experiments used `DATASETS_DIR = Path("./data/synthetic_datasets")`, `RANGE_QUERY_LABELS_CSV = Path("./data/rq_result_66087_balanced.csv")` and separator `,`.

Then run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u train_selectivity_opt1_opt2.py 2>&1 | tee run.log
```

The `-u` flag forces unbuffered output, so progress messages appear immediately.

## Sampling versus sparse attention

Use:

```python
ATTENTION_STRATEGY = "sampling"
```

for dense attention on randomly sampled vertices/geometries.

Use:

```python
ATTENTION_STRATEGY = "sparse"
```

for sparse attention. In sparse mode:

- Stage 1 uses cyclic-boundary neighbors plus spatial nearest neighbors between vertices.
- Stage 2 uses centroid-based nearest neighbors between geometries.

Important parameters include:

```python
SAMPLING_MAX_VERTICES_PER_POLYGON
SAMPLING_MAX_GEOMETRIES_PER_DATASET
SAMPLING_TRAIN_VERTICES_PER_POLYGON
SAMPLING_EVAL_VERTICES_PER_POLYGON

SPARSE_MAX_VERTICES_PER_POLYGON
SPARSE_MAX_GEOMETRIES_PER_DATASET
SPARSE_TRAIN_GEOMETRIES_PER_DATASET
SPARSE_EVAL_GEOMETRIES_PER_DATASET
SPARSE_TRAIN_VERTICES_PER_POLYGON
SPARSE_EVAL_VERTICES_PER_POLYGON
SPARSE_VERTEX_BOUNDARY_WINDOW
SPARSE_VERTEX_SPATIAL_K
SPARSE_GEOMETRY_SPATIAL_K
```

If CUDA out-of-memory errors occur in the polygon encoder, reduce:

```python
POLYGON_BATCH_SIZE
SAMPLING_TRAIN_VERTICES_PER_POLYGON
SAMPLING_EVAL_VERTICES_PER_POLYGON
SPARSE_TRAIN_VERTICES_PER_POLYGON
SPARSE_VERTEX_BOUNDARY_WINDOW
SPARSE_VERTEX_SPATIAL_K
```

For complex real polygons in dense sampling mode, a safe starting point is:

```python
POLYGON_BATCH_SIZE = 1
SAMPLING_TRAIN_VERTICES_PER_POLYGON = 64
SAMPLING_EVAL_VERTICES_PER_POLYGON = 128
```

The training and embedding code also skips degenerate polygons that become invalid after duplicate-closing-vertex removal, cleaning, or vertex sampling. If all polygons in a dataset are filtered out, that dataset is skipped rather than stopping the experiment.

If CUDA out-of-memory errors occur in the dataset encoder, reduce:

```python
SPARSE_TRAIN_GEOMETRIES_PER_DATASET
SPARSE_GEOMETRY_SPATIAL_K
```

## OPT-1 and OPT-2

### OPT-1: end-to-end supervised training

OPT-1 computes the dataset embedding from raw polygons and combines it with a learned query embedding. The loss is backpropagated through:

```text
prediction head -> dataset encoder -> polygon encoder
```

This trains the embedding to be useful for range-query selectivity estimation.

### OPT-2: frozen trained embedding

OPT-2 reuses the encoders trained in OPT-1. It freezes them, computes dataset embeddings once, and trains a new prediction head on top of the frozen embeddings.

This setting evaluates whether the learned embeddings can be reused as fixed dataset representations.

## Query features

The script normalizes query windows with respect to dataset-level spatial statistics and builds the query vector:

```text
rq_minx, rq_miny, rq_maxx, rq_maxy, rq_width, rq_height, rq_area, rq_cx, rq_cy
```

The prediction head uses an interaction vector:

```text
[z_D, q_D, z_D * q_D, |z_D - q_D|]
```

where `z_D` is the dataset embedding and `q_D` is the learned query embedding.

## Metrics and timing

The script reports:

```text
MAE
RMSE
R2
WMAPE
```

It also records timing statistics for loading, training, evaluation, and frozen-embedding computation. A timing summary is printed at the end and written to:

```text
outputs_selectivity/<tier>_<strategy>/timing_summary.csv
```

## Embedding resource benchmark

`benchmark_embedding.py` measures the time and memory needed to build a dataset embedding `z_D`, without training. It follows the same path as `compute_frozen_embeddings` (polygon reading, sampling caps, polygon encoder, dataset encoder), with the paper model configuration and random weights.

Datasets are generated with the AIDA tool and stored as:

```text
data/benchmark/g<geometries>_v<maxseg>/uniform_<replica>.wkt
```

where `maxseg` is the maximum number of vertices per polygon (AIDA draws 4..maxseg vertices). Every dataset is measured with four configurations: `sampling` and `sparse` attention, each with the paper sampling caps (`caps = paper`) and without them (`caps = none`). Each measurement runs in a separate process, with a limit of `MAX_SECONDS_PER_RUN` seconds.

### Generating the benchmark datasets

The `.wkt` files are not versioned (about 13 GB, 11 GB of which for `g100000_v2000`). Each `data/benchmark/<group>/` folder contains the AIDA summary (`sum_<group>.csv`) and `commands.log`, where line `n` is the `Generator.py` command that produced `uniform_<n>.wkt`.

Datasets are generated with `Generator.py` from the [AIDA tool](https://github.com/smigliorini/aida-augumentation-tool), branch `main` (commit `e4a2548`). It only uses the Python standard library, but it imports `cgi`, so it needs Python 3.12 or older. From this repository root:

```bash
AIDA=/path/to/aida-augumentation-tool
for dir in data/benchmark/*/; do
  n=1
  tr -d '\r' < "$dir/commands.log" | while read -r command; do
    (cd "$AIDA" && $command) > "$dir/uniform_$n.wkt"
    n=$((n + 1))
  done
done
```

Alternatively, run AIDA's `GeneratorCSV.py` on each `sum_<group>.csv`, after editing its `MODIFICA` constants. AIDA does not fix a random seed, so regenerated polygons differ from the original ones, but the number of geometries and vertices per polygon (which determine the cost) follow the same parameters.

### Running the benchmark

```bash
mkdir -p outputs_benchmark
python -u benchmark_embedding.py 2>&1 | tee outputs_benchmark/run.log
```

On the reference laptop (see `benchmark.md`) the full grid takes about 2.5–3 hours. Results are written to `outputs_benchmark/embedding_benchmark.csv`, one row per measurement, overwriting the versioned results of the thesis:

```text
group, dataset, geometries, max_vertices, strategy, caps, status,
polygons_read, polygons_used, vertices_used, max_vertices_used,
read_s, stage1_s, stage2_s, total_s, gpu_peak_mb, ram_peak_mb
```

`status` is one of `ok`, `oom_gpu`, `oom_ram`, `crash`, `timeout`, `skipped` (the same configuration already failed on a smaller input). Test results are collected in `benchmark.md`; `embedding_benchmark-2.csv` and `run-2.log` hold the second test (100000 geometries).

## Utility scripts

### Convert MBR CSV files to WKT polygons

Use `convert_csv_to_wkt.py` when each input CSV row contains:

```text
min_x,min_y,max_x,max_y
```

and you want one WKT polygon per row.

### Extract WKT geometries from tab-separated OSM-like files

Use `process_osm_file.py` when the first tab-separated column contains WKT geometries and you want to export one `.wkt` file per input CSV.

## Notes

- The model currently supports only simple polygons without holes.
- Full attention has quadratic memory cost in the number of vertices/geometries.
- Sparse attention reduces pairwise attention memory, but end-to-end OPT-1 training can still be memory intensive because gradients must be retained through the selected polygon and dataset encodings.
- For large experiments, first run with `DEBUG_MAX_DATASETS` set to a small number to validate the pipeline.
