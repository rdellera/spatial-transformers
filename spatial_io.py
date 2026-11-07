import csv
import torch
import shapely

from typing import List, Tuple, Optional
from pathlib import Path
from spatial_utils import canonicalize_polygon_order
from collate_utils import collate_polygons

try:
    from shapely import wkt as shapely_wkt
except ImportError:
    shapely_wkt = None


def polygon_tensor_from_wkt(wkt_text: str) -> torch.Tensor:
    """Parse one simple POLYGON WKT string into a [V, 2] tensor.

    Constraints enforced here:
    - geometry type must be POLYGON;
    - polygon must not have interior rings / holes;
    - the duplicated closing vertex is removed;
    - orientation is canonicalized to counter-clockwise;
    - original translation and scale are kept for dataset-level geometry features.

    The polygon encoder will receive normalized copies later. The raw polygon is kept here
    because the dataset encoder needs geometry-level centroids, areas, perimeters, etc.
    """
    if shapely_wkt is None:
        raise ImportError(
            "This helper uses shapely for robust WKT parsing. Install it with `pip install shapely`, "
            "or replace polygon_tensor_from_wkt with a project-specific parser."
        )

    geom = shapely_wkt.loads(wkt_text)

    if geom.geom_type != "Polygon":
        raise ValueError(f"Only simple POLYGON WKT is supported here; got {geom.geom_type!r}.")

    if len(geom.interiors) > 0:
        raise ValueError("Polygons with holes are not supported in this simplified version.")

    coords = list(geom.exterior.coords)
    vertices = torch.tensor(coords, dtype=torch.float32)[:, :2]
    vertices = canonicalize_polygon_order(vertices)

    if vertices.shape[0] < 3:
        raise ValueError("A polygon must have at least three distinct vertices.")

    return vertices


def read_wkt_dataset_file(
    file_path: str | Path,
    wkt_column: str | None = None,
    delimiter: str | None = None,
) -> List[torch.Tensor]:
    """Read a spatial dataset from a file where each geometry is a simple polygon in WKT format.

    Args:
        file_path:
            Path to a .txt, .wkt, .csv, or .tsv file.
        wkt_column:
            For CSV/TSV input, the name of the column containing WKT strings. If omitted,
            the function tries to use a column named 'wkt' or 'geometry'.
        delimiter:
            Optional delimiter override for CSV/TSV files.

    Returns:
        A list of polygon tensors. Each tensor has shape [V_i, 2].
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    polygons: List[torch.Tensor] = []

    if suffix in {".csv", ".tsv"}:
        if delimiter is None:
            delimiter = "\t" if suffix == ".tsv" else ","

        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError("CSV/TSV file appears to have no header.")

            if wkt_column is None:
                lower_to_original = {name.lower(): name for name in reader.fieldnames}
                if "wkt" in lower_to_original:
                    wkt_column = lower_to_original["wkt"]
                elif "geometry" in lower_to_original:
                    wkt_column = lower_to_original["geometry"]
                else:
                    raise ValueError(
                        "Could not infer the WKT column. Pass wkt_column='your_column_name'. "
                        f"Available columns: {reader.fieldnames}"
                    )

            for row_idx, row in enumerate(reader, start=2):
                wkt_text = (row.get(wkt_column) or "").strip()
                if not wkt_text:
                    continue
                try:
                    polygons.append(polygon_tensor_from_wkt(wkt_text))
                except Exception as exc:
                    raise ValueError(f"Failed parsing WKT at line {row_idx}: {exc}") from exc

    else:
        # Plain text / .wkt format: one WKT geometry per non-empty line.
        with file_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                wkt_text = line.strip()
                if not wkt_text:
                    continue
                try:
                    polygons.append(polygon_tensor_from_wkt(wkt_text))
                except Exception as exc:
                    raise ValueError(f"Failed parsing WKT at line {line_idx}: {exc}") from exc

    if len(polygons) == 0:
        raise ValueError(f"No polygons were read from {file_path}.")

    return polygons



