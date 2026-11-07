import random
from typing import Tuple
import torch


def remove_duplicate_closing_vertex(vertices: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Remove final vertex if it duplicates the first one.

    Args:
        vertices: Tensor of shape [V, 2].
    """
    if vertices.shape[0] > 1 and torch.linalg.norm(vertices[0] - vertices[-1]) < eps:
        return vertices[:-1]
    return vertices


def signed_area(vertices: torch.Tensor) -> torch.Tensor:
    """Signed polygon area using the shoelace (or Gauss' area) formula.

    Positive area means counter-clockwise order for standard x/y coordinates.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_next = torch.roll(x, shifts=-1, dims=0)
    y_next = torch.roll(y, shifts=-1, dims=0)
    return 0.5 * torch.sum(x * y_next - x_next * y)


def polygon_centroid(vertices: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Area-weighted centroid for a simple polygon.

    Falls back to the arithmetic mean if the area is too small.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_next = torch.roll(x, shifts=-1, dims=0)
    y_next = torch.roll(y, shifts=-1, dims=0)
    cross = x * y_next - x_next * y
    area2 = torch.sum(cross)  # equals 2 * signed area

    if torch.abs(area2) < eps:
        return vertices.mean(dim=0)

    cx = torch.sum((x + x_next) * cross) / (3.0 * area2)
    cy = torch.sum((y + y_next) * cross) / (3.0 * area2)
    return torch.stack([cx, cy])


def normalize_polygon(vertices: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a simple polygon while preserving its shape.

    Steps:
    - remove duplicated closing vertex;
    - force counter-clockwise orientation;
    - subtract polygon centroid;
    - divide by a scale derived from area, falling back to bbox diagonal.

    The returned tensor has shape [V, 2].
    """
    vertices = remove_duplicate_closing_vertex(vertices.float())

    if vertices.shape[0] < 3:
        raise ValueError("A simple polygon needs at least 3 vertices.")

    # Canonicalize orientation.
    # The model can theoretically learn from clockwise or counter-clockwise polygons, but canonicalizing
    # orientation removes one unnecessary source of variation
    if signed_area(vertices) < 0:
        vertices = torch.flip(vertices, dims=[0])

    center = polygon_centroid(vertices)
    centered = vertices - center

    bbox_min = centered.min(dim=0).values
    bbox_max = centered.max(dim=0).values
    bbox_wh = bbox_max - bbox_min

    # Use a bounding-box spatial extent rather than sqrt(area).
    # sqrt(area) is an area-equivalent scale and may leave very elongated
    # polygons with large normalized coordinate values.
    scale = torch.max(bbox_wh[0], bbox_wh[1]).clamp(min=eps)
    return centered / scale

def random_cyclic_shift(vertices: torch.Tensor) -> torch.Tensor:
    """Randomly rotate the start vertex.

    This is useful as data augmentation because the same polygon can be represented
    by starting at any boundary vertex.
    """
    v = vertices.shape[0]
    shift = random.randrange(v)
    return torch.roll(vertices, shifts=-shift, dims=0)

def canonicalize_polygon_order(vertices: torch.Tensor) -> torch.Tensor:
    """Remove duplicate closing vertex and force counter-clockwise order.

    Unlike normalize_polygon, this keeps the original translation and scale.
    That is useful at dataset level, where polygon centroids and relative
    geometry positions matter.
    """
    vertices = remove_duplicate_closing_vertex(vertices.float())
    if signed_area(vertices) < 0:
        vertices = torch.flip(vertices, dims=[0])
    return vertices

def polygon_perimeter(vertices: torch.Tensor) -> torch.Tensor:
    """Boundary length of a polygon without repeated closing vertex."""
    nxt = torch.roll(vertices, shifts=-1, dims=0)
    return torch.linalg.norm(nxt - vertices, dim=-1).sum()


def polygon_bbox_wh(vertices: torch.Tensor) -> torch.Tensor:
    """Bounding-box width and height."""
    mn = vertices.min(dim=0).values
    mx = vertices.max(dim=0).values
    return mx - mn


def polygon_summary(vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return centroid and raw scalar features for one polygon.

    Returns:
        centroid: [2]
        features: [5] = area, perimeter, bbox_width, bbox_height, num_vertices
    """
    vertices = remove_duplicate_closing_vertex(vertices)
    centroid = polygon_centroid(vertices)
    area = signed_area(vertices).abs()
    perimeter = polygon_perimeter(vertices)
    bbox_wh = polygon_bbox_wh(vertices)
    num_vertices = torch.tensor(float(vertices.shape[0]), dtype=torch.float32, device=vertices.device)
    features = torch.stack([area, perimeter, bbox_wh[0], bbox_wh[1], num_vertices])
    return centroid, features